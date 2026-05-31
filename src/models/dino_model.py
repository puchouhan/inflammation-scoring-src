import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import timm
from torchmetrics import Accuracy

class DINO(L.LightningModule):
    """
    DINO (Self-Distillation with No Labels) Implementation
    Paper: "Emerging Properties in Self-Supervised Vision Transformers" (Caron et al., 2021)
    
    Key differences from SimCLR:
    - Uses momentum teacher network (EMA)
    - Centering to avoid collapse
    - Only teacher sees global views, student sees local crops
    """
    def __init__(
        self, 
        backbone_name: str = "vit_tiny_patch16_224",
        out_dim: int = 65536,
        use_bn_in_head: bool = False,
        norm_last_layer: bool = True,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        lr: float = 5e-4,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        momentum_teacher: float = 0.996,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Student encoder
        self.student_backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        
        # Get embedding dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            embed_dim = self.student_backbone(dummy_input).shape[1]
        
        # Student projection head
        self.student_head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Teacher encoder (no gradient)
        self.teacher_backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.teacher_head = DINOHead(
            in_dim=embed_dim,
            out_dim=out_dim,
            use_bn=use_bn_in_head,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
        
        # Copy student to teacher
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        
        # Teacher has no gradients
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
            
        # Center for teacher output
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Hyperparameters
        self.lr = lr
        self.momentum_teacher = momentum_teacher
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.student_backbone(x)
    
    @torch.no_grad()
    def update_teacher(self):
        """
        Exponential Moving Average update for teacher
        """
        for param_student, param_teacher in zip(
            list(self.student_backbone.parameters()) + list(self.student_head.parameters()),
            list(self.teacher_backbone.parameters()) + list(self.teacher_head.parameters())
        ):
            param_teacher.data.mul_(self.momentum_teacher).add_(
                param_student.data, alpha=1 - self.momentum_teacher
            )

    def training_step(self, batch, batch_idx):
        # batch contains (global_crops, local_crops)
        # global_crops: 2 views at 224x224
        # local_crops: 6+ views at smaller resolution
        global_crops, local_crops = batch
        
        # Teacher processes only global crops
        teacher_output = []
        with torch.no_grad():
            for img in global_crops:
                h = self.teacher_backbone(img)
                out = self.teacher_head(h)
                # Center + sharpen
                out = F.softmax((out - self.center) / self.teacher_temp, dim=-1)
                teacher_output.append(out)
        teacher_output = torch.stack(teacher_output)
        
        # Student processes all crops
        student_output = []
        for img in global_crops + local_crops:
            h = self.student_backbone(img)
            out = self.student_head(h)
            student_output.append(out)
        student_output = torch.stack(student_output)
        
        # Compute loss
        loss = self.dino_loss(student_output, teacher_output)
        
        # Update center
        self.update_center(teacher_output)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def dino_loss(self, student_output, teacher_output):
        """
        Cross-entropy between teacher and student with temperature
        """
        n_global = teacher_output.shape[0]
        student_out = student_output / 0.1  # Student temperature
        student_out = F.log_softmax(student_out, dim=-1)
        
        loss = 0
        n_loss_terms = 0
        for t_idx in range(n_global):
            for s_idx in range(len(student_output)):
                if t_idx == s_idx:
                    continue  # Don't compare same views
                loss += -torch.sum(teacher_output[t_idx] * student_out[s_idx], dim=-1).mean()
                n_loss_terms += 1
        
        return loss / n_loss_terms

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output
        """
        batch_center = torch.mean(teacher_output, dim=[0, 1], keepdim=True)
        self.center = self.center * 0.9 + batch_center * 0.1

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update teacher after each batch
        self.update_teacher()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.student_backbone.parameters()) + list(self.student_head.parameters()),
            lr=self.lr,
            weight_decay=0.04
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_epochs
        )
        
        return [optimizer], [scheduler]


class DINOHead(nn.Module):
    """
    Projection head for DINO
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
                
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
