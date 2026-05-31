import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import timm
from torchmetrics import Accuracy

class SimCLR(L.LightningModule):
    def __init__(self, backbone_name: str = "resnet18", hidden_dim: int = 128, lr: float = 1e-3, temperature: float = 0.07, max_epochs: int = 100):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder
        self.encoder = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        
        # Get output dim of encoder
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out_dim = self.encoder(dummy_input).shape[1]
            
        # Projection Head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, hidden_dim)
        )
        
        self.temperature = temperature
        self.lr = lr
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        # Batch contains (img1, img2) from SimCLRDataset
        # No labels for self-supervised learning
        x1, x2 = batch
        
        # Encode
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        loss = self.nt_xent_loss(z1, z2)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def nt_xent_loss(self, z1, z2):
        batch_size = z1.shape[0]
        
        # Concatenate
        z = torch.cat([z1, z2], dim=0)
        
        # Similarity matrix
        z = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        
        # Positive pairs
        # z1[i] matches z2[i] -> index i and i+batch_size
        pos_mask = torch.zeros_like(sim_matrix).bool()
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True
            
        # Loss
        # We want sim(z1, z2) to be high compared to sim(z1, z_others)
        # This is essentially CrossEntropy
        
        # Simplified implementation using CrossEntropy
        # Target for i is i+batch_size
        targets = torch.arange(2 * batch_size, device=z.device)
        targets[:batch_size] = targets[:batch_size] + batch_size
        targets[batch_size:] = targets[batch_size:] - batch_size
        
        return F.cross_entropy(sim_matrix, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]
