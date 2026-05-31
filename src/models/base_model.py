import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import timm
import torchmetrics
from omegaconf import DictConfig
import logging
from src.models.loss import OrdinalLabelSmoothingLoss


logger = logging.getLogger(__name__)

class InflammationModel(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Extract optional kwargs for timm if defined in model config
        timm_kwargs = {}
        if 'drop_rate' in cfg['model']:
            timm_kwargs['drop_rate'] = cfg['model']['drop_rate']
        if 'drop_path_rate' in cfg['model']:
            timm_kwargs['drop_path_rate'] = cfg['model']['drop_path_rate']
        if 'attn_drop_rate' in cfg['model'] and 'vit' in cfg['model']['backbone']:
            timm_kwargs['attn_drop_rate'] = cfg['model']['attn_drop_rate']

        # Backbone: robustly retry when some TIMM backbones do not support
        # optional kwargs (e.g., DenseNet does not accept drop_path_rate).

        # --- MaxViT Special Handling: Classifier Head Replacement ---
        #
        # For timm's MaxViT models, the classifier 'head' is the final linear layer mapping features to class logits.
        # This layer is typically named 'head.fc', 'head.classifier', or sometimes just 'classifier', depending on the model implementation.
        #
        # Why is this needed? If you instantiate a pretrained MaxViT with num_classes=1000 (ImageNet),
        # the head matches the pretrained weights. If you set num_classes=5 (our use case), timm will create a new head,
        # but loading a checkpoint with a mismatched head shape will fail (state_dict mismatch).
        #
        # Solution: Always instantiate with the default head (num_classes=1000), then replace the classifier head
        # with a new nn.Linear that matches our required number of classes. This ensures compatibility with both
        # pretrained weights and our custom classification task.
        #
        # 'Head' in this context means the final fully connected (linear) layer that outputs the class logits.
        # For MaxViT, this is usually backbone.head.fc (timm convention), but can vary by model.
        if 'maxvit' in cfg['model']['backbone']:
            # Create model with default num_classes (pretrained head)
            backbone = self._create_backbone_with_safe_kwargs(
                backbone_name=cfg['model']['backbone'],
                num_classes=1000,  # default ImageNet head
                timm_kwargs=timm_kwargs,
            )
            # Replace classifier head if num_classes mismatch
            # Try common conventions: head.fc, head.classifier, classifier
            if hasattr(backbone, 'head') and hasattr(backbone.head, 'fc'):
                # timm MaxViT: backbone.head.fc is the classifier head
                in_features = backbone.head.fc.in_features
                backbone.head.fc = nn.Linear(in_features, cfg['num_classes'])
            elif hasattr(backbone, 'head') and hasattr(backbone.head, 'classifier'):
                # Some models: backbone.head.classifier
                in_features = backbone.head.classifier.in_features
                backbone.head.classifier = nn.Linear(in_features, cfg['num_classes'])
            else:
                # Fallback: try to set 'classifier' directly if present
                if hasattr(backbone, 'classifier') and hasattr(backbone.classifier, 'in_features'):
                    in_features = backbone.classifier.in_features
                    backbone.classifier = nn.Linear(in_features, cfg['num_classes'])
                else:
                    raise RuntimeError("MaxViT backbone: Unable to locate classifier head for replacement.")
            self.backbone = backbone
        else:
            # All other models: use standard logic (timm handles head creation)
            self.backbone = self._create_backbone_with_safe_kwargs(
                backbone_name=cfg['model']['backbone'],
                num_classes=cfg['num_classes'],
                timm_kwargs=timm_kwargs,
            )

        # Configure Loss Function (loss config lives under data.loss in base.yaml)
        loss_cfg = cfg.get('data', {}).get('loss', {'name': 'cross_entropy'})
        if loss_cfg.get('name') == 'ordinal_smoothing':
            smoothing = loss_cfg.get('smoothing_factor', 0.1)
            # We use 4 classes for smoothing (0-3), ignore class handled separately if needed
            self.loss_fn = OrdinalLabelSmoothingLoss(num_classes=4, smoothing=smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=cfg['ignore_class_index'])

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg['num_classes'])
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg['num_classes'])
        
        # Quadratic Weighted Kappa (only for 0-3 classes usually, but we can track it)
        # We will compute QWK on the 4 main classes for scientific relevance
        self.val_kappa = torchmetrics.CohenKappa(task="multiclass", num_classes=4, weights="quadratic")
        
        # Per-class metrics (for 4 main classes, excluding ignore)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=4, average=None)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=4, average=None)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=4, average=None)

        # Per-class specificity (TNR = true negative rate per class)
        self.val_specificity = torchmetrics.Specificity(task="multiclass", num_classes=4, average=None)

        # Matthews Correlation Coefficient (MCC) — robust single-number multi-class quality metric
        self.val_mcc = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=4)

        # Macro-averaged F1 (overall metric)
        self.val_macro_f1 = torchmetrics.F1Score(task="multiclass", num_classes=4, average="macro")
        
        self.ignore_index = cfg['ignore_class_index']
        
        # Flag to suppress overfitting warnings (e.g. during HPO)
        self.suppress_overfitting_warnings: bool = False

    def forward(self, x):
        return self.backbone(x)

    def _create_backbone_with_safe_kwargs(
        self,
        backbone_name: str,
        num_classes: int,
        timm_kwargs: dict,
    ) -> nn.Module:
        """Create TIMM backbone and drop unsupported kwargs automatically."""
        kwargs = dict(timm_kwargs)

        while True:
            try:
                return timm.create_model(
                    backbone_name,
                    pretrained=True,
                    num_classes=num_classes,
                    **kwargs,
                )
            except TypeError as exc:
                message = str(exc)
                marker = "unexpected keyword argument '"
                if marker not in message:
                    raise

                start = message.find(marker)
                end = message.find("'", start + len(marker))
                if start == -1 or end == -1:
                    raise

                bad_key = message[start + len(marker):end]
                if bad_key not in kwargs:
                    raise

                logger.warning(
                    "Backbone '%s' does not support timm kwarg '%s'. Retrying without it.",
                    backbone_name,
                    bad_key,
                )
                kwargs.pop(bad_key)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Use configured loss function (OrdinalLabelSmoothing or CrossEntropy)
        if isinstance(self.loss_fn, OrdinalLabelSmoothingLoss):
            mask = y != self.cfg['ignore_class_index']
            if mask.sum() > 0:
                loss = self.loss_fn(logits[mask, :4], y[mask])
            else:
                loss = torch.tensor(0.0, device=self.device)
        else:
            loss = self.loss_fn(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Loss calculation tailored to method
        if isinstance(self.loss_fn, OrdinalLabelSmoothingLoss):
            mask = y != self.cfg['ignore_class_index']
            if mask.sum() > 0:
                loss = self.loss_fn(logits[mask, :4], y[mask])
            else:
                loss = torch.tensor(0.0, device=self.device)
        else:
            loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        
        # --- Scientific Scoring Logic ---
        # 1. Filter out 'Ignore' class (Ground Truth) for Kappa calculation
        valid_mask = y != self.ignore_index
        if valid_mask.sum() > 0:
            y_valid = y[valid_mask]
            preds_valid = preds[valid_mask]
            
            # If prediction was 'Ignore' but GT was valid, we need to handle it.
            # For Kappa, we usually only compare valid classes. 
            # If model predicts 'Ignore' for a valid class, it's a wrong prediction.
            # We map predicted 'Ignore' (4) to a wrong class (e.g. 0) or handle it carefully.
            # Here, we just clamp to 0-3 for the metric to work, effectively penalizing it.
            preds_valid_clamped = torch.clamp(preds_valid, 0, 3)
            self.val_kappa(preds_valid_clamped, y_valid)
            
            # Update per-class metrics (only on valid samples)
            self.val_precision(preds_valid_clamped, y_valid)
            self.val_recall(preds_valid_clamped, y_valid)
            self.val_f1(preds_valid_clamped, y_valid)
            self.val_macro_f1(preds_valid_clamped, y_valid)
            self.val_specificity(preds_valid_clamped, y_valid)
            self.val_mcc(preds_valid_clamped, y_valid)
            
        # 2. Continuous Score Calculation (Weighted Sum)
        # Softmax over all 5 classes
        probs = F.softmax(logits, dim=1) # [B, 5]
        
        # Exclude Ignore col (index 4)
        probs_inflammation = probs[:, :4] # [B, 4]
        
        # Re-normalize to sum to 1
        sums = probs_inflammation.sum(dim=1, keepdim=True) + 1e-8
        probs_norm = probs_inflammation / sums
        
        # Expected Value: sum(p_i * i) for i in 0..3
        weights = torch.tensor([0, 1, 2, 3], device=self.device, dtype=torch.float)
        continuous_scores = (probs_norm * weights).sum(dim=1)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_kappa", self.val_kappa, prog_bar=True)
        self.log("val_macro_f1", self.val_macro_f1, prog_bar=True)
        
        # --- Per-Class Metrics (Log for later extraction) ---
        if valid_mask.sum() > 0:
            per_class_f1 = self.val_f1.compute()
            per_class_precision = self.val_precision.compute()
            per_class_recall = self.val_recall.compute()
            per_class_specificity = self.val_specificity.compute()
            
            # Log per-class F1 scores
            for i, f1_val in enumerate(per_class_f1):
                self.log(f"val_f1_class_{i}", f1_val, prog_bar=False)

            # Log per-class TPR (sensitivity = recall) and TNR (specificity)
            for i, (tpr, tnr) in enumerate(zip(per_class_recall, per_class_specificity)):
                self.log(f"val_tpr_class_{i}", tpr, prog_bar=False)
                self.log(f"val_tnr_class_{i}", tnr, prog_bar=False)

            # Balanced Accuracy = mean of per-class recall (TPR)
            balanced_acc = per_class_recall.mean()
            self.log("val_balanced_acc", balanced_acc, prog_bar=False)

            # Matthews Correlation Coefficient
            self.log("val_mcc", self.val_mcc, prog_bar=False)
        
        # --- Confidence Tracking ---
        # Store max probability as confidence
        max_probs = probs.max(dim=1)[0]
        
        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "scores": continuous_scores,
            "confidences": max_probs,
            "probabilities": probs,
        }

    def on_validation_epoch_end(self) -> None:
        """Log overfitting metrics once per epoch instead of per batch."""
        if not (hasattr(self, 'train_acc') and self.train_acc is not None):
            return

        train_acc_val: float = self.train_acc.compute().item()
        val_acc_val: float = self.val_acc.compute().item()
        acc_gap: float = train_acc_val - val_acc_val

        self.log("train_val_acc_gap", acc_gap, prog_bar=True)

        if self.suppress_overfitting_warnings:
            return

        if acc_gap > 0.15:
            logger.warning("Large train-val accuracy gap detected (%.3f) - Possible overfitting!", acc_gap)
        elif acc_gap > 0.10:
            logger.warning("Moderate train-val accuracy gap (%.3f) - Monitor closely", acc_gap)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler from config."""
        # Get optimizer config (with defaults)
        opt_cfg = self.cfg['training'].get('optimizer', {})
        opt_type = opt_cfg.get('type', 'adamw').lower()
        betas = tuple(opt_cfg.get('betas', [0.9, 0.999]))
        eps = float(opt_cfg.get('eps', 1e-8))
        
        # Create optimizer
        if opt_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=float(self.cfg['training']['learning_rate']), 
                weight_decay=float(self.cfg['training']['weight_decay']),
                betas=betas,
                eps=eps
            )
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=float(self.cfg['training']['learning_rate']), 
                weight_decay=float(self.cfg['training']['weight_decay']),
                betas=betas,
                eps=eps
            )
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=float(self.cfg['training']['learning_rate']), 
                weight_decay=float(self.cfg['training']['weight_decay']),
                momentum=betas[0]  # Use beta1 as momentum
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
        
        # Get scheduler config (with defaults)
        sched_cfg = self.cfg['training'].get('scheduler', {})
        sched_type = sched_cfg.get('type', 'reduce_on_plateau').lower()
        
        # Create learning rate scheduler
        if sched_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=float(sched_cfg.get('factor', 0.1)),
                patience=int(sched_cfg.get('patience', 5)),
                min_lr=float(sched_cfg.get('min_lr', 1e-7))
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }
        elif sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(sched_cfg.get('T_max', self.cfg['training']['max_epochs'])),
                eta_min=float(sched_cfg.get('eta_min', 1e-7))
            )
            return [optimizer], [scheduler]
        elif sched_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sched_cfg.get('step_size', 10)),
                gamma=float(sched_cfg.get('gamma', 0.1))
            )
            return [optimizer], [scheduler]
        elif sched_type == 'none':
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")
