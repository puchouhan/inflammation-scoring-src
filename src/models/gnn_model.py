import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torchmetrics import Accuracy, CohenKappa

class GNNClassifier(L.LightningModule):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, lr: float = 1e-3, model_type: str = "GCN"):
        super().__init__()
        self.save_hyperparameters()
        
        if model_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == "GAT":
            self.conv1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
            # Input to conv2 is hidden * heads
            self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
            
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        self.lr = lr
        self.model_type = model_type
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_kappa = CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic")

    def forward(self, x, edge_index, batch=None):
        # x: Node features [Num_Nodes, In_Channels]
        # edge_index: [2, Num_Edges]
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Node Classification: each node gets its own prediction
        # No pooling needed for node-level tasks
        
        out = self.classifier(x)
        return out

    def training_step(self, batch, batch_idx):
        # Batch is a PyG Data object with node-level labels
        out = self(batch.x, batch.edge_index)
        
        # Node classification: batch.y is [Num_Nodes] with labels for each patch
        loss = F.cross_entropy(out, batch.y)
        
        preds = out.argmax(dim=1)
        self.train_acc(preds, batch.y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index)
        loss = F.cross_entropy(out, batch.y)
        
        preds = out.argmax(dim=1)
        self.val_acc(preds, batch.y)
        self.val_kappa(preds, batch.y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_kappa", self.val_kappa, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
