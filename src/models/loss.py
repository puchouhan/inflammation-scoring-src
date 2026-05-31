import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class OrdinalLabelSmoothingLoss(nn.Module):
    """Loss function for ordinal regression using label smoothing.
    
    Instead of hard one-hot encoding [0, 0, 1, 0], this loss distributes 
    a small amount of probability mass (epsilon) to the immediate neighbors.
    This reflects the ordinal nature of inflammation scoring (Score 1 is closer 
    to Score 2 than Score 0 is).
    
    Structure:
    - Target class k gets probability: 1 - epsilon
    - Neighbor k-1 gets: epsilon / 2 (if exists)
    - Neighbor k+1 gets: epsilon / 2 (if exists)
    - Mass is renormalized if a neighbor doesn't exist (edges 0 and 3).
    """
    
    def __init__(self, num_classes: int = 4, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss.
        
        Args:
            pred: Logits from model [Batch, Classes]
            target: Integer targets [Batch]
            
        Returns:
            Scalar loss value
        """
        batch_size = pred.size(0)
        
        # Create soft targets
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            
            # Scatter main confidence to correct class
            # Ensure target is on the same device as pred
            target = target.to(pred.device)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
            # Distribute smoothing to neighbors
            # Note: This loop is efficient enough for small batch sizes/class counts
            for i in range(batch_size):
                t = target[i].item()
                neighbor_prob = self.smoothing / 2.0
                
                # Check neighbors
                has_left = (t > 0)
                has_right = (t < self.num_classes - 1)
                
                if has_left and has_right:
                    true_dist[i, t-1] += neighbor_prob
                    true_dist[i, t+1] += neighbor_prob
                elif has_left:
                    # At edge (e.g. class 3), give all smoothing to left
                    true_dist[i, t-1] += self.smoothing
                elif has_right:
                    # At edge (e.g. class 0), give all smoothing to right
                    true_dist[i, t+1] += self.smoothing
                    
        # KL Divergence / Cross Entropy with soft targets
        log_probs = F.log_softmax(pred, dim=1)
        # We use SUM reduction divided by batch size manually to match CE behavior
        return torch.sum(-true_dist * log_probs) / batch_size
