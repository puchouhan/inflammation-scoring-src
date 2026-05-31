import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import os

class PatchGraphDataset(Dataset):
    def __init__(self, root, feature_dir, dataframe, k=8):
        """
        Args:
            root: Root directory for saving processed graphs
            feature_dir: Directory where .pt feature files are stored (one per image)
            dataframe: The main dataframe with 'filepath', 'animal_id', 'x', 'y', 'label'
            k: Number of neighbors for KNN graph
        """
        self.feature_dir = Path(feature_dir)
        self.df = dataframe
        self.k = k
        self.root = root
        super().__init__(root)
        
        # Group by Animal ID (or Slide ID) to form graphs
        # We assume one graph per Animal for now
        self.groups = list(self.df.groupby('animal_id'))
        
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.groups))]

    def process(self):
        # This is called if processed files don't exist
        # We iterate over animals, load all their patch features, build graph, save.
        
        for i, (animal_id, group) in enumerate(tqdm(self.groups, desc="Building Graphs")):
            # 1. Load Features & Coords
            node_features = []
            node_labels = []
            coords = []
            
            for _, row in group.iterrows():
                # Feature filename: we assume features are saved as {filepath_stem}.pt
                # Original path: training/0/Study_Animal_...png
                # Feature path: features/Study_Animal_...pt
                stem = Path(row['filepath']).stem
                feat_path = self.feature_dir / f"{stem}.pt"
                
                if feat_path.exists():
                    feat = torch.load(feat_path, map_location='cpu')
                    node_features.append(feat)
                    node_labels.append(row['label'])
                    coords.append([row['x'], row['y']])
                else:
                    # Skip missing features
                    pass
            
            if not node_features:
                continue
                
            x = torch.stack(node_features) # [Num_Nodes, Feature_Dim]
            y = torch.tensor(node_labels, dtype=torch.long) # [Num_Nodes]
            pos = torch.tensor(coords, dtype=torch.float)
            
            # 2. Build Edges (KNN based on spatial coords)
            # kneighbors_graph returns sparse matrix
            adj = kneighbors_graph(pos, self.k, mode='connectivity', include_self=False)
            edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
            
            # 3. Create Data Object
            data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
            
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def get_graph_dataloaders(config):
    """
    Create graph dataloaders for training and validation.
    
    Args:
        config: Configuration dictionary containing data paths and settings
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Extract paths from config
    feature_dir = config.get('data', {}).get('feature_dir', 'features/')
    train_csv = config.get('data', {}).get('train_csv', 'dataset/training/annotations.csv')
    val_csv = config.get('data', {}).get('val_csv', 'dataset/val/annotations.csv')
    batch_size = config.get('data', {}).get('batch_size', 32)
    k_neighbors = config.get('data', {}).get('k_neighbors', 8)
    
    # Load dataframes
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Create datasets
    train_dataset = PatchGraphDataset(
        root='data/processed/graphs/train',
        feature_dir=feature_dir,
        dataframe=train_df,
        k=k_neighbors
    )
    
    val_dataset = PatchGraphDataset(
        root='data/processed/graphs/val',
        feature_dir=feature_dir,
        dataframe=val_df,
        k=k_neighbors
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('data', {}).get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('data', {}).get('num_workers', 4)
    )
    
    return train_loader, val_loader
