"""
Model Factory for dynamically loading architectures.
Maps model names to their corresponding classes and initialization logic.
"""
from typing import Dict, Any, Callable, TypedDict
import torch.nn as nn

# Import all model classes
from src.models.base_model import InflammationModel
from src.models.gnn_model import GNNClassifier
from src.models.ssl_model import SimCLR
from src.models.dino_model import DINO


ModelInfo = TypedDict(
    "ModelInfo",
    {
        "class": Callable[..., nn.Module],
        "type": str,
        "description": str,
    },
)


class ModelFactory:
    """Factory for creating models dynamically based on configuration."""
    
    # Map model names to their classes and special requirements
    MODEL_REGISTRY: Dict[str, ModelInfo] = {
        "densenet": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "DenseNet - Baseline CNN",
        },
        "efficientnetv2": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "EfficientNetV2 (TIMM backbone)",
        },
        "regnety": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "RegNetY",
        },
        "convnext": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "ConvNeXt - State-of-the-art CNN (2022)",
        },
        "swin": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "Swin Transformer",
        },
        "maxvit": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "MaxViT - Hybrid CNN+Transformer (optimal for cell counting)",
        },
        "vit": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "Vision Transformer (Pure Attention)",
        },
        "convit": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "ConViT (Convolutional Vision Transformer) - Hybrid CNN+Transformer",
        },
        "tnt": {
            "class": InflammationModel,
            "type": "supervised",
            "description": "TNT (Transformer in Transformer) - Nested Transformer Architecture",
        },
        "gnn": {
            "class": GNNClassifier,
            "type": "graph",
            "description": "Graph Neural Network (requires graph data)",
        },
        "simclr": {
            "class": SimCLR,
            "type": "self_supervised",
            "description": "SimCLR Self-Supervised Learning",
        },
        "dino": {
            "class": DINO,
            "type": "self_supervised",
            "description": "DINO Self-Supervised Learning",
        },
    }
    
    # Map model names to TIMM backbone names
    BACKBONE_MAP: Dict[str, str] = {
        "densenet": "densenet121",
        "efficientnetv2": "efficientnetv2_rw_s",
        "regnety": "regnety_002",
        "convnext": "convnext_tiny",
        "swin": "swin_tiny_patch4_window7_224",
        "vit": "vit_small_patch16_224",
        "maxvit": "maxvit_tiny_224",
        "convit": "convit_tiny",
        "tnt": "tnt_s_patch16_224",
    }
    
    @classmethod
    def create_model(cls, model_name: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create a model instance based on name and config.
        
        Args:
            model_name: Name of the model (e.g., "efficientnetv2", "vit")
            config: Full configuration dictionary
            
        Returns:
            Instantiated PyTorch Lightning model
            
        Raises:
            ValueError: If model_name is not recognized
        """
        if model_name not in cls.MODEL_REGISTRY:
            available = ", ".join(cls.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {available}"
            )
        
        model_info = cls.MODEL_REGISTRY[model_name]
        model_class = model_info["class"]
        model_type = model_info["type"]
        
        # Clone config to avoid mutation
        model_config = config.copy()
        
        # Handle supervised models (InflammationModel)
        if model_type == "supervised":
            # Override backbone if model-specific
            if model_name in cls.BACKBONE_MAP:
                model_config["training"]["backbone"] = cls.BACKBONE_MAP[model_name]
            
            return model_class(model_config)
        
        # Handle GNN models
        elif model_type == "graph":
            return model_class(
                in_channels=512,  # Feature dimension from feature extractor
                hidden_channels=256,
                num_classes=config["num_classes"],
                lr=config["training"]["learning_rate"],
                model_type="GCN",  # or "GAT"
            )
        
        # Handle self-supervised models
        elif model_type == "self_supervised":
            if model_name == "simclr":
                return model_class(
                    backbone_name="resnet18",
                    hidden_dim=128,
                    lr=config["training"]["learning_rate"],
                    temperature=0.07,
                    max_epochs=config["training"]["max_epochs"],
                )
            elif model_name == "dino":
                return model_class(
                    backbone_name="vit_tiny_patch16_224",
                    out_dim=65536,
                    lr=config["training"]["learning_rate"],
                    max_epochs=config["training"]["max_epochs"],
                )
        
        raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """Return a dict of available models and their descriptions."""
        return {
            name: info["description"] 
            for name, info in cls.MODEL_REGISTRY.items()
        }
    
    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        """Get the type of a model (supervised, graph, self_supervised)."""
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found")
        return cls.MODEL_REGISTRY[model_name]["type"]
    
    @classmethod
    def is_supervised(cls, model_name: str) -> bool:
        """Check if a model is supervised (needs labels)."""
        return cls.get_model_type(model_name) == "supervised"
    
    @classmethod
    def is_graph_based(cls, model_name: str) -> bool:
        """Check if a model requires graph data structure."""
        return cls.get_model_type(model_name) == "graph"
    
    @classmethod
    def is_self_supervised(cls, model_name: str) -> bool:
        """Check if a model is self-supervised (no labels needed)."""
        return cls.get_model_type(model_name) == "self_supervised"
