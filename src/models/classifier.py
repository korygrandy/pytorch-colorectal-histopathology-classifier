"""
PyTorch model architectures for colorectal histopathology classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ColorectalClassifier(nn.Module):
    """
    Transfer learning based classifier for colorectal histopathology images.
    Uses pre-trained ResNet as backbone with custom classification head.
    """
    
    def __init__(
        self,
        num_classes: int = 8,
        architecture: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Number of output classes
            architecture: Backbone architecture (resnet18, resnet34, resnet50, resnet101)
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
        """
        super(ColorectalClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Load backbone
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        elif architecture == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """
        Get number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(
    num_classes: int = 8,
    architecture: str = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.5,
    device: Optional[str] = None
) -> ColorectalClassifier:
    """
    Factory function to create a colorectal classifier model.
    
    Args:
        num_classes: Number of output classes
        architecture: Backbone architecture
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
        device: Device to place model on (cuda/cpu)
        
    Returns:
        ColorectalClassifier model
    """
    model = ColorectalClassifier(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=pretrained,
        dropout=dropout
    )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    return model


def load_checkpoint(
    checkpoint_path: str,
    num_classes: int = 8,
    architecture: str = "resnet50",
    device: Optional[str] = None
) -> ColorectalClassifier:
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_classes: Number of output classes
        architecture: Backbone architecture
        device: Device to place model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ColorectalClassifier(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model
