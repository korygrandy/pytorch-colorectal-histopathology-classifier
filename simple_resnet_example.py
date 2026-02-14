"""
Simple example of defining and using a ResNet-based Neural Network for 
colorectal histopathology classification.

This is a simplified, standalone example that demonstrates the core concepts.
For the full implementation with training, see train.py and src/models/classifier.py
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class SimpleResNetClassifier(nn.Module):
    """
    Simple ResNet-based classifier for colorectal histopathology images.
    
    This is a straightforward example showing how to:
    1. Use a pretrained ResNet backbone
    2. Replace the final layer for our specific task
    3. Add dropout for regularization
    """
    
    def __init__(self, num_classes=8, pretrained=True):
        """
        Args:
            num_classes: Number of tissue types to classify (default: 8)
            pretrained: Use ImageNet pretrained weights (default: True)
        """
        super(SimpleResNetClassifier, self).__init__()
        
        # Load pretrained ResNet50
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        # Get the number of features from the last layer
        num_features = self.resnet.fc.in_features  # 2048 for ResNet50
        
        # Replace the final fully connected layer
        # Original: Linear(2048, 1000) for ImageNet
        # New: Custom classifier for our 8 tissue types
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),                    # Dropout for regularization
            nn.Linear(num_features, 512),       # Reduce dimensions
            nn.ReLU(),                          # Activation
            nn.Dropout(0.3),                    # More dropout
            nn.Linear(512, num_classes)         # Final classification layer
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.resnet(x)


def example_usage():
    """
    Example showing how to create and use the model.
    """
    print("=" * 80)
    print("Simple ResNet-based Neural Network for Colorectal Histopathology")
    print("=" * 80)
    print()
    
    # Create the model
    print("1. Creating model...")
    model = SimpleResNetClassifier(num_classes=8, pretrained=False)
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Show model architecture (just the custom classifier head)
    print("2. Custom classifier head (replaces ResNet's fc layer):")
    print(model.resnet.fc)
    print()
    
    # Example forward pass
    print("3. Testing forward pass...")
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    print(f"   Input shape: {dummy_images.shape}")
    
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output logits (first sample): {outputs[0]}")
    print()
    
    # Convert to probabilities
    probabilities = torch.softmax(outputs, dim=1)
    print(f"4. Converting to probabilities:")
    print(f"   Probabilities (first sample): {probabilities[0]}")
    print(f"   Sum of probabilities: {probabilities[0].sum():.4f} (should be 1.0)")
    print()
    
    # Get predictions
    predicted_classes = outputs.argmax(dim=1)
    print(f"5. Predicted classes: {predicted_classes}")
    print()
    
    print("=" * 80)
    print("✅ Example complete!")
    print("=" * 80)
    print()
    print("The 8 tissue classes are:")
    classes = [
        "0: Adipose tissue",
        "1: Empty/Background",
        "2: Debris",
        "3: Lymphocytes (immune cells)",
        "4: Mucus",
        "5: Smooth muscle",
        "6: Normal colon mucosa",
        "7: Tumor epithelium (colorectal adenocarcinoma)"
    ]
    for cls in classes:
        print(f"  {cls}")
    print()
    print("Next steps:")
    print("  - Download the dataset: python download_dataset.py")
    print("  - Train the model: python train.py")
    print("  - Make predictions: python predict.py --image path/to/image.tif")
    print()


if __name__ == '__main__':
    example_usage()
