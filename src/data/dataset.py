"""
Dataset handlers for colorectal histopathology images.
"""

import os
from typing import Tuple, Optional, Callable, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class ColorectalHistologyDataset(Dataset):
    """
    Dataset class for colorectal histopathology images.
    
    Expected directory structure:
        data_dir/
            class1/
                image1.tif
                image2.tif
                ...
            class2/
                image1.tif
                ...
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        classes: Optional[List[str]] = None
    ):
        """
        Args:
            root_dir: Root directory containing class subdirectories
            transform: Optional transform to apply to images
            classes: Optional list of class names (if None, auto-detected)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Auto-detect classes if not provided
        if classes is None:
            self.classes = sorted([d for d in os.listdir(root_dir) 
                                 if os.path.isdir(os.path.join(root_dir, d))])
        else:
            self.classes = classes
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_class_distribution(self) -> dict:
        """Get the distribution of samples across classes."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1
        return distribution


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    """
    Get image transforms for training or validation.
    
    Args:
        image_size: Target image size
        is_training: Whether transforms are for training (includes augmentation)
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Composed transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    image_size: int = 224,
    classes: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing image data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        image_size: Target image size
        classes: Optional list of class names
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets with appropriate transforms
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    # Full dataset
    full_dataset = ColorectalHistologyDataset(data_dir, transform=None, classes=classes)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create subset datasets
    train_dataset = ColorectalHistologyDataset(data_dir, transform=train_transform, classes=classes)
    val_dataset = ColorectalHistologyDataset(data_dir, transform=val_transform, classes=classes)
    test_dataset = ColorectalHistologyDataset(data_dir, transform=val_transform, classes=classes)
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
