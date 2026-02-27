"""
Evaluation script for colorectal histopathology classifier.
Evaluate a trained model on test data.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import load_checkpoint
from src.data import ColorectalHistologyDataset, get_transforms
from src.utils import load_config, evaluate_model, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate colorectal histopathology classifier'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation',
        help='Directory to save evaluation results'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_checkpoint(
        args.checkpoint,
        num_classes=config['model']['num_classes'],
        architecture=config['model']['architecture'],
        device=device
    )
    
    print(f"Model architecture: {config['model']['architecture']}")
    print(f"Number of classes: {config['model']['num_classes']}")
    
    # Create dataset and dataloader
    print(f"\nLoading test data from {args.data_dir}...")
    transform = get_transforms(
        image_size=config['data']['image_size'],
        is_training=False,
        mean=config['augmentation']['normalize']['mean'],
        std=config['augmentation']['normalize']['std']
    )
    
    test_dataset = ColorectalHistologyDataset(
        root_dir=args.data_dir,
        transform=transform,
        classes=config['data']['classes']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Print class distribution
    distribution = test_dataset.get_class_distribution()
    print("\nClass distribution:")
    for class_name, count in distribution.items():
        print(f"  {class_name:20s}: {count:5d} samples")
    
    # Evaluate model
    print("\n" + "=" * 80)
    print("Evaluating model...")
    print("=" * 80 + "\n")
    
    criterion = nn.CrossEntropyLoss()
    
    results = evaluate_model(
        model, test_loader, criterion, device,
        config['data']['classes']
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print results
    print(f"\nTest Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print("\n" + "=" * 80)
    print("Classification Report")
    print("=" * 80)
    print(f"{'Class':20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-" * 80)
    
    for class_name in config['data']['classes']:
        metrics = results['classification_report'][class_name]
        print(f"{class_name:20s} {metrics['precision']:>10.3f} "
              f"{metrics['recall']:>10.3f} {metrics['f1-score']:>10.3f} "
              f"{int(metrics['support']):>10d}")
    
    print("-" * 80)
    overall = results['classification_report']['weighted avg']
    print(f"{'Weighted Avg':20s} {overall['precision']:>10.3f} "
          f"{overall['recall']:>10.3f} {overall['f1-score']:>10.3f}")
    print("=" * 80)
    
    # Plot and save confusion matrix
    print("\nGenerating confusion matrix...")
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        results['confusion_matrix'],
        config['data']['classes'],
        save_path=cm_path,
        title='Confusion Matrix - Test Set'
    )
    
    # Save detailed results to file
    results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Colorectal Histopathology Classifier - Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Test Data: {args.data_dir}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n\n")
        f.write(f"Test Loss: {results['loss']:.4f}\n")
        f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n\n")
        f.write("=" * 80 + "\n")
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Class':20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}\n")
        f.write("-" * 80 + "\n")
        
        for class_name in config['data']['classes']:
            metrics = results['classification_report'][class_name]
            f.write(f"{class_name:20s} {metrics['precision']:>10.3f} "
                   f"{metrics['recall']:>10.3f} {metrics['f1-score']:>10.3f} "
                   f"{int(metrics['support']):>10d}\n")
        
        f.write("-" * 80 + "\n")
        overall = results['classification_report']['weighted avg']
        f.write(f"{'Weighted Avg':20s} {overall['precision']:>10.3f} "
               f"{overall['recall']:>10.3f} {overall['f1-score']:>10.3f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
