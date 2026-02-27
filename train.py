"""
Training script for colorectal histopathology classifier.

Dedicated to the memory of those lost to colon cancer.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import create_dataloaders
from src.models import create_model
from src.utils import (
    load_config,
    save_checkpoint,
    AverageMeter,
    compute_accuracy,
    evaluate_model,
    plot_confusion_matrix,
    plot_training_history
)


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    print_freq: int = 10
) -> tuple:
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        acc = compute_accuracy(outputs, labels)
        
        # Update meters
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
    
    return losses.avg, accuracies.avg


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            acc = compute_accuracy(outputs, labels)
            
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
    
    return losses.avg, accuracies.avg


def main():
    parser = argparse.ArgumentParser(
        description='Train colorectal histopathology classifier'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Override data directory from config'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data directory if provided
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        image_size=config['data']['image_size'],
        classes=config['data']['classes']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {config['model']['architecture']} model...")
    model = create_model(
        num_classes=config['model']['num_classes'],
        architecture=config['model']['architecture'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        device=device
    )
    
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Learning rate scheduler
    if config['training']['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['step_size'],
            gamma=config['training']['gamma']
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    
    # Tensorboard writer
    if config['logging']['tensorboard']:
        writer = SummaryWriter(config['logging']['log_dir'])
    else:
        writer = None
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('accuracy', 0.0)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch + 1, config['logging']['print_freq']
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Tensorboard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(config['training']['save_dir'], 'best_model.pth'),
                scheduler
            )
        else:
            patience_counter += 1
        
        # Save regular checkpoint
        if not config['training']['save_best_only']:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(config['training']['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth'),
                scheduler
            )
        
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(
        history,
        save_path=os.path.join(config['logging']['log_dir'], 'training_history.png')
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_model(
        model, test_loader, criterion, device,
        config['data']['classes']
    )
    
    print(f"\nTest Loss: {test_results['loss']:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print("\nClassification Report:")
    print("-" * 80)
    
    # Print classification report
    for class_name in config['data']['classes']:
        metrics = test_results['classification_report'][class_name]
        print(f"{class_name:20s} - Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        config['data']['classes'],
        save_path=os.path.join(config['logging']['log_dir'], 'confusion_matrix.png'),
        title='Test Set Confusion Matrix'
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {test_results['accuracy']:.2f}%")
    print(f"Model saved to: {config['training']['save_dir']}/best_model.pth")
    print("=" * 80)


if __name__ == '__main__':
    main()
