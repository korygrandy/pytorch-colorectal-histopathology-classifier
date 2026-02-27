"""
Inference script for colorectal histopathology classifier.
Make predictions on new images.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.models import load_checkpoint
from src.data import get_transforms
from src.utils import load_config


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform: callable,
    device: str,
    class_names: list,
    top_k: int = 3
) -> dict:
    """
    Predict the class of a single image.
    
    Args:
        model: Trained model
        image_path: Path to image
        transform: Image transformation pipeline
        device: Device to run on
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get top k predictions
    probs, indices = torch.topk(probabilities, top_k)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    predictions = []
    for i in range(top_k):
        predictions.append({
            'class': class_names[indices[i]],
            'probability': float(probs[i]),
            'confidence': float(probs[i] * 100)
        })
    
    return {
        'top_prediction': predictions[0]['class'],
        'top_confidence': predictions[0]['confidence'],
        'all_predictions': predictions,
        'original_image': original_image
    }


def visualize_prediction(
    result: dict,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize prediction results.
    
    Args:
        result: Prediction result dictionary
        save_path: Optional path to save visualization
        show: Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Show original image
    ax1.imshow(result['original_image'])
    ax1.axis('off')
    ax1.set_title(f"Prediction: {result['top_prediction']}\n"
                 f"Confidence: {result['top_confidence']:.2f}%",
                 fontsize=12, fontweight='bold')
    
    # Show probability distribution
    classes = [pred['class'] for pred in result['all_predictions']]
    probs = [pred['probability'] for pred in result['all_predictions']]
    
    colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(classes))]
    ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Probability', fontsize=11)
    ax2.set_title('Top Predictions', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # Add probability values
    for i, (cls, prob) in enumerate(zip(classes, probs)):
        ax2.text(prob + 0.01, i, f'{prob:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Make predictions on colorectal histopathology images'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file or directory of images'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='predictions',
        help='Directory to save prediction results'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization'
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = config['inference']['checkpoint_path']
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_checkpoint(
        checkpoint_path,
        num_classes=config['model']['num_classes'],
        architecture=config['model']['architecture'],
        device=device
    )
    
    # Get transforms
    transform = get_transforms(
        image_size=config['data']['image_size'],
        is_training=False,
        mean=config['augmentation']['normalize']['mean'],
        std=config['augmentation']['normalize']['std']
    )
    
    class_names = config['data']['classes']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if input is file or directory
    if os.path.isfile(args.image):
        image_paths = [args.image]
    elif os.path.isdir(args.image):
        image_paths = [
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        ]
    else:
        raise ValueError(f"Invalid path: {args.image}")
    
    print(f"\nProcessing {len(image_paths)} image(s)...\n")
    
    # Process each image
    results = []
    for img_path in image_paths:
        print(f"Predicting: {os.path.basename(img_path)}")
        
        try:
            result = predict_image(
                model, img_path, transform, device,
                class_names, args.top_k
            )
            
            print(f"  -> {result['top_prediction']} "
                  f"({result['top_confidence']:.2f}% confidence)")
            
            # Print all top-k predictions
            for i, pred in enumerate(result['all_predictions'], 1):
                if i > 1:  # Skip first one as it's already printed
                    print(f"     {i}. {pred['class']}: {pred['confidence']:.2f}%")
            
            print()
            
            results.append({
                'image': os.path.basename(img_path),
                'prediction': result['top_prediction'],
                'confidence': result['top_confidence']
            })
            
            # Visualize if not disabled
            if not args.no_viz:
                viz_path = os.path.join(
                    args.output_dir,
                    f"{os.path.splitext(os.path.basename(img_path))[0]}_prediction.png"
                )
                visualize_prediction(result, save_path=viz_path, show=False)
        
        except Exception as e:
            print(f"  Error processing {img_path}: {str(e)}\n")
            continue
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'predictions_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Colorectal Histopathology Predictions\n")
        f.write("=" * 80 + "\n\n")
        for res in results:
            f.write(f"{res['image']}: {res['prediction']} "
                   f"({res['confidence']:.2f}% confidence)\n")
    
    print(f"\nPredictions complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
