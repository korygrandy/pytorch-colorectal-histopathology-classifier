"""
Data preparation helper script.
Shows example code for organizing data into the required structure.
"""

import os
import shutil
from pathlib import Path


def prepare_kather_dataset(source_dir: str, output_dir: str):
    """
    Prepare Kather dataset into the required directory structure.
    
    This is an example function showing how to organize the Kather
    colorectal histology dataset into the format expected by the classifier.
    
    Args:
        source_dir: Directory containing the original dataset
        output_dir: Directory where organized data will be saved
    """
    # Kather dataset class mapping
    # Adjust these based on your actual dataset folder names
    class_mapping = {
        '01_TUMOR': 'tumor',
        '02_STROMA': 'stroma',
        '03_COMPLEX': 'complex',
        '04_LYMPHO': 'lympho',
        '05_DEBRIS': 'mucosa',
        '06_MUCOSA': 'muscle',
        '07_ADIPOSE': 'normal',
        '08_EMPTY': 'tumor_epithelium'
    }
    
    print("Preparing dataset structure...")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each class
    for original_name, target_name in class_mapping.items():
        source_class_dir = os.path.join(source_dir, original_name)
        target_class_dir = os.path.join(output_dir, target_name)
        
        if not os.path.exists(source_class_dir):
            print(f"Warning: {source_class_dir} not found, skipping...")
            continue
        
        # Create target directory
        os.makedirs(target_class_dir, exist_ok=True)
        
        # Copy images
        image_files = list(Path(source_class_dir).glob('*.tif')) + \
                     list(Path(source_class_dir).glob('*.jpg')) + \
                     list(Path(source_class_dir).glob('*.png'))
        
        print(f"\nProcessing {original_name} -> {target_name}")
        print(f"  Found {len(image_files)} images")
        
        for img_file in image_files:
            target_file = os.path.join(target_class_dir, img_file.name)
            shutil.copy2(img_file, target_file)
        
        print(f"  Copied to {target_class_dir}")
    
    print("\nDataset preparation complete!")
    print(f"Organized data saved to: {output_dir}")


def create_sample_dataset_info():
    """
    Print information about creating a sample dataset for testing.
    """
    info = """
    ================================================================================
    Creating a Sample Dataset for Testing
    ================================================================================
    
    To test the classifier, you need to organize your histopathology images into
    the following directory structure:
    
    data/colorectal_histology/
    ├── tumor/
    │   ├── image1.tif
    │   ├── image2.tif
    │   └── ...
    ├── stroma/
    ├── complex/
    ├── lympho/
    ├── mucosa/
    ├── muscle/
    ├── normal/
    └── tumor_epithelium/
    
    Each subdirectory should contain images of that specific tissue type.
    
    ================================================================================
    Recommended Datasets:
    ================================================================================
    
    1. Kather Colorectal Histology Dataset
       - 5,000 histological images (150 x 150 pixels)
       - 8 tissue classes
       - Available at: https://zenodo.org/record/53169
    
    2. NCT-CRC-HE-100K Dataset
       - 100,000 non-overlapping image patches
       - 224 x 224 pixels
       - Available through Jakob Nikolas Kather's research
    
    ================================================================================
    Quick Start with Your Own Data:
    ================================================================================
    
    1. Create the directory structure:
       mkdir -p data/colorectal_histology/{tumor,stroma,complex,lympho,mucosa,muscle,normal,tumor_epithelium}
    
    2. Place your labeled images into the appropriate subdirectories
    
    3. Ensure images are in supported formats: .tif, .tiff, .jpg, .jpeg, .png
    
    4. Start training:
       python train.py --config config.yaml --data-dir data/colorectal_histology
    
    ================================================================================
    """
    print(info)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Data preparation helper for colorectal histology dataset'
    )
    parser.add_argument(
        '--source',
        type=str,
        help='Source directory containing original dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/colorectal_histology',
        help='Output directory for organized dataset'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset preparation information'
    )
    
    args = parser.parse_args()
    
    if args.info or (not args.source):
        create_sample_dataset_info()
    else:
        prepare_kather_dataset(args.source, args.output)
