"""
Download script for the Kather Colorectal Histology Dataset.

This script downloads the publicly available Kather Colorectal Histology dataset
from Zenodo and prepares it for training.

Dataset Citations:

1. Texture Dataset (5,000 images):
   Kather, J. N., Weis, C.-A., Bianconi, F., Melchers, S. M., Schad, L. R., 
   Gaiser, T., … Zöllner, F. G. (2016). Multi-class texture analysis in colorectal 
   cancer histology. Scientific Reports, 6, 27988. http://doi.org/10.1038/srep27988
   Dataset URL: https://zenodo.org/record/53169

2. CRC Validation Dataset (7,180 images):
   Kather, J. N., Halama, N., & Marx, A. (2018). 100,000 histological images of 
   human colorectal cancer and healthy tissue (v0.1) [Data set]. Zenodo.
   http://doi.org/10.5281/zenodo.1214456
   Dataset URL: https://zenodo.org/record/1214456
"""

import os
import argparse
import zipfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    """
    Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save downloaded file
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_kather_dataset(output_dir: str = 'data', dataset_type: str = 'texture'):
    """
    Download and extract the Kather Colorectal Histology dataset.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_type: Type of dataset to download ('texture' or 'crc')
    """
    print("=" * 80)
    print("Kather Colorectal Histology Dataset Downloader")
    print("=" * 80)
    print()
    print("This script downloads the publicly available dataset from Zenodo.")
    print()
    print("Citation:")
    print("Kather, J. N., et al. (2016). Multi-class texture analysis in")
    print("colorectal cancer histology. Scientific Reports, 6, 27988.")
    print("http://doi.org/10.1038/srep27988")
    print()
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset URLs
    datasets = {
        'texture': {
            'url': 'https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip',
            'filename': 'Kather_texture_2016_image_tiles_5000.zip',
            'extract_dir': 'Kather_texture_2016_image_tiles_5000',
            'description': 'Texture dataset (5,000 images, 150x150 pixels, 8 classes)'
        },
        'crc': {
            'url': 'https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip',
            'filename': 'CRC-VAL-HE-7K.zip',
            'extract_dir': 'CRC-VAL-HE-7K',
            'description': 'CRC validation dataset (7,180 images, 224x224 pixels, 9 classes)'
        }
    }
    
    if dataset_type not in datasets:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose 'texture' or 'crc'")
    
    dataset_info = datasets[dataset_type]
    
    print(f"Dataset: {dataset_info['description']}")
    print(f"URL: {dataset_info['url']}")
    print()
    
    # Download
    zip_path = os.path.join(output_dir, dataset_info['filename'])
    
    if os.path.exists(zip_path):
        print(f"Found existing file: {zip_path}")
        print("Skipping download...")
    else:
        print(f"Downloading to: {zip_path}")
        try:
            download_url(dataset_info['url'], zip_path)
            print(f"✓ Download complete!")
        except Exception as e:
            print(f"✗ Download failed: {str(e)}")
            print()
            print("Manual download instructions:")
            print(f"1. Visit: {dataset_info['url']}")
            print(f"2. Download the file manually")
            print(f"3. Save it to: {zip_path}")
            print(f"4. Run this script again")
            return False
    
    # Extract
    extract_path = os.path.join(output_dir, dataset_info['extract_dir'])
    
    if os.path.exists(extract_path):
        print(f"Found existing extracted directory: {extract_path}")
        print("Skipping extraction...")
    else:
        print(f"Extracting to: {extract_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"✓ Extraction complete!")
        except Exception as e:
            print(f"✗ Extraction failed: {str(e)}")
            return False
    
    # Organize into standard structure
    standard_path = os.path.join(output_dir, 'colorectal_histology')
    
    print()
    print("Organizing dataset into standard structure...")
    
    if dataset_type == 'texture':
        # For texture dataset, the extracted folder contains class subfolders
        if os.path.exists(extract_path):
            # Create class mapping for Kather texture dataset
            class_mapping = {
                '01_TUMOR': 'tumor_epithelium',
                '02_STROMA': 'stroma',
                '03_COMPLEX': 'complex',
                '04_LYMPHO': 'lympho',
                '05_DEBRIS': 'debris',
                '06_MUCOSA': 'mucosa',
                '07_ADIPOSE': 'adipose',
                '08_EMPTY': 'empty'
            }
            
            os.makedirs(standard_path, exist_ok=True)
            
            for original_name, target_name in class_mapping.items():
                source_dir = os.path.join(extract_path, original_name)
                target_dir = os.path.join(standard_path, target_name)
                
                if os.path.exists(source_dir):
                    if os.path.exists(target_dir):
                        print(f"  {target_name}: Already exists, skipping...")
                    else:
                        print(f"  {original_name} -> {target_name}")
                        shutil.copytree(source_dir, target_dir)
                        
                        # Count images
                        num_images = len(list(Path(target_dir).glob('*.tif')))
                        print(f"    {num_images} images")
    
    print()
    print("=" * 80)
    print("✅ Dataset download and preparation complete!")
    print("=" * 80)
    print()
    print(f"Dataset location: {standard_path}")
    print()
    print("Next steps:")
    print(f"1. Verify the data: ls {standard_path}")
    print("2. Update config.yaml with the correct data_dir path")
    print("3. Start training: python train.py --data-dir " + standard_path)
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download Kather Colorectal Histology Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download the texture dataset (5,000 images, 8 classes)
  python download_dataset.py --dataset texture
  
  # Download to a custom directory
  python download_dataset.py --dataset texture --output-dir /path/to/data
  
  # Download the CRC validation dataset (7,180 images, 9 classes)
  python download_dataset.py --dataset crc

Dataset Information:
  - texture: Kather 2016 texture dataset (5,000 images, 150x150px, 8 classes)
  - crc: CRC-VAL-HE-7K dataset (7,180 images, 224x224px, 9 classes)

Note: These datasets are publicly available from Zenodo under Creative Commons license.
Please cite the original papers when using this data.
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='texture',
        choices=['texture', 'crc'],
        help='Dataset to download (default: texture)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for downloaded dataset (default: data)'
    )
    
    args = parser.parse_args()
    
    try:
        success = download_kather_dataset(args.output_dir, args.dataset)
        if success:
            print("Done!")
            return 0
        else:
            print("Download failed. Please check the error messages above.")
            return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())
