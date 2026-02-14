# Colorectal Histopathology Classifier
Mission Statement: A PyTorch-based Proof of Concept (POC) dedicated to the legacy of the user's late sister, who passed away from Stage IV colon cancer. The goal is to move beyond "Hello World" tutorials and build a meaningful image classifier for colorectal cancer diagnostics.

1. Technical Objective
Build a binary or multi-class image classifier using PyTorch to distinguish between healthy and malignant colorectal tissue.

Primary Dataset: NCT-CRC-HE-100K (or the beginner-friendly MedMNIST version).

Model Architecture: Utilize Transfer Learning via a pre-trained model (e.g., ResNet-18 or EfficientNet).

Key Libraries: torch, torchvision, matplotlib, and medmnist (if applicable).

2. Core Features Required
Data Pipeline: Custom DataLoader with normalization specific to histopathology slides.

Training Loop: Implementation of Cross-Entropy Loss and Adam Optimizer, with a focus on Recall (minimizing False Negatives in a clinical context).

Interpretability (Crucial): Implementation of Grad-CAM or "Saliency Maps" to generate heatmaps over tissue slides, showing where the model detects potential malignancy.

Inference Script: A way to test the model on single, unseen "out-of-sample" biopsy tiles.

3. Tone & Direction
Precision & Empathy: The code should be professional and well-documented, reflecting the weight of the subject matter.

Educational Depth: Prioritize explaining why certain layers or transforms are used, rather than just providing "black box" code.

Legacy Focus: The project should culminate in a clean, visual output (a "Diagnostic Dashboard" or heatmap visualization) suitable for a memorial presentation or portfolio.

# PyTorch Colorectal Histopathology Classifier

> *Dedicated to the memory of those who have passed from colon cancer, especially to the late sister of this project's creator. This is more than code—it's a legacy of love and a step toward better cancer diagnostics.*

## Mission Statement

This project represents a meaningful step beyond "Hello World" tutorials. It's a PyTorch-based **Proof of Concept (POC)** for classifying colorectal histopathology images, created in loving memory of someone who lost their battle with Stage IV colon cancer.

The goal is to contribute to the field of medical AI by building a robust image classifier that could assist pathologists in identifying different tissue types in colorectal cancer samples, potentially leading to faster and more accurate diagnoses.

---

## Overview

This repository contains a complete implementation of a deep learning-based colorectal tissue classifier using PyTorch and transfer learning. The classifier can identify **8 different tissue types** commonly found in colorectal histopathology:

1. **Tumor epithelium** (TUM) - Colorectal adenocarcinoma epithelium
2. **Simple stroma** (STR) - Simple stroma tissue
3. **Complex stroma** (COMPLEX) - Complex stroma
4. **Immune cells** (LYM) - Lymphocytes
5. **Debris** (DEB) - Tissue debris
6. **Normal mucosal glands** (NORM) - Normal colon mucosa
7. **Adipose tissue** (ADI) - Fat tissue
8. **Smooth muscle** (MUS) - Smooth muscle tissue

The classifier uses a **ResNet-based architecture** with transfer learning from ImageNet pretrained weights, fine-tuned on colorectal histopathology images.

---

## Features

✅ **Transfer Learning** - Uses pretrained ResNet (18/34/50/101) as backbone  
✅ **Data Augmentation** - Comprehensive augmentation pipeline for better generalization  
✅ **Training Pipeline** - Complete training script with validation and checkpointing  
✅ **Inference Tool** - Easy-to-use prediction script for new images  
✅ **Evaluation Metrics** - Detailed classification reports and confusion matrices  
✅ **TensorBoard Support** - Real-time training visualization  
✅ **Early Stopping** - Prevents overfitting with configurable patience  
✅ **Configurable** - YAML-based configuration for easy experimentation  

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/korygrandy/pytorch-colorectal-histopathology-classifier.git
cd pytorch-colorectal-histopathology-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset

### Quick Start: Download Dataset

We provide a convenient script to download the Kather Colorectal Histology dataset:

```bash
# Download the texture dataset (5,000 images, 8 classes)
python download_dataset.py --dataset texture

# Or download the CRC validation dataset (7,180 images, 9 classes)
python download_dataset.py --dataset crc --output-dir data
```

The script will:
- Download the dataset from Zenodo
- Extract and organize it into the correct directory structure
- Prepare it for immediate use with the training script

### Dataset Structure

The classifier expects data organized in the following structure:

```
data/colorectal_histology/
├── tumor/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── stroma/
│   ├── image1.tif
│   └── ...
├── complex/
├── lympho/
├── mucosa/
├── muscle/
├── normal/
└── tumor_epithelium/
```

### About the Dataset

This classifier is designed to work with the **Kather Colorectal Histology Dataset**, which contains 5,000 histological images of human colorectal cancer. The dataset is publicly available and widely used in medical imaging research.

**Citation:**
Kather, J. N., Weis, C.-A., Bianconi, F., Melchers, S. M., Schad, L. R., Gaiser, T., … Zöllner, F. G. (2016). Multi-class texture analysis in colorectal cancer histology. *Scientific Reports*, 6, 27988. http://doi.org/10.1038/srep27988

**Manual Download:**
- [Zenodo - Kather et al.](https://zenodo.org/record/53169)
- [TCIA - The Cancer Imaging Archive](https://www.cancerimagingarchive.net/)

---

## Usage

### 0. Quick Example (Optional)

To understand the ResNet-based neural network architecture, run the simple example:

```bash
python simple_resnet_example.py
```

This standalone script demonstrates:
- How to create a ResNet-based classifier
- The model architecture and custom classifier head
- A forward pass example with dummy data

### 1. Configuration

Edit `config.yaml` to customize training parameters:

```yaml
# Key configurations
data:
  data_dir: "data/colorectal_histology"
  batch_size: 32
  image_size: 224

model:
  architecture: "resnet50"  # Options: resnet18, resnet34, resnet50, resnet101
  pretrained: true
  num_classes: 8

training:
  epochs: 50
  learning_rate: 0.001
  patience: 10  # Early stopping patience
```

### 2. Training

Train the model on your dataset:

```bash
python train.py --config config.yaml --data-dir data/colorectal_histology
```

Optional arguments:
- `--resume <checkpoint_path>` - Resume training from a checkpoint
- `--data-dir <path>` - Override data directory from config

The training script will:
- Split data into train/val/test sets (80/10/10 by default)
- Train the model with data augmentation
- Save checkpoints to `checkpoints/`
- Log training metrics to `logs/`
- Generate training history plots and confusion matrices

### 3. Monitoring Training

If TensorBoard logging is enabled in config:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### 4. Making Predictions

Predict on a single image:

```bash
python predict.py --image path/to/image.tif --checkpoint checkpoints/best_model.pth
```

Predict on a directory of images:

```bash
python predict.py --image path/to/images/ --checkpoint checkpoints/best_model.pth --output-dir predictions/
```

Optional arguments:
- `--config config.yaml` - Specify config file
- `--top-k 5` - Show top 5 predictions
- `--no-viz` - Skip visualization generation

The prediction script will:
- Load the trained model
- Process images and generate predictions
- Save visualizations showing predictions and confidence scores
- Create a summary file with all predictions

### 5. Evaluation

The training script automatically evaluates on the test set after training completes. Results include:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix visualization
- Training/validation loss and accuracy curves

---

## Project Structure

```
pytorch-colorectal-histopathology-classifier/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py              # Dataset classes and data loaders
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py            # Model architectures
│   └── utils/
│       ├── __init__.py
│       └── helpers.py               # Training utilities
├── train.py                         # Training script
├── predict.py                       # Inference script
├── evaluate.py                      # Evaluation script
├── download_dataset.py              # Dataset download script
├── simple_resnet_example.py         # Simple standalone example
├── prepare_data.py                  # Data organization helper
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore patterns
└── README.md                        # This file
```

---

## Model Architecture

The classifier uses a **ResNet-based transfer learning** approach:

1. **Backbone**: Pretrained ResNet (18/34/50/101) on ImageNet
2. **Custom Head**: 
   - Dropout layer (0.5)
   - Fully connected layer (features → 512)
   - ReLU activation
   - Dropout layer (0.5)
   - Output layer (512 → num_classes)

This architecture balances performance and computational efficiency while leveraging powerful pretrained features from ImageNet.

---

## Training Details

### Data Augmentation
- Random horizontal and vertical flips
- Random rotation (±15°)
- Color jittering (brightness, contrast, saturation, hue)
- Normalization using ImageNet statistics

### Optimization
- **Optimizer**: Adam or SGD (configurable)
- **Learning Rate**: 0.001 (default) with step or cosine decay
- **Loss Function**: Cross-entropy loss
- **Early Stopping**: Stops training if validation accuracy doesn't improve

### Regularization
- Dropout (0.5)
- Weight decay (L2 regularization)
- Data augmentation

---

## Performance Expectations

With proper training on a colorectal histology dataset:
- **Expected Accuracy**: 85-95% on test set
- **Training Time**: ~1-3 hours on modern GPU for 50 epochs
- **Inference Speed**: ~50-100 images/second on GPU

*Note: Actual performance depends on dataset quality, size, and hardware.*

---

## Contributing

This project is a meaningful POC built with love and dedication. Contributions that improve the classifier's accuracy, efficiency, or usability are welcome.

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test thoroughly
5. Submit a pull request

Please ensure contributions maintain the project's mission and quality standards.

---

## License

This project is open source and available for research and educational purposes. When using this code, please cite appropriately and respect the memory of those lost to cancer.

---

## Acknowledgments

- **In Memory**: This project is dedicated to the late sister of the creator, who passed from Stage IV colon cancer. Your memory inspires meaningful work.
- **Dataset**: Based on the methodology from Kather et al.'s colorectal histology research
- **PyTorch Team**: For the excellent deep learning framework
- **Medical Community**: For their tireless work in cancer research and treatment

---

## Citation

If you use this code in your research, please cite:

```
@misc{pytorch-colorectal-classifier,
  author = {Kory Grandy},
  title = {PyTorch Colorectal Histopathology Classifier},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/korygrandy/pytorch-colorectal-histopathology-classifier}
}
```

---

## Contact

For questions, suggestions, or collaborations:
- **GitHub Issues**: [Open an issue](https://github.com/korygrandy/pytorch-colorectal-histopathology-classifier/issues)
- **Repository**: [korygrandy/pytorch-colorectal-histopathology-classifier](https://github.com/korygrandy/pytorch-colorectal-histopathology-classifier)

---

## Final Words

This project represents more than just code—it's a tribute to loved ones lost to cancer and a small contribution toward better medical diagnostics. Every line written carries the hope that AI can help save lives and spare families from the pain of losing someone to this devastating disease.

**In loving memory of all those who have fought the battle against cancer. You are not forgotten.**

---

*"The best way to honor those we've lost is to work toward a future where others don't have to suffer the same fate."*