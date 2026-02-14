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
