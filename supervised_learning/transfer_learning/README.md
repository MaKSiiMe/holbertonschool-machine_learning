# 🧠 Transfer Learning on CIFAR-10: An Experimental Study

> Experimentation workflow for image classification with Keras and an analysis dashboard with Streamlit.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

This repository contains a complete solution for the Holberton School project on transfer learning. The goal is to build a high‑accuracy image classifier for the CIFAR‑10 dataset using a pre‑trained neural network and to document the process in a scientific report.

This project provides a framework for training and evaluating image classification models on the CIFAR-10 dataset (60,000 RGB images of size 32 × 32 pixels across ten classes) using transfer learning. Starting from a pre-trained MobileNetV2 model, we conducted a series of experiments to optimize performance, progressing from a baseline model at 87% accuracy to a final optimized model exceeding 92%.

The project consists of two main scripts:

1. **`0-transfer.py`**: A modular command-line configurable training script implementing a two-stage pipeline to launch experiments.
2. **`dashboard_app.py`**: An interactive Streamlit-based dashboard to visualize and compare experiment results.

Each training session generates a `JSON` file containing the configuration, metrics, learning curves, and confusion matrix, enabling complete analysis and reproducibility.

### Project Goals

- **Implement a training script** (`0-transfer.py`) that uses transfer learning to classify CIFAR-10 images. The script should achieve at least 87% validation accuracy and save the trained model to `cifar10.h5`.

- **Write a blog post** describing the experimental process. The post must follow a journal-style structure (Abstract, Introduction, Materials & Methods, Results, Discussion, etc.), include at least one picture and be published on Medium or LinkedIn.

## Key Features

- 🔬 **Scientific Approach**: A series of iterative experiments to understand the impact of each hyperparameter
- 🎯 **Two-Stage Training**: Fast feature extraction phase followed by careful fine-tuning phase
- 🔧 **Modular Script**: A single training script (`0-transfer.py`) configurable via command-line arguments
- 📊 **Interactive Analysis**: A dashboard developed with Streamlit and Plotly to visualize and compare all results
- 📈 **Structured Logging**: Automatic saving of each experiment (configuration, metrics, curves, confusion matrix) in JSON files
- 🎨 **Advanced Augmentation**: Support for standard augmentation and Mixup techniques
- ⚡ **GPU Acceleration**: Optimized for CUDA-compatible GPUs with TensorFlow

---

## 📁 Project Structure

```
transfer_learning/
│
├── 0-transfer.py         # Main training script implementing the two-stage pipeline
├── dashboard_app.py      # Streamlit dashboard to explore experiment logs
│
├── results/              # Folder where JSON logs from each run are stored
│   └── run_*.json
│
├── cifar10.h5            # Saved Keras model from the most recent training run
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── blog_post.md          # Markdown version of the blog article
```

---

## 🔧 Installation

### Prerequisites

- Python 3.9 or later (Python 3.10+ recommended)
- TensorFlow 2.15+ with GPU support (strongly recommended)
- CUDA-compatible GPU (highly recommended)
- Optional: Streamlit, pandas and Plotly for the dashboard

### Setup

Clone the repository and create a virtual environment:

```bash
# 1. Clone the project
git clone https://github.com/MaKSiiMe/holbertonschool-machine_learning.git
cd transfer_learning_project/transfer_learning

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

The `requirements.txt` file includes:
- `tensorflow` (2.15+)
- `scikit-learn`
- `pandas`
- `plotly`
- `streamlit`

**Note**: GPU drivers (CUDA 11.8) should be installed separately if you want GPU acceleration.

---

## 🚀 Usage

The workflow is simple: train models, then analyze them.

### 1. Running the Training Script

The training script implements a two-stage transfer learning pipeline. It accepts several command-line arguments to configure the training process. At minimum, you should specify the path where the results JSON will be saved.

**Basic Example (feature extraction only, no fine-tuning):**

```bash
python3 0-transfer.py --json_output_path "results/run_baseline.json"
```

**Advanced Example (fine-tune the last 30 layers with augmentation):**

```bash
python3 0-transfer.py \
    --n_unfreeze 30 \        # number of layers to unfreeze
    --epochs_stage1 10 \     # epochs for head training
    --epochs_stage2 50 \     # epochs for fine-tuning (stops early when val_loss plateaus)
    --batch_size 64 \        # batch size
    --augment \              # enable rotation/shift/zoom and horizontal flips
    --rotation 20 \          # maximum rotation angle in degrees
    --zoom 0.15 \            # zoom range
    --width_shift 0.1 \      # horizontal shift range
    --height_shift 0.1 \     # vertical shift range
    --json_output_path "results/run_finetune.json"
```

#### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_unfreeze` | Number of layers to unfreeze for fine-tuning (0 = keep base model frozen) | 0 |
| `--epochs_stage1` | Maximum epochs to train the classification head | 10 |
| `--epochs_stage2` | Maximum epochs to fine-tune the combined model | 10 |
| `--batch_size` | Batch size for training and feature extraction | 128 |
| `--augment` | Enable standard data augmentation (rotation, zoom, shifts and horizontal flips) | False |
| `--rotation` | Maximum rotation angle in degrees | 15 |
| `--zoom` | Zoom range for augmentation | 0.1 |
| `--width_shift` | Horizontal shift range | 0.1 |
| `--height_shift` | Vertical shift range | 0.1 |
| `--use_mixup` | Enable Mixup augmentation (combine images and labels linearly) | False |
| `--json_output_path` | Path to save the JSON results file. If empty, no log is written | "" |

**Note**: After training, the final model is automatically saved to `cifar10.h5` in the current directory. If a `--json_output_path` is provided, a structured JSON file is written with the configuration, training curves, timing, metrics and confusion matrix.

### 2. Visualizing Experiments

An optional Streamlit dashboard (`dashboard_app.py`) allows you to compare different runs side by side. After generating one or more JSON logs in the `results/` folder, launch the dashboard:

```bash
streamlit run dashboard_app.py
```

The web app automatically discovers JSON files in the `results/` directory, displays training curves and confusion matrices, and highlights the best configurations.

---

## 🧠 Architecture and Training Pipeline

### Background

Transfer learning is a technique where a model trained on one task is reused as the starting point for a different, but related task. The Keras developer guide describes it as taking layers from a previously trained model, freezing them to preserve their learned weights, adding new trainable layers on top, and training these new layers on the target dataset. An optional fine-tuning step unfreezes part of the pre-trained model and retrains it on the new data with a very low learning rate. Transfer learning is particularly useful when the target dataset is small or similar to the source data, as it reduces training time and risk of overfitting.

In this project we classify images from CIFAR-10 using MobileNetV2, a convolutional network pre-trained on ImageNet. We resize images to 160 × 160 pixels, normalize them and extract features with the frozen base model.

### Model Architecture

The model is built as follows:

```
Input (32x32) → Resizing (160x160) → Preprocessing → MobileNetV2 → GlobalAveragePooling → MLP Head → Output (10 classes)
```

### Stage 1: Head Training (Feature Extraction)

- The MobileNetV2 base is entirely frozen
- Features are pre-computed once to speed up the process
- Only the classification head is trained, with an AdamW optimizer and label smoothing
- Early Stopping monitors `val_loss` to find the optimal point
- The head includes batch normalization, dropout and softmax output

### Stage 2: Fine-Tuning

- The classification head retains its learned weights
- The last N layers of the MobileNetV2 base are unfrozen (excluding batch normalization layers)
- The complete model is re-trained with a very low learning rate
- The optimizer used is AdamW, which provides better regularization
- Data Augmentation is applied on-the-fly (rotation, zoom, shifts, horizontal flips)
- Early Stopping and ReduceLROnPlateau optimize training duration and prevent overfitting

### Development Notes

The script defines several helper functions:

- **`preprocess_data(X, y)`**: Normalizes CIFAR-10 images to the [0, 1] range and converts labels to one-hot encoding
- **`build_feature_extractor()`**: Constructs a Keras model that resizes images to 160 × 160, rescales them to [−1, 1] and passes them through a frozen MobileNetV2 to output a 1280-dimensional feature vector
- **`build_top_classifier()`**: Builds a small fully connected head with batch normalization, dropout and softmax output
- **`unfreeze_model_tail(base_model, n_unfreeze)`**: Sets all layers of the base model to non-trainable then unfreezes the last n_unfreeze layers
- **`load_and_prepare_data()`**: Loads CIFAR-10, splits it into training/validation/test sets, preprocesses the images and labels

The training pipeline uses `run_stage1_training()` for feature extraction and `run_stage2_finetuning()` for fine-tuning. After training, the model is evaluated with test-time augmentation (TTA), the confusion matrix is computed and all information is saved via `save_results_as_json()`.

---

## 📊 Experiment Logging (JSON)

Each run generates a structured JSON file containing all experiment information, as in this example:

```json
{
  "config": {
    "base_model": "MobileNetV2",
    "n_unfreeze": 30,
    "augment": "True",
    "batch_size": 64,
    "epochs_stage2": 100
  },
  "metrics": {
    "val_end2end": 0.9228,
    "test_end2end": 0.9156
  },
  "timing_sec": {
    "total": 937.45
  },
  "curves": { "..." },
  "confusion_matrix": [ [922, 5, ...], ... ]
}
```

These logs can be visualized in the dashboard to compare different configurations side by side.

---

## 🧪 Experimental Results

The project followed an iterative approach, summarized here:

| Stage | Objective | Findings |
|-------|-----------|----------|
| 1. Baseline | Reach 87% with n_unfreeze=0 | Goal achieved, but strong overfitting detected |
| 2. Optimizers | Compare SGD, Adam, AdamW | Little difference without fine-tuning |
| 3. Fine-Tuning | Test n_unfreeze > 0 | Improves performance but increases instability risk |
| 4. Failure | Aggressive fine-tuning with standard LR | Catastrophic forgetting, performance collapses |
| 5. Success | Aggressive fine-tuning with very low LR + Data Augmentation | Overfitting controlled, performance > 92% |
| 6. Finalization | Add TTA and Early Stopping | Time savings and small performance bonus |

**Summary**: The baseline configuration (feature extraction only) consistently yields around 87% accuracy on the validation set. Unfreezing a moderate number of layers and enabling data augmentation raises accuracy into the low 90% range, with 92% being achievable when unfreezing 30 layers and using rotation/shift/zoom augmentation. Aggressive fine-tuning of too many layers or large learning rates can degrade performance (catastrophic forgetting).

For a detailed discussion and figures, see the blog post in `blog_post.md`.

---

## 💾 Best Configuration Found

After multiple experiments, the following configuration yielded the best results:

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 |
| n_unfreeze | 30 |
| batch_size | 64 |
| Data Augmentation | Enabled (rotation: 20, zoom: 0.15, width/height shift: 0.1) |
| Optimizer (Stage 2) | AdamW |
| learning_rate (Stage 2) | 1e-5 |
| Callbacks | EarlyStopping (patience=10) + ReduceLROnPlateau |
| Validation Accuracy | ~92.3% |
| Test Accuracy (with TTA) | ~91.6% |

---

## 📝 Blog Article

The file [[index.md](https://maksiime.github.io/holbertonschool-machine_learning/)] contains a complete journal-style report of this project, including:
- Abstract
- Introduction
- Materials & Methods
- Results
- Discussion
- Acknowledgements
- References

It also embeds visualizations such as a confusion matrix and an abstract illustration of transfer learning. When publishing on Medium or LinkedIn, convert the Markdown into the appropriate format and upload the accompanying images.

---

## 👥 Credits

- **Developer**: Maxime
- **Institution**: Holberton School Machine Learning Curriculum
- **Framework**: TensorFlow / Keras
- **Dataset**: CIFAR-10 (University of Toronto)

We gratefully acknowledge François Chollet for the Keras library and the maintainers of TensorFlow and scikit-learn.

---

## 📝 License

This project is released under the MIT License - see the LICENSE file for details.
