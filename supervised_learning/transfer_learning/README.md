# 🧠 Transfer Learning on CIFAR-10: An Experimental Study

> Experimentation workflow for image classification with Keras and an analysis dashboard with Streamlit.

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

This project provides a framework for training and evaluating image classification models on the CIFAR-10 dataset using transfer learning. Starting from a pre-trained MobileNetV2 model, we conducted a series of experiments to optimize performance, progressing from a baseline model at 87% accuracy to a final optimized model exceeding 92%.

The project consists of two main scripts:

1. **`0-transfer.py`**: A modular command-line configurable training script to launch experiments.
2. **`dashboard_app.py`**: An interactive Streamlit-based dashboard to visualize and compare experiment results.

Each training session generates a `JSON` file containing the configuration, metrics, learning curves, and confusion matrix, enabling complete analysis and reproducibility.

## Key Features

- 🔬 **Scientific Approach**: A series of iterative experiments to understand the impact of each hyperparameter
- 🎯 **Two-Stage Training**: Fast feature extraction phase followed by careful fine-tuning phase
- 🔧 **Modular Script**: A single training script (`0-transfer.py`) configurable via command-line arguments
- 📊 **Interactive Analysis**: A dashboard developed with Streamlit and Plotly to visualize and compare all results
- 📈 **Structured Logging**: Automatic saving of each experiment (configuration, metrics, curves, confusion matrix) in JSON files

---

## 📁 Project Structure

```
transfer_learning/
│
├── 0-transfer.py         # Main training script
├── dashboard_app.py      # Dashboard application with Streamlit
│
├── results/              # Contains JSON logs of each experiment
│   └── run_*.json
│
├── cifar10.h5            # Latest trained and saved model
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (highly recommended)

### Setup

```bash
# 1. Clone the project
git clone https://github.com/your-name/your-project.git
cd your-project/transfer_learning

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

- `tensorflow`
- `streamlit`
- `pandas`
- `plotly`
- `scikit-learn`

---

## 🚀 Usage

The workflow is simple: train models, then analyze them.

### 1. Launch Training

Use the `0-transfer.py` script with arguments to configure each experiment.

**Basic Example (without fine-tuning):**

```bash
python3 0-transfer.py --json_output_path "results/run_baseline.json"
```

**Advanced Example (with fine-tuning and data augmentation):**

```bash
python3 0-transfer.py \
    --n_unfreeze 30 \
    --epochs_stage1 30 \
    --epochs_stage2 100 \
    --batch_size 64 \
    --augment \
    --rotation 20 \
    --zoom 0.15 \
    --json_output_path "results/run_advanced.json"
```

#### Available Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_unfreeze` | Number of layers to unfreeze for fine-tuning | 0 |
| `--epochs_stage1` | Number of epochs for head training | 10 |
| `--epochs_stage2` | Maximum number of epochs for fine-tuning | 10 |
| `--batch_size` | Batch size | 128 |
| `--augment` | Enable standard data augmentation | False |
| `--rotation` | Rotation range for augmentation (in degrees) | 15 |
| `--zoom` | Zoom range for augmentation | 0.1 |
| `--width_shift` | Horizontal shift range | 0.1 |
| `--height_shift` | Vertical shift range | 0.1 |
| `--json_output_path` | Path to save the JSON results file | "" |

### 2. Launch Analysis Dashboard

Once you have generated one or more JSON files in the `results/` folder:

```bash
streamlit run dashboard_app.py
```

The dashboard will open in your browser and automatically detect all JSON files in the `results/` folder.

---

## 🧠 Architecture and Training Pipeline

The model is built as follows:

```
Input (32x32) → Resizing (160x160) → Preprocessing → MobileNetV2 → GlobalAveragePooling → MLP Head → Output (10 classes)
```

### Stage 1: Head Training

- The MobileNetV2 base is entirely frozen
- Features are pre-computed once to speed up the process
- Only the classification head is trained, with an Adam optimizer
- Early Stopping monitors `val_loss` to find the optimal point

### Stage 2: Fine-Tuning

- The classification head retains its learned weights
- The last N layers of the MobileNetV2 base are unfrozen
- The complete model is re-trained with a very low learning rate
- The optimizer used is AdamW, which provides better regularization
- Data Augmentation is applied on-the-fly
- Early Stopping and ReduceLROnPlateau optimize training duration

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

---

## 🧪 Our Experimental Journey

The project followed an iterative approach, summarized here:

| Stage | Objective | Findings |
|-------|-----------|----------|
| 1. Baseline | Reach 87% with n_unfreeze=0 | Goal achieved, but strong overfitting detected |
| 2. Optimizers | Compare SGD, Adam, AdamW | Little difference without fine-tuning |
| 3. Fine-Tuning | Test n_unfreeze > 0 | Improves performance but increases instability risk |
| 4. Failure | Aggressive fine-tuning with standard LR | Catastrophic forgetting, performance collapses |
| 5. Success | Aggressive fine-tuning with very low LR + Data Augmentation | Overfitting controlled, performance > 92% |
| 6. Finalization | Add TTA and Early Stopping | Time savings and small performance bonus |

---

## 💾 Best Configuration Found

After multiple experiments, the following configuration yielded the best results:

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 |
| n_unfreeze | 30 |
| batch_size | 64 |
| Data Augmentation | Enabled (rotation: 20, zoom: 0.15...) |
| Optimizer (Stage 2) | AdamW |
| learning_rate (Stage 2) | 1e-5 |
| Callbacks | EarlyStopping (patience=10) + ReduceLROnPlateau |
| Validation Accuracy | ~92.3% |
| Test Accuracy (with TTA) | ~91.6% |

---

## 👥 Credits

- **Developer**: Maxime
- **Framework**: TensorFlow / Keras
- **Dataset**: CIFAR-10

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
