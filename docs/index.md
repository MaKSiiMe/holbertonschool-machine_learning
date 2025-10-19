# From 87% to 92%: A Deep Dive into Optimizing Transfer Learning for CIFAR-10

**By Maxime** (with assistance from Gemini)  
*Published on: October 19, 2025*

---

## Abstract
**What did I do in a nutshell?**

This paper details an experimental process for classifying the CIFAR-10 dataset using transfer learning with a MobileNetV2 backbone. An initial baseline model was established, meeting the preliminary goal of >87% accuracy but exhibiting significant overfitting. A systematic series of optimization experiments was then conducted to mitigate overfitting and improve generalization.

Key hyperparameters were iteratively tuned, including fine-tuning depth (`n_unfreeze`), optimizer choice (SGD, Adam, AdamW), data augmentation strategies, batch size, and learning rate. The study revealed that a combination of deep fine-tuning, strong regularization (data augmentation, AdamW with decoupled weight decay), and a critically low learning rate (1e-5) was essential to overcome catastrophic forgetting and push performance.

The final optimized model, enhanced with Early Stopping and Test-Time Augmentation (TTA), achieved a **final test accuracy of 91.6%**.

---

## Introduction
**What is the problem?**

Image classification is a cornerstone of computer vision, and the CIFAR-10 dataset, while ubiquitous, presents a non-trivial challenge due to its low-resolution (32×32) images and diverse classes. Training deep convolutional neural networks from scratch requires vast computational resources and datasets. Transfer learning offers a powerful and efficient alternative, enabling the adaptation of models pre-trained on large-scale datasets like ImageNet to new, specific tasks.

### Project Objectives

This project's objective was twofold:

1. **Establish a baseline model** using a frozen MobileNetV2 backbone that could achieve a target accuracy of over 87%
2. **Conduct a systematic investigation** into optimization techniques to surpass this baseline and achieve high performance (>90%)

### Research Hypothesis

Our central hypothesis was that a simple feature-extraction model would meet the baseline but suffer from severe overfitting. We further hypothesized that a **carefully tuned fine-tuning strategy**, coupled with a suite of modern regularization techniques, would be necessary to control this overfitting and elevate the model's accuracy beyond 90%.

### Background on Transfer Learning

The CIFAR-10 dataset contains 60,000 colour images (32 × 32 pixels) divided into ten classes—airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships and trucks. Training a deep neural network from scratch on such small images can be challenging: the model may overfit quickly and struggle to learn high-level features.

Transfer learning addresses this problem by reusing knowledge from a network trained on a related, larger dataset. According to the [Keras guide](https://keras.io/guides/transfer_learning/), transfer learning involves:
1. Taking layers from a previously trained model
2. Freezing them to preserve their learned weights
3. Adding new trainable layers on top
4. Training these new layers on the target dataset

A final fine-tuning step optionally unfreezes part of the base model and trains it with a low learning rate. [GeeksforGeeks](https://www.geeksforgeeks.org/what-is-transfer-learning/) succinctly describes transfer learning as repurposing a model trained on one task to serve as the foundation for a second, related task.

---

## Materials and Methods
**How did I solve the problem?**

Our experimental framework was built in Python using TensorFlow with the Keras API. A custom interactive dashboard was developed with Streamlit and Plotly to facilitate real-time analysis of experimental results.

### Dataset

The **CIFAR-10 dataset** was used, split into:
- **Training**: 45,000 images
- **Validation**: 5,000 images
- **Testing**: 10,000 images

**Preprocessing pipeline**:
1. **Normalization**: Images are cast to `float32` and scaled to the [0, 1] range
2. **Label Encoding**: Labels are one-hot encoded into a 10-dimensional vector
3. **Resizing**: Images are resized from 32 × 32 to 160 × 160 using bilinear interpolation to match MobileNetV2's expected input size
4. **Rescaling**: A `Rescaling` layer maps pixel values from [0, 1] to [−1, 1] because MobileNetV2 was trained with this normalization

### Model Architecture

The core of the model is a **MobileNetV2 backbone** with weights pre-trained on ImageNet.

Our base model is:
```python
mobilenet_v2.MobileNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(160, 160, 3)
)
```

**Architecture Overview**:
```
Input (32×32) → Lambda Resizing (160×160) → Rescaling → MobileNetV2 → Custom Head
```

**Classification Head** (replacing the original classification layer):
- **Global Average Pooling**: Reduces the spatial dimension of feature maps to a 1280-dimensional vector
- **Batch Normalization**: Normalizes activations
- **Dense Layer**: With ReLU activation to introduce non-linearity
- **Dropout (0.25)**: For regularization
- **Output Layer**: Dense layer with 10 units and softmax activation

The head is compiled with:
- **Loss**: Categorical cross-entropy with **label smoothing (0.05)**
- **Optimizer**: AdamW with decoupled weight decay
- **Callbacks**: Early stopping monitoring validation loss

### Training Pipeline

A **two-stage training process** was employed:

#### Stage 1: Head Training (Feature Extraction)

The MobileNetV2 backbone was completely frozen. To accelerate this stage, **features were pre-computed** from the training and validation sets. Only the custom MLP head was trained on these static features using an **Adam optimizer**.

**Configuration**:
- Epochs: Up to 10
- Batch size: 128
- Early stopping: Patience 7
- **Result**: ≈87% validation accuracy

#### Stage 2: Fine-Tuning

The full end-to-end model was reassembled. The top *n* layers of the MobileNetV2 backbone were unfrozen, and the entire model was trained with a **very low learning rate** to delicately adapt the pre-trained weights.

**Data Augmentation** (applied on-the-fly):
- Random rotations (up to 20°)
- Horizontal flips
- Slight zooms
- Width and height shifts
- Optional: Mixup (interpolating two random images and their labels)

**Configuration**:
- Learning rate: **1×10⁻⁵** (critical parameter)
- Early stopping: Patience 10
- Callback: ReduceLROnPlateau

### Experimental Design & Regularization

The following hyperparameters and techniques were systematically investigated:

| Parameter | Values Tested | Final Choice |
|-----------|--------------|--------------|
| **Fine-Tuning Depth** (`n_unfreeze`) | [0, 10, 20, 30] | 30 |
| **Optimizer** | SGD (with Nesterov momentum), Adam, AdamW | AdamW |
| **Learning Rate** (fine-tuning) | [3e-4, 1e-4, 1e-5] | 1e-5 |
| **Batch Size** | [32, 64, 128] | 64 |
| **Data Augmentation** | None, Standard, Standard + Mixup | Standard |
| **Dropout** | - | 0.25 |
| **Label Smoothing** | - | 0.05 |

**Callbacks**:
- **EarlyStopping**: Monitoring `val_loss` with patience of 10
- **ReduceLROnPlateau**: Automatically adjusts learning rate when validation loss plateaus

### Evaluation Protocol

The final performance of the best model was evaluated using **Test-Time Augmentation (TTA)**, where predictions on the original test images and their horizontally flipped counterparts were averaged.

---

## Results
**What did I find out?**

### Baseline Performance

The initial model with a fully frozen backbone (`n_unfreeze=0`) successfully achieved a **test accuracy of 87.4%**, meeting the project requirement.

However, the learning curves from the analysis dashboard revealed a **significant overfitting problem**:
- Training accuracy: ~98%
- Validation accuracy: ~87%
- **Gap**: >10%

This substantial gap indicated that the model was memorizing the training set rather than learning generalizable features.

### The Catastrophic Forgetting Event

An initial attempt to combine deep fine-tuning (`n=30`) with a **standard learning rate (3e-4)** led to a catastrophic failure:
- Validation accuracy collapsed to near-random levels (~10-15%)
- The model's pre-trained weights were destroyed by aggressive updates
- **Lesson learned**: Fine-tuning is an act of **adaptation, not re-training**

This event demonstrated that a very low learning rate is essential to gently nudge the pre-trained weights into a new optimal configuration without erasing their learned knowledge.

### Experimental Results Summary

| Experiment | Unfrozen Layers | Augmentation | Learning Rate | Val. Accuracy | Test Accuracy |
|------------|----------------|--------------|---------------|---------------|---------------|
| Baseline | 0 | None | - | 0.873 | 0.868 |
| Fine-tune 10 layers | 10 | None | 1e-4 | 0.895 | 0.888 |
| Fine-tune 30 layers (Failed) | 30 | None | 3e-4 | ~0.15 | ~0.14 |
| Fine-tune 30 layers | 30 | Standard | 1e-5 | **0.923** | **0.916** |
| Fine-tune 30 layers + Mixup | 30 | Standard + Mixup | 1e-5 | 0.917 | 0.910 |

### Optimized Model Performance

The key breakthrough was combining the deep fine-tuning strategy with a **drastically reduced learning rate (1e-5)** and **strong regularization**. This approach not only prevented catastrophic forgetting but propelled the model to a new level of performance.

**Final Results**:
- **Peak Validation Accuracy**: 92.3%
- **Final Test Accuracy (with TTA)**: 91.6%
- **Training Time**: ~937 seconds (~15.6 minutes)

The learning curves for this final model show that the training and validation performance are closely aligned, indicating that **overfitting has been successfully controlled**.

### Key Findings

1. **Baseline Success**: The baseline model with frozen features already meets the 87% requirement
2. **Fine-Tuning Impact**: Unfreezing layers progressively improves accuracy
3. **Learning Rate is Critical**: The difference between catastrophic failure and success was a 30× reduction in learning rate
4. **Regularization Cocktail**: The combination of data augmentation, AdamW optimizer, dropout, and label smoothing works synergistically
5. **Mixup Limitation**: Aggressive mixup slightly reduces accuracy on CIFAR-10's small images

### Confusion Matrix Analysis

The final confusion matrix confirms high performance with a much stronger diagonal and fewer classification errors compared to the baseline.

**Common confusions** still occur between visually similar classes:
- Cats vs. Dogs
- Trucks vs. Automobiles
- Birds vs. Airplanes

---

## Discussion
**What does it mean?**

The experimental journey from an **87% baseline to a 91.6% optimized model** highlights several key principles of effective transfer learning.

### 1. Expected Overfitting in the Baseline

The initial overfitting of the baseline model was expected and confirmed that even a powerful pre-trained feature extractor requires careful regularization when applied to a smaller dataset.

### 2. The Critical Importance of Learning Rate

The most critical lesson was the **sensitivity of the fine-tuning process**, particularly with respect to the learning rate. The catastrophic forgetting event served as a stark reminder that fine-tuning is an **act of adaptation, not re-training**.

A very low learning rate (1e-5) is essential to gently nudge the pre-trained weights into a new optimal configuration without erasing their learned knowledge.

### 3. Synergistic Regularization Techniques

The success of the final model can be attributed to a synergistic **"cocktail" of regularization techniques**:
- Data Augmentation (rotation, zoom, flips, shifts)
- Smaller batch size (64 vs. 128)
- Dropout (0.25)
- Decoupled weight decay in AdamW optimizer
- Label smoothing (0.05)

All worked in concert to combat the increased tendency to overfit that comes with greater model complexity (i.e., more unfrozen layers).

### 4. Label Smoothing and Loss Interpretation

An interesting observation was the relatively high validation loss (>0.5) coexisting with high accuracy (>92%). This was diagnosed as a **positive side effect of label smoothing**.

By training the model to be **less overconfident**, we improved its generalization capabilities at the cost of a mathematically higher, but more realistic, loss value. This trade-off is acceptable and even desirable for robust models.

### 5. Final Workflow Efficiency

The final workflow, enhanced with **EarlyStopping** and **TTA**, represents a robust and efficient pipeline:
- Early Stopping prevents wasting epochs when validation loss plateaus
- TTA provides a small but consistent accuracy boost (~0.5%) at minimal computational cost
- The entire training process completes in under 16 minutes on a GPU

### Implications for Future Work

This methodology can be readily adapted to other image classification tasks:
- Try different pre-trained architectures (EfficientNet, ResNet, Vision Transformers)
- Experiment with other datasets of similar size and complexity
- Explore more advanced augmentation techniques (CutMix, AutoAugment)
- Investigate ensemble methods to push accuracy even higher

### Limitations

While our results are strong, several limitations should be noted:
- Results are specific to CIFAR-10; other datasets may require different hyperparameter settings
- Training was performed on a single GPU; larger-scale experiments might benefit from distributed training
- The study focused on MobileNetV2; other architectures might yield different optimal configurations

---

## Acknowledgements
**Who helped me out?**

This work was made possible by the open-source community behind:
- Python, TensorFlow, and Keras
- Scikit-learn
- Streamlit and Plotly
- The **Holberton School** for the project framework

---

## Literature Cited
**Whose work did I refer to?**

1. **F. Chollet**. *Transfer Learning & Fine-tuning*. Keras Developer Guides (accessed Jun 25, 2023). Available at: [https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/)

3. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C.** (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510-4520.

4. **Loshchilov, I., & Hutter, F.** (2019). *Decoupled Weight Decay Regularization*. In International Conference on Learning Representations (ICLR).

5. **Kingma, D. P., & Ba, J.** (2014). *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980.

6. **Krizhevsky, A., & Hinton, G.** (2009). *Learning multiple layers of features from tiny images*. Technical Report, University of Toronto.

---

## Appendices
**Extra Information**

### Appendix A: Experimental Timeline

The complete experimental process took approximately **1 week** and included:

1. **Days 1-2**: Infrastructure setup, baseline establishment, initial dashboard development
2. **Days 3-5**: Systematic hyperparameter exploration, catastrophic forgetting incident, learning rate optimization
3. **Days 6-7**: Final model refinement, TTA implementation, documentation and blog post writing

### Appendix B: Deliverables

The outcome of this project includes:

#### 1. Robust Training Script (`0-transfer.py`)
- ✅ Configurable via command-line arguments
- ✅ Two-stage workflow (feature extraction + fine-tuning)
- ✅ Data augmentation and mixup support
- ✅ Complete JSON logging

#### 2. Interactive Dashboard (`dashboard_app.py`)
- ✅ Built with Streamlit and Plotly
- ✅ Loads and filters experiment logs
- ✅ Visualizes training curves, performance and confusion matrices

#### 3. High-Performing Model
- ✅ **91.6% test accuracy**
- ✅ Controlled overfitting
- ✅ Significantly surpasses 87% target

### Appendix C: Technical Issues Encountered

#### 1. Missing Data Logs
**Problem**: Early versions didn't save confusion matrices or training curves  
**Solution**: Revised script to record comprehensive JSON logs  
**Lesson**: Always log everything for complete analysis

#### 2. API Changes
**Problem**: TensorFlow deprecated `lr` in favor of `learning_rate`  
**Solution**: Updated code to match latest API  
**Lesson**: Always consult latest documentation

#### 3. Memory Errors
**Problem**: VGG16 with 224×224 input caused OOM errors  
**Solution**: Stuck with MobileNetV2 and 160×160 input  
**Lesson**: Model choice and input size greatly impact resource requirements

#### 4. Version Compatibility
**Problem**: Missing `RandomCutout` API in TensorFlow version  
**Solution**: Used standard augmentation instead  
**Lesson**: Keep environment clean and up-to-date

### Appendix D: Key Takeaways

- **Two-stage pipeline**: Fast feature extraction followed by selective fine-tuning makes experimentation efficient
- **Learning rate is paramount**: The difference between failure and success was a 30× learning rate reduction
- **Regularization cocktail**: Multiple techniques work synergistically to combat overfitting
- **Methodology is reusable**: Adjust hyperparameters, try different base models or explore other augmentations
- **Documentation matters**: Comprehensive logging and visualization enable rapid iteration
- **Balance is critical**: Too much fine-tuning destroys pre-trained knowledge; too little leaves performance on the table

The complete code, trained models and experiment logs are available in the project repository.

---

**Published on**: October 19, 2025  
**Author**: Maxime  
**Institution**: Holberton School  
**GitHub**: [[Repository Link](https://github.com/MaKSiiMe/holbertonschool-machine_learning/tree/main/supervised_learning/transfer_learning)]  
**Contact**: [[Email](9865@holbertonstudents.com)/[LinkedIn](https://www.linkedin.com/in/maxime-truel/)]
