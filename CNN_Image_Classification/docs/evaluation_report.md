# CNN Image Classification Evaluation Report

## Project Overview

This project implements a Convolutional Neural Network (CNN) using **PyTorch** to classify images from the **CIFAR-10** dataset. The objective was to understand the complete computer vision pipeline, including dataset preprocessing, CNN architecture design, model training, performance evaluation, and visualization.

---

# Dataset Information

**Dataset:** CIFAR-10

- Total Images: 60,000
- Training Images: 50,000
- Testing Images: 10,000
- Image Size: 32 × 32 pixels
- Color Channels: RGB (3 Channels)
- Number of Classes: 10

Classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

# Data Preprocessing

The following preprocessing techniques were applied before training:

- Random Horizontal Flip
- Random Crop (Padding = 4)
- Conversion to Tensor
- Image Normalization

These preprocessing techniques improve model generalization by introducing slight variations in training images while maintaining consistent input scaling.

---

# CNN Architecture

The implemented CNN consists of:

- 3 Convolutional Layers
- Batch Normalization after each Convolution
- ReLU Activation
- Max Pooling Layers
- Flatten Layer
- Fully Connected Layer
- Dropout Layer (0.5)
- Output Layer (10 Classes)

---

# Hyperparameters

| Hyperparameter | Value |
|----------------|------:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss |
| Batch Size | 128 |
| Epochs | 20 |
| Dropout | 0.5 |

---

# Training Performance

The model demonstrated steady improvement throughout the training process, with the training loss decreasing consistently while accuracy increased over successive epochs.

| Epoch | Training Loss | Training Accuracy |
|------:|--------------:|------------------:|
| 1 | 1.6978 | 36.83% |
| 2 | 1.4087 | 48.33% |
| 3 | 1.2933 | 53.30% |
| 4 | 1.2161 | 56.43% |
| 5 | 1.1597 | 58.58% |
| 6 | 1.1171 | 60.34% |
| 7 | 1.0875 | 61.39% |
| 8 | 1.0589 | 62.16% |
| 9 | 1.0402 | 63.56% |
| 10 | 1.0162 | 64.20% |
| 11 | 0.9965 | 65.17% |
| 12 | 0.9845 | 65.57% |
| 13 | 0.9608 | 66.63% |
| 14 | 0.9498 | 66.92% |
| 15 | 0.9364 | 67.33% |
| 16 | 0.9264 | 67.70% |
| 17 | 0.9137 | 68.20% |
| 18 | 0.9023 | 68.67% |
| 19 | 0.8940 | 68.98% |
| 20 | 0.8889 | 69.18% |

---

# Final Model Performance

**Final Training Loss:** **0.8889**

**Final Training Accuracy:** **69.18%**

**Test Accuracy:** **74%**

The decreasing loss and increasing accuracy indicate that the CNN successfully learned meaningful image representations from the CIFAR-10 dataset.

---

# Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------|----------:|-------:|---------:|--------:|
| Airplane | 0.74 | 0.75 | 0.74 | 1000 |
| Automobile | 0.85 | 0.90 | 0.87 | 1000 |
| Bird | 0.70 | 0.55 | 0.62 | 1000 |
| Cat | 0.55 | 0.55 | 0.55 | 1000 |
| Deer | 0.72 | 0.71 | 0.71 | 1000 |
| Dog | 0.66 | 0.63 | 0.64 | 1000 |
| Frog | 0.69 | 0.87 | 0.77 | 1000 |
| Horse | 0.89 | 0.72 | 0.79 | 1000 |
| Ship | 0.77 | 0.91 | 0.83 | 1000 |
| Truck | 0.87 | 0.83 | 0.85 | 1000 |

### Overall Metrics

| Metric | Value |
|---------|------:|
| Overall Accuracy | **74%** |
| Macro Average Precision | 0.74 |
| Macro Average Recall | 0.74 |
| Macro Average F1-Score | 0.74 |
| Weighted Average Precision | 0.74 |
| Weighted Average Recall | 0.74 |
| Weighted Average F1-Score | 0.74 |

---

# Performance Analysis

### Strengths

- Automobile achieved the highest recall (0.90).
- Ship achieved the highest recall (0.91).
- Horse obtained the highest precision (0.89).
- Truck achieved a strong F1-score of 0.85.
- The model maintained balanced performance across most classes with an overall accuracy of 74%.

### Areas for Improvement

- Cat and Bird classes obtained comparatively lower F1-scores due to visual similarities with other animal categories.
- Additional data augmentation and deeper CNN architectures could further improve performance.
- Transfer learning models such as ResNet or EfficientNet can significantly improve classification accuracy.

---

# Challenges Faced

During the implementation of this project, the following challenges were encountered:

- Understanding convolution and feature extraction concepts.
- Designing a suitable CNN architecture.
- Managing longer training times on CPU hardware.
- Debugging tensor shape mismatch errors.
- Resolving Batch Normalization channel mismatch issues.
- Fixing incorrect model save paths.
- Handling Windows DataLoader multiprocessing errors.
- Generating confusion matrix and classification report correctly.

---

# Future Improvements

The following enhancements can further improve model performance:

- Apply advanced data augmentation techniques.
- Use Transfer Learning (ResNet, VGG, EfficientNet).
- Implement Early Stopping.
- Add Learning Rate Scheduling.
- Perform Hyperparameter Tuning.
- Train using GPU acceleration.
- Introduce Model Checkpointing.
- Optimize the CNN architecture for higher accuracy.

---

# Conclusion

The project successfully demonstrated the complete workflow of image classification using a Convolutional Neural Network (CNN). The model learned meaningful visual features through convolution, pooling, and fully connected layers while Batch Normalization and Dropout improved training stability and generalization.

The model achieved:

- **Training Accuracy:** **69.18%**
- **Test Accuracy:** **74%**
- **Weighted F1-Score:** **0.74**

Overall, the project provided practical experience in deep learning model development, computer vision, CNN architecture design, model evaluation, and PyTorch implementation, establishing a strong foundation for advanced topics such as transfer learning, object detection, and image segmentation.
