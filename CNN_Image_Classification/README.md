# CNN Image Classification using PyTorch

## Project Overview

This project demonstrates the implementation of a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10** dataset in PyTorch. The project covers the complete deep learning workflow, including data preprocessing, data augmentation, CNN architecture design, model training, evaluation, and performance visualization.

The objective is to understand how convolutional neural networks learn hierarchical image features and classify images into one of the ten CIFAR-10 categories.

---

## Dataset

**Dataset:** CIFAR-10

* Total Images: 60,000
* Training Images: 50,000
* Testing Images: 10,000
* Image Size: 32 × 32 pixels
* Color Channels: RGB (3 Channels)
* Number of Classes: 10

### Classes

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

## Project Structure

``
CNN_Image_Classification/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── dataset.py
│   └── cnn_model.py
│
├── docs/
│   └── evaluation_report.md
│
├── models/
│   └── saved_cnn_model.pth
│
└── data/

```

---
```
## Features

* Image preprocessing and normalization
* Data augmentation using Random Horizontal Flip and Random Crop
* Custom CNN architecture
* Batch Normalization
* Max Pooling
* Dropout Regularization
* Adam Optimizer
* Cross Entropy Loss
* Model Training
* Model Evaluation
* Confusion Matrix
* Classification Report
* Accuracy and Loss Visualization
* Model Saving (.pth)

---

## CNN Architecture

Input Image (32×32×3)

↓

Conv2D (32 Filters)

↓

Batch Normalization

↓

ReLU

↓

Max Pooling

↓

Conv2D (64 Filters)

↓

Batch Normalization

↓

ReLU

↓

Max Pooling

↓

Conv2D (128 Filters)

↓

Batch Normalization

↓

ReLU

↓

Max Pooling

↓

Flatten

↓

Fully Connected Layer (256)

↓

Dropout (0.5)

↓

Output Layer (10 Classes)

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Navigate to the source folder and execute:

```bash
python cnn_model.py
```

---

## Evaluation Metrics

The model is evaluated using:

* Training Accuracy
* Training Loss
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Classification Report

---

## Learning Outcomes

Through this project, the following concepts were explored:

* Convolution Operation
* Filters and Feature Maps
* Stride and Padding
* Pooling Layers
* Batch Normalization
* Dropout
* Forward Pass
* Backpropagation
* CNN Training
* Model Evaluation
* Image Classification Pipeline

---

## Future Improvements

* Transfer Learning using ResNet or EfficientNet
* Hyperparameter Tuning
* Learning Rate Scheduling
* Data Augmentation Techniques
* Early Stopping
* Model Checkpointing
* GPU Training
* TensorBoard Integration

---

