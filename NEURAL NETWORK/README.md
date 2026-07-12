# Neural Network Classification using PyTorch

## Overview

This project implements a **Multi-Layer Perceptron (MLP)** using **PyTorch** to perform binary classification on the Breast Cancer Wisconsin dataset. The model consists of two hidden layers with ReLU activation functions and is trained using the Adam optimizer with CrossEntropy Loss.

---

## Objectives

* Build a neural network using PyTorch.
* Understand the architecture of a Multi-Layer Perceptron (MLP).
* Implement forward propagation, backpropagation, and weight optimization.
* Evaluate the model on unseen test data.

---

## Dataset


* **Dataset:** Breast Cancer Wisconsin Dataset
* **Source:** Scikit-learn ('load_breast_cancer')
* **Samples:** 569
* **Features:** 30 numerical features
* **Classes:**

  * Benign
  * Malignant

---

## Neural Network Architecture

**Input Layer**

* 30 Input Features

**Hidden Layer 1**

* 64 Neurons
* ReLU Activation

**Hidden Layer 2**

* 32 Neurons
* ReLU Activation

**Output Layer**

* 2 Neurons (Logits)

---

## Hyperparameters

| Hyperparameter | Value            |
| -------------- | ---------------- |
| Optimizer      | Adam             |
| Learning Rate  | 0.001            |
| Loss Function  | CrossEntropyLoss |
| Epochs         | 50               |
| Batch Size     | 32               |
| Hidden Layers  | 2                |
| Hidden Neurons | 64, 32           |

---

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* NumPy

---

## Project Structure

``
Neural-Network-Classification/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── model.py
│   └── train.py
│
└── docs/
    └── architecture_report.md
```

---

```

## Training Workflow

1. Load dataset.
2. Split into training and testing sets.
3. Standardize features.
4. Convert data to PyTorch tensors.
5. Create DataLoader.
6. Define the MLP architecture.
7. Train the model using forward propagation and backpropagation.
8. Evaluate validation accuracy after each epoch.
9. Test the trained model on unseen data.

``

---

## Results

The model prints:

* Training Loss (per epoch)
* Validation Accuracy (per epoch)
* Final Test Accuracy

> **Note:** The exact accuracy may vary slightly because neural network weights are initialized randomly before training.

---

## Key Concepts Demonstrated

* Artificial Neural Networks (ANN)
* Multi-Layer Perceptron (MLP)
* ReLU Activation Function
* Forward Propagation
* Backpropagation
* CrossEntropy Loss
* Adam Optimizer
* Gradient Descent
* Binary Classification

---

## Conclusion

This project demonstrates the complete implementation of a feedforward neural network for binary classification using PyTorch. It highlights how neural networks learn by minimizing prediction error through backpropagation and optimization, providing a practical introduction to deep learning fundamentals.
