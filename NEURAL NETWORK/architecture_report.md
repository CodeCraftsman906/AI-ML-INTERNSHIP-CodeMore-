# Neural Network Architecture Report

## Introduction

This project implements a Multi-Layer Perceptron (MLP) using the PyTorch deep learning framework for binary classification. The objective is to classify breast cancer tumors as either benign or malignant by learning patterns from numerical input features.

---

# Neural Network Architecture

The implemented model consists of:

* **Input Layer:** 30 input features
* **Hidden Layer 1:** 64 neurons with ReLU activation
* **Hidden Layer 2:** 32 neurons with ReLU activation
* **Output Layer:** 2 output neurons representing the two target classes

The architecture is fully connected, meaning every neuron in one layer is connected to every neuron in the next layer.

---

# Activation Functions

## ReLU (Rectified Linear Unit)

The hidden layers use the ReLU activation function.

**Formula:**

ReLU(x) = max(0, x)

### Advantages

* Introduces non-linearity into the neural network.
* Helps reduce the vanishing gradient problem.
* Computationally efficient.
* Enables faster convergence during training.

ReLU is widely used in modern deep learning models because of its simplicity and effectiveness.

---

# Forward Propagation

Forward propagation is the process of passing input data through the neural network.

For each layer, the following operations occur:

1. Compute the weighted sum.
2. Add the bias.
3. Apply the ReLU activation function.
4. Pass the output to the next layer.

The output layer produces logits, which are used by the loss function during training.

---

# Backpropagation

Backpropagation is the learning mechanism of the neural network.

After forward propagation:

1. The model computes the prediction.
2. The prediction is compared with the actual target using the loss function.
3. Gradients are calculated for every trainable parameter.
4. These gradients indicate how each weight contributed to the prediction error.
5. The optimizer updates the weights to reduce future error.

This iterative process continues for every epoch until the model converges.

---

# Loss Function

The project uses **CrossEntropyLoss**, which is designed for classification tasks.

CrossEntropyLoss:

* Compares predicted class scores with actual labels.
* Produces a numerical measure of prediction error.
* Internally applies Softmax to convert logits into probabilities.

The objective during training is to minimize this loss.

---

# Optimization Algorithm

The model is trained using the **Adam Optimizer**.

Adam combines the advantages of momentum and adaptive learning rates, allowing faster and more stable convergence than standard gradient descent.

### Benefits

* Faster convergence
* Adaptive learning rates
* Efficient for deep neural networks
* Minimal hyperparameter tuning

---

# Hyperparameters

| Parameter     | Value            |
| ------------- | ---------------- |
| Epochs        | 50               |
| Batch Size    | 32               |
| Learning Rate | 0.001            |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |
| Hidden Layers | 2                |

---

# Training Process

The training pipeline consists of:

1. Dataset loading
2. Train-test split
3. Feature standardization
4. Tensor conversion
5. Forward propagation
6. Loss computation
7. Backpropagation
8. Weight updates using Adam
9. Validation accuracy monitoring
10. Final model evaluation

---

# Conclusion

The implemented Multi-Layer Perceptron successfully demonstrates the core principles of deep learning, including forward propagation, activation functions, loss minimization, backpropagation, and optimization. This project provides a practical understanding of how neural networks learn from data and improve their predictions through iterative training.
