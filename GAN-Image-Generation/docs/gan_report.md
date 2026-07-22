# GAN Implementation Report

## Introduction

Generative Adversarial Networks (GANs) are a class of deep learning models introduced by Ian Goodfellow in 2014. GANs consist of two competing neural networks: a Generator that produces synthetic data and a Discriminator that distinguishes between real and generated samples. Through adversarial learning, the Generator gradually learns to generate increasingly realistic data.

---

# Objective

The objective of this project was to implement a GAN capable of generating handwritten digit images using the MNIST dataset.

---

# Dataset

* Dataset: MNIST
* Number of Training Images: 60,000
* Image Size: 28 × 28
* Image Type: Grayscale

Images were converted to tensors and normalized to the range **[-1, 1]** before training.

---

# Generator Architecture

The Generator accepts a 100-dimensional latent noise vector as input.

Architecture:

* Linear (100 → 256)
* ReLU
* Linear (256 → 512)
* Batch Normalization
* ReLU
* Linear (512 → 1024)
* Batch Normalization
* ReLU
* Linear (1024 → 784)
* Tanh Activation

The output is reshaped into a 28 × 28 grayscale image.

---

# Discriminator Architecture

The Discriminator classifies whether an image is real or fake.

Architecture:

* Flatten Layer
* Linear (784 → 512)
* LeakyReLU
* Linear (512 → 256)
* LeakyReLU
* Linear (256 → 1)
* Sigmoid Activation

The output represents the probability that an image is real.

---

# Training Methodology

The Generator and Discriminator were trained alternately using Binary Cross Entropy Loss.

Training workflow:

1. Train the Discriminator using real images.
2. Generate fake images from random noise.
3. Train the Discriminator on generated images.
4. Train the Generator to maximize the Discriminator's confidence that generated images are real.
5. Repeat for each epoch.

Generated image samples were saved every five epochs to monitor progress.

---

# Hyperparameters

| Parameter        |                Value |
| ---------------- | -------------------: |
| Epochs           |                   20 |
| Batch Size       |                  128 |
| Learning Rate    |               0.0002 |
| Latent Dimension |                  100 |
| Optimizer        |                 Adam |
| Betas            |         (0.5, 0.999) |
| Loss Function    | Binary Cross Entropy |

---

# Results

The Generator progressively improved its ability to produce handwritten digits during training. Sample images generated at regular intervals demonstrated the evolution from random noise to recognizable digit-like structures.

The trained Generator and Discriminator models were saved for future inference and evaluation.

---

# Challenges Faced

* Maintaining a balance between Generator and Discriminator learning.
* Training instability due to adversarial optimization.
* Selecting suitable hyperparameters.
* Monitoring model performance despite non-monotonic loss values.
* Preventing one network from dominating the training process.

---

# Conclusion

This project successfully implemented a Vanilla GAN using PyTorch. The Generator learned to synthesize realistic handwritten digit images from random noise while the Discriminator learned to distinguish generated samples from real MNIST images. The project demonstrates the core principles of adversarial learning and provides a strong foundation for more advanced GAN architectures such as DCGAN, Conditional GAN, and Wasserstein GAN.
