# Dimensionality Reduction using PCA and t-SNE

## Problem Statement

High-dimensional datasets often increase computational complexity and make visualization difficult. This project investigates how PCA and t-SNE reduce dimensionality while preserving meaningful information and compares their impact on classification performance.

---

# Dataset

The MNIST handwritten digit dataset was selected due to its high dimensionality and multiple target classes.

After preprocessing, the data was standardized before applying dimensionality reduction techniques.

---

# Principal Component Analysis (PCA)

PCA is a linear dimensionality reduction technique that transforms correlated features into a smaller number of orthogonal principal components while maximizing variance.

### Advantages

- Fast computation
- Preserves global structure
- Suitable for machine learning models
- Reduces redundancy

### Disadvantages

- Linear method
- Limited visualization capability for nonlinear data

---

# t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a nonlinear dimensionality reduction technique designed primarily for visualization.

It preserves local similarities between observations, allowing natural clusters to emerge in lower-dimensional space.

### Advantages

- Excellent cluster visualization
- Captures nonlinear relationships
- Reveals hidden data structure

### Disadvantages

- Computationally expensive
- Sensitive to hyperparameters
- Not intended for feature extraction

---

# Visualization Comparison

### PCA

The PCA scatter plot displayed overlapping clusters, indicating that the first two principal components could not completely separate all digit classes.

### t-SNE

The t-SNE visualization produced clearly separated clusters, demonstrating superior capability in revealing the intrinsic structure of the dataset.

---

# Classification Performance

| Dataset | Accuracy |
|----------|----------|
| Original Dataset | **97.2%** |
| PCA Reduced Dataset | **95.0%** |
| t-SNE Reduced Dataset | **90.8%** |

---

# Analysis

### Original Dataset

- Highest classification accuracy
- Largest computational cost
- No information loss

---

### PCA Dataset

- Slight reduction in accuracy
- Significant reduction in dimensionality
- Faster model training
- Best balance between efficiency and predictive performance

---

### t-SNE Dataset

- Lowest classification accuracy
- Best visualization quality
- High computational cost
- Primarily useful for exploratory data analysis rather than classification

---

# PCA vs t-SNE

| Feature | PCA | t-SNE |
|----------|-----|--------|
| Method | Linear | Nonlinear |
| Speed | Fast | Slow |
| Visualization | Good | Excellent |
| Classification | Excellent | Moderate |
| Preserves Global Structure | Yes | No |
| Preserves Local Structure | Moderate | Excellent |
| Feature Extraction | Yes | No |

---

# Key Findings

- PCA retained most of the dataset information while significantly reducing dimensionality.
- t-SNE generated the clearest visual representation of the digit classes.
- Logistic Regression achieved the highest accuracy on the original dataset.
- PCA maintained classification performance close to the original data while improving computational efficiency.
- t-SNE is more appropriate for visualization tasks than for supervised learning.

---

# Conclusion

PCA and t-SNE serve different purposes in dimensionality reduction.

PCA is the preferred choice when the reduced data will be used for machine learning models because it preserves most of the useful information with minimal accuracy loss.

t-SNE is the preferred choice for exploratory data analysis and visualization because it effectively reveals hidden patterns and naturally separated clusters within high-dimensional data.

Selecting the appropriate technique depends on whether the objective is predictive modeling or visual interpretation of complex datasets.
