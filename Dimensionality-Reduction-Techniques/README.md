# Dimensionality Reduction using PCA and t-SNE

## Project Overview

This project demonstrates two widely used dimensionality reduction techniques—**Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)**—on a high-dimensional dataset. The project explores how each technique reduces dimensionality, visualizes data, and impacts classification performance.

---

## Objectives

- Perform exploratory data analysis (EDA) on a high-dimensional dataset.
- Reduce dimensionality using PCA while retaining 95% of the explained variance.
- Apply t-SNE for nonlinear dimensionality reduction and visualization.
- Compare the visual quality of PCA and t-SNE embeddings.
- Evaluate classification performance on:
  - Original Dataset
  - PCA-Reduced Dataset
  - t-SNE-Reduced Dataset
- Compare execution time and overall performance.

---

## Dataset

**Dataset Used:** MNIST Handwritten Digits Dataset

Characteristics:

- High-dimensional image dataset
- 784 numerical features (28 × 28 pixels)
- 10 target classes (digits 0–9)
- Standardized before dimensionality reduction

---

## Project Structure

```
Dimensionality_Reduction_Project/
│
│
├── notebooks/
│   └── dimensionality_reduction.ipynb
│
├── src/
│   ├── pca_reduction.py
│   └── tsne_visualization.py
│
|
|── assets/
|    └── plots
|
|
├── README.md
├── requirements.txt
└── comparative_report.md
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Data Standardization
4. PCA Implementation
5. Explained Variance Analysis
6. t-SNE Implementation
7. Visualization
8. Logistic Regression Classification
9. Performance Comparison
10. Conclusions

---

## Results Summary

| Dataset | Accuracy |
|----------|----------|
| Original | **97.2%** |
| PCA | **95.0%** |
| t-SNE | **90.8%** |

---

## Observations

### PCA

- Preserved most of the important information.
- Reduced dimensionality significantly.
- Achieved classification accuracy close to the original dataset.
- Computationally efficient.

### t-SNE

- Produced excellent visual separation of classes.
- Generated well-defined clusters.
- Required significantly more computation time.
- Slightly reduced classification accuracy because it is primarily designed for visualization rather than predictive modeling.

---

## Conclusion

The project demonstrates that **PCA is an effective dimensionality reduction technique for machine learning pipelines**, preserving high classification accuracy while reducing computational complexity.

In contrast, **t-SNE excels at visualizing complex high-dimensional datasets**, producing clearly separated clusters that reveal the intrinsic structure of the data. However, it is less suitable as a preprocessing step for supervised learning due to its emphasis on preserving local neighborhood relationships rather than global feature representation.

---

## Future Improvements

- Compare with UMAP
- Hyperparameter optimization
- Evaluate additional classifiers
- 3D visualization
- Larger benchmark datasets

---
