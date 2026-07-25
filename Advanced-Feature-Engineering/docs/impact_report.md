# Impact Report

## Objective

The objective of this project was to investigate how different feature engineering techniques influence the performance of a Logistic Regression classifier on the Wine Quality dataset.

---

## Baseline Model

The baseline model was trained using the original dataset after preprocessing and feature scaling.

### Observation

- Simple implementation
- Easy to interpret
- Limited ability to model non-linear relationships

---

## Polynomial Features

Polynomial Features generated higher-order terms and squared features.

### Impact

Advantages

- Captured non-linear relationships
- Improved model flexibility
- Increased predictive capability

Disadvantages

- Large increase in feature dimensions
- Higher computational cost
- Reduced interpretability

---

## Interaction Features

Interaction Features created products between different variables without introducing squared terms.

### Impact

Advantages

- Captured relationships between multiple features
- Lower dimensionality compared to Polynomial Features
- Better interpretability

Disadvantages

- Increased training time
- Some interaction terms contributed little to prediction

---

## Log Transformation

The Log Transformation was applied to skewed numerical features such as:

- Chlorides
- Sulphates
- Residual Sugar

### Impact

Advantages

- Reduced positive skewness
- Stabilized variance
- Improved feature distributions

Disadvantages

- Only useful for positively skewed variables
- Slightly reduces interpretability

---

## Box-Cox Transformation

The Box-Cox Transformation further normalized skewed numerical features.

### Impact

Advantages

- Produced distributions closer to normal
- Improved optimisation during model training
- Reduced influence of extreme values

Disadvantages

- Applicable only to positive numerical values
- Slightly more computationally intensive

---

## Performance Comparison

| Method | Accuracy |
|---------|----------|
| Baseline | Replace with output |
| Polynomial Features | Replace with output |
| Interaction Features | Replace with output |
| Log Transformation | Replace with output |
| Box-Cox Transformation | Replace with output |

---

## Overall Findings

- Polynomial Features improved the model's ability to capture non-linear patterns but significantly increased feature dimensionality.
- Interaction Features provided a good balance between performance improvement and model complexity.
- Log Transformation effectively reduced skewness and improved numerical stability.
- Box-Cox Transformation produced the most normalized feature distributions and often resulted in the best predictive performance.

---

## Conclusion

Feature engineering plays a critical role in improving machine learning performance. Appropriate transformations can significantly enhance predictive accuracy without changing the underlying learning algorithm. However, increasing feature complexity also reduces model interpretability and increases computational cost. Selecting suitable feature engineering techniques requires balancing accuracy, efficiency, and interpretability.
