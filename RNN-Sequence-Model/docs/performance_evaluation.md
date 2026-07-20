# Performance Evaluation Report

## Project

**RNN-Based Sentiment Analysis using LSTM**

---

# Objective

The objective of this project was to develop an LSTM-based Recurrent Neural Network capable of classifying IMDB movie reviews into Positive or Negative sentiment by learning sequential dependencies in text data.

---

# Training Configuration

| Parameter               |                Value |
| ----------------------- | -------------------: |
| Model                   |                 LSTM |
| Optimizer               |                 Adam |
| Learning Rate           |                0.001 |
| Loss Function           | Binary Cross Entropy |
| Batch Size              |                   64 |
| Epochs                  |                    5 |
| Gradient Clipping       |              Enabled |
| Maximum Sequence Length |                  200 |

---

# Training Performance

During training, the model demonstrated continuous improvement in learning by reducing loss over successive epochs while increasing classification accuracy.

The following visualizations should be included:

* Training Loss vs Epoch
* Training Accuracy vs Epoch

---

# Evaluation Metrics

Record your final results below after training.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |0.7772 %|
| Precision |0.7835 %|
| Recall    |0.7662 %|
| F1-Score  |0.7748 %|

---

# Confusion Matrix

Include the confusion matrix generated during evaluation to illustrate the model's classification performance on Positive and Negative reviews.

---

# Prediction Analysis

Compare a small number of predictions with their true labels to demonstrate the model's effectiveness.

Example:

| Review   | Actual   | Predicted |
| -------- | -------- | --------- |
| Review 1 | Positive | Positive  |
| Review 2 | Negative | Negative  |
| Review 3 | Positive | Positive  |

---

# Discussion

The LSTM model successfully learned contextual relationships within movie reviews by maintaining long-term memory through its memory cells. Gradient Clipping helped improve training stability by preventing exploding gradients, while sequence padding enabled efficient batch processing.

Although the model achieved strong classification performance, accuracy can be further improved through additional hyperparameter tuning, pretrained embeddings, Bidirectional LSTMs, or transformer-based architectures.

---

# LSTM vs Standard RNN

| Feature                        | Standard RNN | LSTM                  |
| ------------------------------ | ------------ | --------------------- |
| Long-Term Memory               | Limited      | Excellent             |
| Vanishing Gradient Problem     | Severe       | Significantly Reduced |
| Cell State                     | No           | Yes                   |
| Gates                          | No           | Forget, Input, Output |
| Suitable for Long Sequences    | Limited      | Yes                   |
| Sentiment Analysis Performance | Moderate     | High                  |

---

# Conclusion

The LSTM architecture proved highly effective for sentiment classification on sequential text data. Compared with a standard RNN, it better preserved contextual information across long sequences, resulting in more reliable predictions and improved classification performance. This project reinforced the complete NLP workflow, including preprocessing, sequence modeling, training, evaluation, and visualization.
