# RNN-Based Sentiment Analysis using LSTM (PyTorch)

## Project Overview

This project implements a Recurrent Neural Network (RNN) using a Long Short-Term Memory (LSTM) architecture for binary sentiment classification on the IMDB Movie Reviews dataset. The objective is to classify movie reviews as **Positive** or **Negative** by learning long-term dependencies in sequential text data.

The project demonstrates the complete deep learning workflow, including data preprocessing, sequence tokenization, sequence padding, model training, evaluation, and performance visualization.

---

## Objectives

* Perform text preprocessing for sequential data.
* Tokenize and pad text sequences.
* Build an LSTM-based sentiment analysis model.
* Train the model while addressing exploding gradients using Gradient Clipping.
* Evaluate model performance using classification metrics.
* Visualize the training history and prediction performance.

---

## Project Structure

```text
RNN-Sequence-Model/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── preprocess.py
│   └── rnn_model.py
│
└── docs/
    └── performance_evaluation.md
```

---

## Dataset

**Dataset:** IMDB Movie Reviews

* Training Samples: 25,000
* Testing Samples: 25,000
* Classes:

  * Positive
  * Negative

---

## Data Preprocessing

The preprocessing pipeline includes:

* Loading the IMDB dataset
* Text cleaning
* Lowercasing
* Removing HTML tags
* Removing punctuation
* Removing numbers
* Tokenization
* Integer encoding
* Sequence padding

---

## Model Architecture

```text
Input Review
      │
      ▼
Embedding Layer
      │
      ▼
LSTM Layer
      │
      ▼
Dropout Layer
      │
      ▼
Fully Connected Layer
      │
      ▼
Sigmoid Output
```

---

## Hyperparameters

| Hyperparameter          | Value |
| ----------------------- | ----: |
| Vocabulary Size         |  10000|
| Embedding Dimension     |    128|
| Hidden Dimension        |    128|
| Output Dimension        |      1|
| Number of LSTM Layers   |      2|
| Dropout                 |    0.5|
| Batch Size              |     64|
| Learning Rate           |  0.001|
| Epochs                  |      5|
| Maximum Sequence Length |    200|

---

## Training

The model is trained using:

* Binary Cross Entropy Loss
* Adam Optimizer
* Gradient Clipping to reduce exploding gradients

---

## Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Visualizations

The project includes:

* Training Loss vs Epoch
* Training Accuracy vs Epoch
* Confusion Matrix
* Prediction Comparison (Actual vs Predicted)

---

## Key Learning Outcomes

* Understanding sequential data processing
* Text preprocessing for NLP
* Tokenization and sequence padding
* Embedding representations
* Long Short-Term Memory (LSTM) networks
* Gradient Clipping
* Binary sentiment classification
* Performance evaluation using classification metrics

---

## Future Improvements

* Compare LSTM with Standard RNN
* Implement GRU architecture
* Use pretrained word embeddings (GloVe or FastText)
* Experiment with Bidirectional LSTM
* Hyperparameter tuning for improved accuracy

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Datasets
* TensorFlow/Keras (Tokenizer & Padding)
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Conclusion

This project demonstrates how Long Short-Term Memory (LSTM) networks effectively capture long-term dependencies in sequential text, making them well suited for sentiment analysis. It provides practical experience with the complete NLP workflow—from preprocessing and sequence modeling to evaluation and visualization.
