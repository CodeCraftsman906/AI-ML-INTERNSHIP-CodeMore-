#  IMPORT LIBRARIES

import pandas as pd
import numpy as np

from datasets import load_dataset

import re
import string

import nltk
from nltk.tokenize import word_tokenize

from collections import Counter

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

# LOAD IMDB DATASET

def load_imdb_dataset():
    dataset = load_dataset("imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"]) 

    return train_df, test_df



#-------------------------------------------
# TEXT CLEANING
#-------------------------------------------


from tensorflow.keras.preprocessing.text import Tokenizer   # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # # type: ignore

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"http\S+|www\S+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# #-------------------------------------------
# # PREPROCESSING DATA
# #-------------------------------------------

def preprocess_data(train_df, test_df):

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    tokenizer = Tokenizer(
        num_words=10000,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(train_df["text"])

    X_train = tokenizer.texts_to_sequences(train_df["text"])
    X_test = tokenizer.texts_to_sequences(test_df["text"])

    max_length = 200

    X_train = pad_sequences(
        X_train,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    X_test = pad_sequences(
        X_test,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer
    )


#-------------------------------------------
# MAIN FUNCTION
#-------------------------------------------

if __name__ == "__main__":

    train_df, test_df = load_imdb_dataset()

    X_train, X_test, y_train, y_test, tokenizer = preprocess_data(
        train_df,
        test_df
    )

    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    print("Training Labels :", y_train.shape)
    print("Testing Labels  :", y_test.shape)

    print("\nSample Sequence:\n")
    print(X_train[0])

