# IMPORT LIBRARIES 

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocess import (
    load_imdb_dataset,
    preprocess_data
)


# HYPERPARAMETERS

VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
OUTPUT_DIM = 1

NUM_LAYERS = 2

DROPOUT = 0.5

BATCH_SIZE = 64

LEARNING_RATE = 0.001

EPOCHS = 5


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)




#---------------------------------------------------------
#              BUILDING OUR LSTM MODEL
#---------------------------------------------------------



class SentimentLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.fc = nn.Linear(
            HIDDEN_DIM,
            OUTPUT_DIM
        )


    def forward(self, x):

        embedded = self.embedding(x)

        _, (hidden, _) = self.lstm(embedded)

        hidden = hidden[-1]

        hidden = self.dropout(hidden)

        out = self.fc(hidden)

        return out
    


# TESTING THE MODEL

model = SentimentLSTM().to(device)

print(model)


# Load and preprocess the dataset

train_df, test_df = load_imdb_dataset()

X_train, X_test, y_train, y_test, tokenizer = preprocess_data(
    train_df,
    test_df
)

X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Creating Objects

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Creating Dataloaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# LOSS FUNCTION AND OPTIMIZER

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



#---------------------------------------------------------
#               TRAINING FUNCTION
#---------------------------------------------------------


def train_model(model, train_loader):

    model.train()

    running_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5
        )

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)



#---------------------------------------------------------
#              EVALUATING THE FUNCTION
#---------------------------------------------------------


def evaluate_model(model, test_loader):

    model.eval()

    predictions = []
    actual_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            outputs = model(inputs)

            predicted = (torch.sigmoid(outputs) >= 0.5).float()

            predictions.extend(
                predicted.cpu().numpy()
            )

            actual_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(actual_labels, predictions)

    precision = precision_score(actual_labels, predictions)

    recall = recall_score(actual_labels, predictions)

    f1 = f1_score(actual_labels, predictions)

    return accuracy, precision, recall, f1



#---------------------------------------------------------
#              TRAINING THE MODEL
#---------------------------------------------------------

for epoch in range(EPOCHS):

    loss = train_model(
        model,
        train_loader
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {loss:.4f}"
    )


#---------------------------------------------------------
#              EVALUATING THE MODEL
#---------------------------------------------------------

accuracy, precision, recall, f1 = evaluate_model(
    model,
    test_loader
)

print("\nModel Performance")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")


