import torch
import torch.nn as nn
import torch.optim as optim


# LOADING DATASET

from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target


# TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,random_state=42
)


# FEATURE SCALING

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# CONVERTING INTO PyTorch Tensors

from torch.utils.data import TensorDataset

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# CREATE DataLoader

from torch.utils.data import DataLoader

train_dataset = TensorDataset(X_train, y_train)

train_Loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# CREATE MODEL

from model import NeuralNetwork

model = NeuralNetwork()


# LOSS FUNCTION AND OPTIMIZER

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# TRAINING LOOP
epochs = 50

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for inputs, labels in train_Loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    model.eval()

    with torch.no_grad():

        predictions = model(X_test)

        _, predicted = torch.max(predictions, 1)

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(
            y_test,
            predicted
        )

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Validation Accuracy: {accuracy:.4f}"
    )


# FINAL EVALUATION

model.eval()

with torch.no_grad():

    outputs = model(X_test)

    _, predicted = torch.max(outputs, 1)

    test_accuracy = accuracy_score(
        y_test,
        predicted
    )

print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")


