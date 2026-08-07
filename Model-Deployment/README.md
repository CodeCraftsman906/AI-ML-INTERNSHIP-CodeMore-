# Credit Card Fraud Detection API

## Overview

This project demonstrates how to deploy a trained Logistic Regression model using FastAPI. The model is serialized with Joblib, loaded when the API starts, and exposed through a REST endpoint that accepts JSON input and returns predictions.

---

## Project Structure

```
Model-Deployment/
│
├── app/
│   ├── main.py
│   └── model.joblib
│
├── tests/
│   └── test_api.sh
│
├── requirements.txt
└── README.md
```

---

## Installation

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Server runs at

```
http://127.0.0.1:8000
```

---

## API Documentation

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

## Endpoint

### POST /predict

Returns the prediction for a credit card transaction.

### Sample Request

```json
{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

### Sample Response

```json
{
  "prediction": 1
}
```

---

## Testing

Run:

```bash
bash tests/test_api.sh
```

Or test interactively using Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Joblib
- Pandas
- NumPy
- Pydantic
