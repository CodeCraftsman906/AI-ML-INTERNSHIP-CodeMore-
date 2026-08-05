import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# True Positive Rate and False Positive Rate
# ==========================================================

def calculate_tpr_fpr(y_true, y_pred):
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    tpr = TP / (TP + FN) if (TP + FN) else 0
    fpr = FP / (FP + TN) if (FP + TN) else 0

    return tpr, fpr


# ==========================================================
# Precision and Recall
# ==========================================================

def calculate_precision_recall(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) else 1
    recall = TP / (TP + FN) if (TP + FN) else 0

    return precision, recall


# ==========================================================
# Custom ROC Curve
# ==========================================================

def custom_roc_curve(y_true, y_prob):

    thresholds = np.sort(np.unique(y_prob))[::-1]

    fpr_list = []
    tpr_list = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        tpr, fpr = calculate_tpr_fpr(y_true, y_pred)

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return (
        np.array(fpr_list),
        np.array(tpr_list),
        thresholds
    )


# ==========================================================
# Custom Precision Recall Curve
# ==========================================================

def custom_precision_recall_curve(y_true, y_prob):

    thresholds = np.sort(np.unique(y_prob))[::-1]

    precision_list = []
    recall_list = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        precision, recall = calculate_precision_recall(
            y_true,
            y_pred
        )

        precision_list.append(precision)
        recall_list.append(recall)

    return (
        np.array(precision_list),
        np.array(recall_list),
        thresholds
    )


# ==========================================================
# Area Under Curve
# ==========================================================

def calculate_auc(x, y):

    order = np.argsort(x)

    return np.trapz(
        y[order],
        x[order]
    )


# ==========================================================
# Plot ROC Curve
# ==========================================================

def plot_roc_curve(fpr, tpr, auc_score):

    plt.figure(figsize=(8,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC Curve (AUC = {auc_score:.4f})"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="red",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Custom ROC Curve")

    plt.legend()
    plt.grid(True)

    plt.show()


# ==========================================================
# Plot Precision Recall Curve
# ==========================================================

def plot_precision_recall_curve(
    precision,
    recall,
    auc_score
):

    plt.figure(figsize=(8,6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        color="green",
        label=f"PR Curve (AUC = {auc_score:.4f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Custom Precision-Recall Curve")

    plt.legend()
    plt.grid(True)

    plt.show()
