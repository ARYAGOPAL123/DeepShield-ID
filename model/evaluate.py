import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_classification_metrics(y_true, y_pred, labels=None):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    if labels:
        metrics['label_names'] = labels
    return metrics


def save_metrics(metrics, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
