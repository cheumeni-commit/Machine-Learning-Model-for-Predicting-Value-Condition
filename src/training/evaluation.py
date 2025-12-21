import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def get_metrics(y_true, y_pred, classes):
    """Per-class performance metrics."""
    # Performance
    performance = {"overall": {}, "class": {}}

    # Overall performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    performance["overall"]["precision"] = metrics[0]
    performance["overall"]["recall"] = metrics[1]
    performance["overall"]["f1"] = metrics[2]
    performance["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for _, cl in enumerate(classes):
        performance["class"][cl] = {
            "precision": np.mean(metrics[0]),
            "recall": np.mean(metrics[1]),
            "f1": np.mean(metrics[2]),
            "num_samples": np.float64(np.mean(metrics[3])),
        }
    return performance
    

_METRICS = {
    'get_metrics': get_metrics,
}


def evaluate_model(fitted_model, *, X, y, classes):
    y_pred = fitted_model.predict(X)
    metrics = {name: func(y, y_pred, classes) for name, func in _METRICS.items()}
    return metrics.get("get_metrics")