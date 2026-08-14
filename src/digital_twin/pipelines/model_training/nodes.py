"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.3.1
"""
from typing import Dict, Any
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def train_svm_model(x_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict) -> SVC:
    """Trains an SVM classifier on the final selected features.
    
    Args:
        x_train: Final training feature matrix.
        y_train: Training labels (single-column DataFrame).
        parameters: Pipeline parameters containing SVM hyperparameters

    Returns:
        Fitted SVC model.
    """
    svm_params = parameters["svm"]

    svm_model = SVC(
        kernel=svm_params["kernel"],
        class_weight=svm_params["class_weight"],
        C=svm_params["C"],
        probability=svm_params["probability"],
        random_state=parameters["random_state"]
    )

    svm_model.fit(x_train, y_train.values.ravel())
    return svm_model


def evaluate_model(model: SVC, x_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the trained model on the held-out test set.
    Args:
        model: Fitted classifier.
        x_test: Test feature matrix.
        y_test: Test labels (single-column DataFrame).
    
    Returns: 
        Dictionary of evaluation metrics (safe to log/serialize as JSON).
    """

    y_true = y_test.values.ravel()
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

    print(classification_report(y_true, y_pred))
    return metrics

def package_production_artifacts(model, scaler, feature_columns, x_train, col_to_max, col_to_min, col_to_median, kind_to_fc_parameters) -> dict:
    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "x_train_columns": list(x_train.columns),
        "col_to_max": col_to_max,
        "col_to_min": col_to_min,
        "col_to_median": col_to_median,
        "kind_to_fc_parameters": kind_to_fc_parameters,
    }
