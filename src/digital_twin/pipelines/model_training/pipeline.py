"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.3.1
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_svm_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_svm_model,
            inputs=["x_train", "y_train", "parameters"],
            outputs="svm_model",
            name="train_svm_model_node"
        ),
        node(
            func=evaluate_model,
            inputs=["svm_model", "x_test", "y_test"],
            outputs="svm_model_metrics",
            name="evalute_svm_model_node"
        )
    ])
