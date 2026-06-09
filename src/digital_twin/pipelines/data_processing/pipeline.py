"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.3.1
"""

from kedro.pipeline import node, Pipeline, pipeline

from .nodes import preprocess_and_merge_charts, preprocess_experiment_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_and_merge_charts,
            inputs="chart_data",
            outputs="preprocessed_chart_data",
            name="preprocess_chart_data_node"
        ),
        node(
            func=preprocess_experiment_data,
            inputs="experiment_data",
            outputs="preprocessed_experiment_data",
            name="preprocess_experiment_node"
        )

    ])
