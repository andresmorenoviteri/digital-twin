"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.3.1
"""

from kedro.pipeline import node, Pipeline, pipeline

from .nodes import preprocess_and_merge_charts, preprocess_experiment_data, preprocess_charts_excel, concat_chart

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_and_merge_charts,
            inputs="chart_data",
            outputs=["preprocessed_chart_data", "cols"],
            name="preprocess_chart_data_node"
        ),
        node(
            func=preprocess_experiment_data,
            inputs="input_data_updated",
            outputs="preprocessed_experiment_data",
            name="preprocess_experiment_node"
        ),
        node(
            func=preprocess_charts_excel,
            inputs=["chart_data_updated", "cols"],
            outputs="preprocessed_chart_data_new",
            name="preprocess_chart_data_new_node"
        ),
        node(
            func=concat_chart,
            inputs=["preprocessed_chart_data", "preprocessed_chart_data_new"],
            outputs="concatenated_chart_data",
            name="concatenate_chart_data_node"
        )

    ])
