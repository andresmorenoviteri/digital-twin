"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.3.1
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (split_by_experiment_id,
                    transform_time_series_into_phases,
                    extract_tsfresh_features,
                    scale_and_align_features)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Step 1: Split both charts and experiments by unique ID (Stratified)
        node(
            func=split_by_experiment_id,
            inputs=["preprocessed_chart_data", "preprocessed_experiment_data", "parameters"],
            outputs=["chart_train", "chart_test", "experiment_train", "experiment_test"],
            name="split_by_id_node"
        ),
        
        # Step 2: Slice and flatten the training chart data into phase sections
        node(
            func=transform_time_series_into_phases,
            inputs="chart_train",
            outputs="phase_features_train",
            name="transform_train_phases_node"
        ),
        
        # Step 3: Slice and flatten the test chart data into phase sections (No Leakage)
        node(
            func=transform_time_series_into_phases,
            inputs="chart_test",
            outputs="phase_features_test",
            name="transform_test_phases_node"
        ),

        # 4. Tsfresh Feature Extraction with melt optimization (Train)
        node(
            func=extract_tsfresh_features,
            inputs="phase_features_train",
            outputs="extracted_features_train",
            name="extract_tsfresh_features_train_node"
        ),

        # 5. Tsfresh Feature Extraction with melt optimization (Test)
        node(
            func=extract_tsfresh_features,
            inputs="phase_features_test",
            outputs="extracted_features_test",
            name="extract_tsfresh_features_test_node"
        ),

        # 6. Align columns, isolate targets, and scale features
        node(
            func=scale_and_align_features,
            inputs=["experiment_train",
                    "experiment_test",
                    "extracted_features_train",
                    "extracted_features_test",
                    "parameters"
            ],
            outputs=["X_train_scaled", "X_test_scaled", "y_train", "y_test"],
            name="scaled_and_align_features_node"
        )
    ]
)
