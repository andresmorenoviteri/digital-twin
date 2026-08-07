"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.3.1
"""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (split_by_experiment_id,
                    transform_time_series_into_phases,
                    extract_tsfresh_features,
                    scale_and_align_features,
                    drop_correlated_features,
                    select_top_features_by_importance,
                    rank_features_by_anova,
                    align_train_test_features)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Step 1: Split both charts and experiments by unique ID (Stratified)
        node(
            func=split_by_experiment_id,
            inputs=["concatenated_chart_data", "preprocessed_experiment_data", "parameters"],
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
            outputs=["X_train_scaled", "X_test_scaled", "y_train", "y_test", "scaler", "col_to_max", "col_to_min", "col_to_median", "feature_columns", "X_train_before_impute", "X_test_before_impute"],
            name="scaled_and_align_features_node"
        ),
        
        # 7. Drop highly correlated features from the training matrix
        node(
            func=drop_correlated_features,
            inputs=["X_train_scaled", "parameters"],
            outputs="X_train_features_reduced",
            name="drop_correlated_features_node"
        ),

        # 8. Train a quick RandomForest and keep only the top N features by importance
        node(
            func=select_top_features_by_importance,
            inputs=["X_train_features_reduced", "y_train", "parameters"],
            outputs="X_train_final",
            name="select_top_features_node"
        ),

        # 9. Rank remaining features by ANOVA F-score
        node(
            func=rank_features_by_anova,
            inputs=["X_train_final", "y_train", "parameters"],
            outputs="feature_scores",
            name="rank_features_by_anova_node"
        ),

        # 10. Align train/test sets to the final feature list
        node(
            func=align_train_test_features,
            inputs=["X_train_final", "X_test_scaled", "feature_scores"],
            outputs=["x_train", "x_test"],
            name="align_train_test_features_node"
        )
    ]
)
