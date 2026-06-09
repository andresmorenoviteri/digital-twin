"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 1.3.1
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute


def split_by_experiment_id(chart_df: pd.DataFrame, experiment_df: pd.DataFrame, parameters: Dict
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits both time-series and label dataframes by stratified experiment IDs.

    This ensures that all time-steps for a given cycle ID stay together in
    either the train set or the test set, preventing leakage.

    Args:
        chart_df: Cleaned long-form time-series data.
        experiment_df: Input parameter and label dataset containing exactly one row per cycle ID.
        parameters: Pipeline parameter dictionary containing split settings.

    Returns:
        Tuple containing chart_train, chart_test, experiment_train, experiment_test
    """
    target_col = parameters["target_column"]  # e.g., "Quality"
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]

    # Force the column to be an integer safely ---
    experiment_df[target_col] = experiment_df[target_col].astype("Int64")

    # 1. Stratify split the unique IDs using the label dataframe
    train_ids, test_ids = train_test_split(
        experiment_df["id"],
        test_size=test_size,
        random_state=random_state,
        stratify=experiment_df[target_col],
    )

    # 2. Slice the time-series chart data matching those IDs
    chart_train = chart_df[chart_df["id"].isin(train_ids)].reset_index(drop=True)
    chart_test = chart_df[chart_df["id"].isin(test_ids)].reset_index(drop=True)

    # 3. Slice the experiment label data matching those IDs
    experiment_train = experiment_df[experiment_df["id"].isin(train_ids)].reset_index(drop=True)
    experiment_test = experiment_df[experiment_df["id"].isin(test_ids)].reset_index(drop=True)

    return chart_train, chart_test, experiment_train, experiment_test


def _detect_cycle_phases(signal: pd.DataFrame, threshold_pct: float = 0.18) -> dict:
    """Internal helper to detect phase transition indices for a single cycle.

    Identifies transition frames based on initial gradient and sharpest drops.
    """
    # Filter for time > 0.4 to ignore initialization/pre-injection noise artifacts
    signal_to_use = signal["Einspritzstrom, Ist"][signal["time"] > 0.4]
    len_removed_signal = len(signal[signal["time"] <= 0.4])

    grad = np.gradient(signal_to_use)

    # --- Step A: Find injection phase end ---
    max_grad = np.max(np.abs(grad))
    significant_changes = np.where(np.abs(grad) > (max_grad * threshold_pct))[0]
    first_phase_end_idx = significant_changes[0] if len(significant_changes) > 0 else 0

    # --- Step B: Find holding and dosage drop peaks ---
    search_area = -grad[first_phase_end_idx:]
    peaks, props = find_peaks(search_area, height=0.03)
    peaks = peaks + first_phase_end_idx

    if len(peaks) < 2:
        return {}  # Graceful exit if the signal shape is unexpected

    strongest_indices = np.argsort(props["peak_heights"])[-2:]
    top_two_peaks = np.sort(peaks[strongest_indices])

    # --- Step C: Find decompression phase end ---
    last_phase_value = np.max(grad[int(top_two_peaks.max()) :])
    last_phase_index = np.where(np.isclose(grad, last_phase_value))[0]

    return {
        "yellow_injection": int(first_phase_end_idx) + len_removed_signal,
        "dark_blue_holding_pressure": int(top_two_peaks[0]) + len_removed_signal,
        "green_dosage": int(top_two_peaks[1]) + len_removed_signal - 1,
        "red_decompression": int(last_phase_index[0]) + len_removed_signal + 1,
    }


def transform_time_series_into_phases(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates phase calculation and reshapes the dataframe for tsfresh.

    Iterates safely through available IDs, slices sensor variables into
    isolated horizontal blocks, and appends time tracking features.

    Args:
        df: Cleaned long-form time-series data containing columns 'id' and 'time'.

    Returns:
        full_joined_df: A phase-separated dataframe ready for tsfresh extraction.
    """
    unique_ids = df["id"].unique()
    all_dfs = []

    for cycle_id in unique_ids:
        # Filter dataframe for the current ID safely
        df_id = df[df["id"] == cycle_id].reset_index(drop=True)

        # Detect phase transitions for this specific ID
        phase_indices = _detect_cycle_phases(df_id, threshold_pct=0.18)
        if not phase_indices:
            continue

        # Extract timestamps for the phase boundaries
        t_inj = df_id.loc[phase_indices["yellow_injection"], "time"]
        t_hold = df_id.loc[phase_indices["dark_blue_holding_pressure"], "time"]
        t_dose = df_id.loc[phase_indices["green_dosage"], "time"]
        t_decomp = df_id.loc[phase_indices["red_decompression"], "time"]

        # Drop non-feature columns before adding suffixes
        features_only = df_id.drop(columns=["time", "id"], errors="ignore")

        # Slice features into distinct phase blocks based on time boundaries
        inj_df = features_only[df_id["time"] <= t_inj].add_suffix("_inj")
        hold_df = features_only[(df_id["time"] > t_inj) & (df_id["time"] <= t_hold)].add_suffix("_hold")
        dose_df = features_only[(df_id["time"] > t_hold) & (df_id["time"] <= t_dose)].add_suffix("_dosage")
        decomp_df = features_only[(df_id["time"] > t_dose) & (df_id["time"] <= t_decomp)].add_suffix("_comp")

        # Horizontally merge phases. Missing rows in shorter phases auto-fill with NaN
        dfs_to_concat = [inj_df, hold_df, dose_df, decomp_df]
        joined_df = pd.concat([d.reset_index(drop=True) for d in dfs_to_concat], axis=1)

        # Append identification keys for tsfresh parsing
        joined_df["id"] = int(cycle_id)
        joined_df["time_stamp"] = joined_df.index + 1
        all_dfs.append(joined_df)

    if not all_dfs:
        return pd.DataFrame()

    full_joined_df = pd.concat(all_dfs, ignore_index=True)
    return full_joined_df



def extract_tsfresh_features(phase_df: pd.DataFrame) -> pd.DataFrame:
    """Melts the sparse phase dataframe to drop NaNs and extracts tsfresh features.
    
    This function implements long-format compression to speed up te engine significantly.
    """
    if phase_df.empty:
        return pd.DataFrame()
    
    # 1. Transform the wide table into a long one 
    long_df = phase_df.melt(id_vars=['id', 'time_stamp'], var_name='kind', value_name='value')

    # 2. Drop the rows with NaNs safely
    long_df = long_df.dropna(subset=['value'])

    # 3. Extract features utilizing the compressed long structure
    extracted_features = extract_features(long_df,
                                          column_id='id',
                                          column_sort='time_stamp',
                                          column_kind='kind',
                                          column_value='value'
    )
    
    return extracted_features


def scale_and_align_features(train_static: pd.DataFrame,
                             test_static: pd.DataFrame,
                             train_features: pd.DataFrame,
                             test_features: pd.DataFrame,
                             parameters: Dict) -> Tuple[pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.Series,
                                                        pd.Series]:
    """Merges static metrics, imputes raw values, filters relevant features via tsfresh,

    and safely scales inputs using an isolated scaler configuration.
    """
    target_col = parameters["target_column"]

    # 1. Merge static parameters and labels with tsfresh features matching on id
    train_master = pd.merge(train_static, train_features, left_on="id", right_index=True, how="inner")
    test_master = pd.merge(test_static, test_features, left_on="id", right_index=True, how="inner")

    # 2. Isolate target labels from independent features
    X_train = train_master.drop(columns=["id", target_col])
    y_train = train_master[target_col]

    X_test_raw = test_master.drop(columns=["id", target_col])
    y_test = test_master[target_col]

    # 3. Safe imputation: Automatic handling of nested inf, -inf, and NaN math values
    # (Modifies the datafrmaes in-place)
    impute(X_train)
    impute(X_test_raw)

    # 4. Target Label Encoding
    l_encoder = LabelEncoder()
    y_train_enc = pd.Series(l_encoder.fit_transform(y_train), index=X_train.index)
    y_test_enc = pd.Series(l_encoder.transform(y_test), index=X_test_raw.index)

    # 5. Feature Selection
    # Statistical analysis drops irrelevant/flat-lined signal features
    X_train_selected = select_features(X_train, y_train_enc)

    # 6. Align Columns: force the test matrix columns to mirror the training columns exactly
    X_test_selected = X_test_raw.reindex(columns=X_train_selected.columns, fill_value=0)

    # 7. Initialize and fit scaling transformations safely
    scaler = MinMaxScaler()
    # Scale arrays and reconstruct back into clear Pandas DataFrames
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_selected), columns=X_train_selected.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_selected), columns=X_train_selected.columns)


    # Convert the label series into dataframes to save them as Parquet
    y_train_df = y_train_enc.to_frame(name=target_col)
    y_test_df = y_test_enc.to_frame(name=target_col)

    return X_train_scaled, X_test_scaled, y_train_df, y_test_df