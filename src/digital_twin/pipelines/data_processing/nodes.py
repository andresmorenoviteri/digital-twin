"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.3.1
"""

import pandas as pd
import re
from collections import defaultdict
from functools import reduce
from typing import Dict, Any, List


# Groups filenames using regex
def _group_partitions(partition_ids: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    pattern = re.compile(r"(.*_measurement_)\d+(_.*)")

    for p_id in partition_ids:
        match = pattern.match(p_id)
        if match:
            key = match.group(1) + match.group(2)
            groups[key].append(p_id)
    return groups


# Loads, cleans, and merges a single group of files
def _load_and_merge_group(partition_ids: List[str]) -> pd.DataFrame:
    df_list = []
    # Relative path ensures this works on ANY computer, not just yours!
    base_path = "data/01_raw/chart_data"

    for p_id in sorted(partition_ids):
        try:
            df = pd.read_csv(
                f"{base_path}/{p_id}.txt",
                sep=";",
                skiprows=11,
                encoding="utf-16",
                engine="python",
            )
            if df.empty:
                continue

            # Clean trailing columns and whitespaces
            df = df.iloc[:, :-1]
            df.columns = df.columns.str.strip()
            df_list.append(df)
        except Exception:
            continue

    if not df_list:
        return pd.DataFrame()

    # Outer merge all dataframes within this specific group
    return reduce(
        lambda left, right: pd.merge(left, right, on="time", how="outer"), df_list
    )


# The main orchestrator for processing chart_data
def preprocess_and_merge_charts(partitioned_input: Dict[str, Any]) -> pd.DataFrame:
    """Processes, cleans, outer-merges, and concatenates all chart data files."""

    # Step 1: Group the partitions
    groups = _group_partitions(list(partitioned_input.keys()))

    # Step 2: Merge data within each group
    merged_dfs = []
    for key, partition_ids in groups.items():
        df_merge = _load_and_merge_group(partition_ids)
        if not df_merge.empty:
            merged_dfs.append(df_merge)

    if not merged_dfs:
        return pd.DataFrame()

    # Step 3: Add incremental experiment IDs
    all_dfs = []
    for idx, df in enumerate(merged_dfs):
        cleaned_df = df[df["time"] != "-start data-"].dropna(subset=["time"])
        temp_df = cleaned_df.copy()
        temp_df["id"] = idx + 1
        all_dfs.append(temp_df)

    # Step 4: Combine, format types, and sort
    chart_data = pd.concat(all_dfs, ignore_index=True)
    chart_data = chart_data.apply(pd.to_numeric, errors="coerce")
    chart_data = chart_data.sort_values(by=["id", "time"]).reset_index(drop=True)
    cols = ['id'] + [col for col in chart_data.columns if col != 'id']
    chart_data = chart_data[cols]

    return chart_data, cols


# Get only relevant data with labels from experiment_data
def preprocess_experiment_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw injection molding experiment data"""
    columns_to_use = ['id', 'T_melt', 't_holdingpressure', 'Reskühlzeit_t_cooling',
       'p_holdingpressure', 'V_injection', 'T_cooling', 'Zeit', 'Quality']
    df = df[columns_to_use]
    df = df.drop(columns=['V_injection'])
    df = df[df['id'] != 594]
    return df.dropna(subset=["Quality"])


# Get other data from updated chart data from the excel file
def preprocess_charts_excel(df: pd.DataFrame, cols: list):
    """Cleans the raw chart data from the second files"""
    df = df.rename(columns={'ID': 'id'})
    df = df[df['id'] > 120]
    df['id'] = df['id'] + 165
    df = df.drop(columns=['Timestamp', 'Time'])
    df.columns = cols

    return df

# concatenate the two chart dataframes and remove undesired columns
def concat_chart(df1: pd.DataFrame, df2: pd.DataFrame):
    df = pd.concat([df1, df2], ignore_index=True)
    # Drop columns that don't contribute valuable information
    df = df.drop(columns=['Auswerferweg, Ist', 'Werkzeuggeschwindigkeit, Ist', 'Auswerfergeschwindigkeit, Ist', 'Werkzeugweg, Ist'])
    # remove data from id=594, it has missing columns
    df = df[df['id'] != 594]

    return df