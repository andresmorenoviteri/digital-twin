import shutil, tempfile, os
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File
from typing import List 

from src.digital_twin.pipelines.data_processing.nodes import (
    preprocess_experiment_data,
    process_single_chart_group,
    _load_and_merge_groups_from_paths
)

from src.digital_twin.pipelines.data_science.nodes import (
    transform_time_series_into_phases,
    extract_tsfresh_features
)

from tsfresh.utilities.dataframe_functions import impute_dataframe_range

app = FastAPI()

# Load artifacts once at startup
artifacts = joblib.load("data/06_models/production_artifacts.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
feature_columns = artifacts["feature_columns"]
x_train_columns = artifacts["x_train_columns"]
col_to_max = artifacts["col_to_max"]
col_to_min = artifacts["col_to_min"]
col_to_median = artifacts["col_to_median"]
kind_to_fc_parameters = artifacts["kind_to_fc_parameters"]


DROPS_COLUMNS = ["Auswerferweg, Ist", "Werkzeuggeschwindigkeit, Ist",
             "Auswerfergeschwindigkeit, Ist", "Werkzeugweg, Ist"]


@app.post("/predict")
async def predict(
    chart_files: List[UploadFile] = File(...), # the 3 chart .txt files
    exp_file: UploadFile = File(...) # the excel file
):
    with tempfile.TemporaryDirectory() as tmpdir:
        #save uploaded chart files to a temp folder
        chart_paths = []
        for f in chart_files:
            path = os.path.join(tmpdir, f.filename)
            with open(path, "wb") as buf:
                shutil.copyfileobj(f.file, buf)
            chart_paths.append(path)

        # --- Chart pipeline ---
        chart_data = _load_and_merge_groups_from_paths(chart_paths)
        chart_data, _ = process_single_chart_group(chart_data)
        chart_data = chart_data.drop(columns=DROPS_COLUMNS)
        chart_phases = transform_time_series_into_phases(chart_data)
        tsfresh_features = extract_tsfresh_features(chart_phases, kind_to_fc_parameters=kind_to_fc_parameters)


        # --- Experiment excel pipeline --- 
        exp_path = os.path.join(tmpdir, exp_file.filename)
        with open(exp_path, "wb") as buf:
            shutil.copyfileobj(exp_file.file, buf)
        exp_data = pd.read_excel(exp_path)
        exp_data = preprocess_experiment_data(exp_data)

        # --- Merge ---
        master = pd.merge(exp_data, tsfresh_features, left_on="id", right_index=True, how="inner")
        master = master.drop(columns=['V_injection'], errors='ignore')

        x = master.drop(columns=['id', 'Quality'], errors='ignore')

        # --- Numeric coercion ---
        non_numeric_cols = x.select_dtypes(exclude=['number']).columns
        x[non_numeric_cols] = x[non_numeric_cols].apply(pd.to_numeric, errors='coerce')

        # --- Impute ----
        impute_dataframe_range(x, col_to_max, col_to_min, col_to_median)

        # --- Align + scale ---
        x_selected = x.reindex(columns=feature_columns, fill_value=0)
        x_scaled = pd.DataFrame(scaler.transform(x_selected), columns=feature_columns)
        x_final = x_scaled[x_train_columns]

        # --- Predict ---
        prediction = model.predict(x_final)


    return {
        "prediction": int(prediction[0]),
        "label": "Good" if prediction[0] == 0 else "Faulty"
    }

