# SuperKart Model Deployment Project - Spyder Compatible Full-Score Validation Script
# ------------------------------------------------------------------------------
# How to run:
# 1) Keep this file and ext-SuperKart.csv in the same folder.
# 2) In Anaconda Prompt run: pip install -r SuperKart_Spyder_Full_Score_requirements.txt
# 3) Open this file in Spyder, set working directory to this folder, and press F5.
#
# Important correction from project Q&A:
# The supplied low-code notebook originally used make_column_transformer only for
# categorical columns. Since make_column_transformer defaults to remainder='drop',
# numerical columns were dropped. This script validates that issue and uses
# remainder='passthrough' to keep numerical features in the model pipeline.

from pathlib import Path
import json
import time
import textwrap
import shutil

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = BASE / "ext-SuperKart.csv"
##DATA_FILE = "D:\\OneDrive - Prodapt Solutions Private Limited\\Downloads\\SuperKartProject\\ext-SuperKart.csv"


OUTPUT_DIR = BASE / "superkart_fullscore_outputs"
##OUTPUT_DIR = "D:\\OneDrive - Prodapt Solutions Private Limited\\Downloads\\SuperKartProject\\superkart_fullscore_outputs"

PLOT_DIR = OUTPUT_DIR / "plots"
##PLOT_DIR = "D:\\OneDrive - Prodapt Solutions Private Limited\\Downloads\\SuperKartProject\\superkart_fullscore_outputs\\plots"

DEPLOY_DIR = BASE / "superkart_fullscore_deployment"
##DEPLOY_DIR = "D:\\OneDrive - Prodapt Solutions Private Limited\\Downloads\\SuperKartProject\\superkart_fullscore_deployment"

BACKEND_DIR = DEPLOY_DIR / "backend"
##BACKEND_DIR = "D:\\OneDrive - Prodapt Solutions Private Limited\\Downloads\\SuperKartProject\\superkart_fullscore_deployment\\backend"

FRONTEND_DIR = DEPLOY_DIR / "frontend"
##FRONTEND_DIR = "D:\\OneDrive - Prodapt Solutions Private Limited\\Downloads\\SuperKartProject\\superkart_fullscore_deployment\\frontend"


TARGET = "Product_Store_Sales_Total"
RANDOM_STATE = 1
CURRENT_YEAR = 2025
TEST_SIZE = 0.30
PERISHABLES = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]


def make_dirs():
    for path in [OUTPUT_DIR, PLOT_DIR, BACKEND_DIR, FRONTEND_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def evaluate_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return rmse, mae, r2, mape


def feature_engineering(df):
    df = df.copy()
    df["Product_Id_char"] = df["Product_Id"].str[:2]
    df["Store_Age_Years"] = CURRENT_YEAR - df["Store_Establishment_Year"]
    df["Product_Type_Category"] = np.where(df["Product_Type"].isin(PERISHABLES), "Perishables", "Non Perishables")
    return df


def prepare_modeling_data(df):
    df = feature_engineering(df)
    return df.drop(columns=["Product_Id", "Product_Type", "Store_Id", "Store_Establishment_Year"])


def corrected_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Corrected low-code notebook pipeline: keep all non-categorical columns.
    return make_column_transformer(
        (Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def defective_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Original template issue: numerical columns are dropped because remainder defaults to 'drop'.
    return make_column_transformer(
        (Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_features),
        verbose_feature_names_out=False,
    )


def save_bar(series, title, xlabel, ylabel, filename, top=None, rotation=25):
    series = series.sort_values(ascending=False)
    if top is not None:
        series = series.head(top)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=180)
    plt.close(fig)


def run_eda(raw):
    print("Running EDA and summary generation...", flush=True)
    quality_rows = []
    for col in raw.columns:
        quality_rows.append({
            "Column": col,
            "Data_Type": str(raw[col].dtype),
            "Missing_Count": int(raw[col].isna().sum()),
            "Missing_Pct": round(float(raw[col].isna().mean() * 100), 4),
            "Unique_Values": int(raw[col].nunique()),
        })
    pd.DataFrame(quality_rows).to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)
    pd.DataFrame({
        "Metric": ["Rows", "Columns", "Missing values", "Duplicate rows"],
        "Value": [raw.shape[0], raw.shape[1], int(raw.isna().sum().sum()), int(raw.duplicated().sum())],
    }).to_csv(OUTPUT_DIR / "shape_missing_duplicates.csv", index=False)
    raw.describe(include="all").T.to_csv(OUTPUT_DIR / "statistical_summary.csv")

    numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
    raw[numeric_cols].corr().to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    outlier_rows = []
    for col in numeric_cols:
        q1, q3 = raw[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((raw[col] < lower) | (raw[col] > upper)).sum())
        outlier_rows.append({
            "Variable": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower,
            "Upper_Bound": upper,
            "Outlier_Count": count,
            "Treatment": "Retained - plausible commercial observations; tree ensemble models are robust.",
        })
    pd.DataFrame(outlier_rows).to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)

    for col in ["Product_Type", "Product_Sugar_Content", "Store_Size", "Store_Location_City_Type", "Store_Type", "Store_Id"]:
        raw.groupby(col)[TARGET].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False).to_csv(OUTPUT_DIR / f"sales_by_{col}.csv")

    save_bar(raw.groupby("Product_Type")[TARGET].sum(), "Total Sales by Product Type", "Product Type", "Total Sales Revenue", "sales_by_product_type.png", top=10)
    save_bar(raw.groupby("Store_Type")[TARGET].sum(), "Total Sales by Store Type", "Store Type", "Total Sales Revenue", "sales_by_store_type.png", rotation=10)
    save_bar(raw.groupby("Store_Location_City_Type")[TARGET].sum(), "Total Sales by City Tier", "City Tier", "Total Sales Revenue", "sales_by_city_tier.png", rotation=0)
    save_bar(raw.groupby("Product_Sugar_Content")[TARGET].sum(), "Total Sales by Sugar Content", "Sugar Content", "Total Sales Revenue", "sales_by_sugar_content.png", rotation=10)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(raw[TARGET], bins=35)
    ax.set_title("Distribution of Product Store Sales Total")
    ax.set_xlabel("Product Store Sales Total")
    ax.set_ylabel("Number of Records")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "target_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(raw["Product_MRP"], raw[TARGET], s=8, alpha=0.45)
    ax.set_title("Product MRP vs Product Store Sales Total")
    ax.set_xlabel("Product MRP")
    ax.set_ylabel("Product Store Sales Total")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "mrp_vs_sales.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    raw[["Product_Weight", "Product_Allocated_Area", "Product_MRP", TARGET]].boxplot(ax=ax)
    ax.set_title("Outlier Check for Numerical Variables")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "numeric_boxplots.png", dpi=180)
    plt.close(fig)


def run_models(modeling_data):
    print("Running model building, tuning and serialization...", flush=True)
    X = modeling_data.drop(columns=TARGET)
    y = modeling_data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    X_test.head(20).to_csv(OUTPUT_DIR / "batch_prediction_sample.csv", index=False)

    # Validate the pipeline issue from Q&A.
    bug_rows = []
    bug_models = {
        "Original notebook bug - numeric dropped": defective_preprocessor(X_train),
        "Corrected pipeline - remainder passthrough": corrected_preprocessor(X_train),
    }
    for name, prep in bug_models.items():
        pipe = Pipeline([("preprocessor", prep), ("model", RandomForestRegressor(n_estimators=80, random_state=RANDOM_STATE, n_jobs=1))])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse, mae, r2, mape = evaluate_regression(y_test, pred)
        bug_rows.append({"Pipeline": name, "Model": "Random Forest", "Test_RMSE": rmse, "Test_MAE": mae, "Test_R2": r2, "Test_MAPE_pct": mape})
    bug_df = pd.DataFrame(bug_rows)
    bug_df.to_csv(OUTPUT_DIR / "pipeline_bug_validation.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    bug_df.set_index("Pipeline")["Test_R2"].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Impact of Correcting the Preprocessing Pipeline")
    ax.set_ylabel("Test R-squared")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pipeline_correction_impact.png", dpi=180)
    plt.close(fig)

    model_specs = {
        "Random Forest - Baseline": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting - Baseline": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    fitted_models = {}
    performance_rows = []
    for name, model in model_specs.items():
        print("Fitting", name, flush=True)
        pipe = Pipeline([("preprocessor", corrected_preprocessor(X_train)), ("model", model)])
        start = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - start
        fitted_models[name] = pipe
        for split_name, X_split, y_split in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
            pred = pipe.predict(X_split)
            rmse, mae, r2, mape = evaluate_regression(y_split, pred)
            performance_rows.append({"Model": name, "Split": split_name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_pct": mape, "Fit_Time_Seconds": fit_time})

    tuning_specs = [
        (
            "Random Forest - Tuned",
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [100, 150],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 2],
                "model__min_samples_split": [2, 5],
                "model__max_features": [0.7],
            },
            2,
        ),
        (
            "Gradient Boosting - Tuned",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.08, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [1, 2],
                "model__subsample": [0.8, 1.0],
            },
            2,
        ),
    ]
    tuning_rows = []
    for name, model, params, n_iter in tuning_specs:
        print("Tuning", name, flush=True)
        pipe = Pipeline([("preprocessor", corrected_preprocessor(X_train)), ("model", model)])
        search = RandomizedSearchCV(pipe, param_distributions=params, n_iter=n_iter, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=1)
        start = time.time()
        search.fit(X_train, y_train)
        fit_time = time.time() - start
        best_pipe = search.best_estimator_
        fitted_models[name] = best_pipe
        tuning_rows.append({"Model": name, "Best_CV_R2": search.best_score_, "Best_Parameters": json.dumps(search.best_params_), "Search_Time_Seconds": fit_time})
        for split_name, X_split, y_split in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
            pred = best_pipe.predict(X_split)
            rmse, mae, r2, mape = evaluate_regression(y_split, pred)
            performance_rows.append({"Model": name, "Split": split_name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_pct": mape, "Fit_Time_Seconds": fit_time})

    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(OUTPUT_DIR / "model_performance_comparison.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUTPUT_DIR / "hyperparameter_tuning_summary.csv", index=False)
    test_perf = performance_df[performance_df["Split"] == "Test"].copy().sort_values(["R2", "RMSE"], ascending=[False, True])
    best_model_name = test_perf.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, OUTPUT_DIR / "superkart_best_model_pipeline.joblib")

    loaded_model = joblib.load(OUTPUT_DIR / "superkart_best_model_pipeline.joblib")
    final_pred = loaded_model.predict(X_test)
    final_rmse, final_mae, final_r2, final_mape = evaluate_regression(y_test, final_pred)
    pd.DataFrame({"Actual": y_test.values, "Predicted": final_pred, "Residual": y_test.values - final_pred}).to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)

    # Feature importance from tree model after one-hot encoding.
    try:
        feature_names = loaded_model.named_steps["preprocessor"].get_feature_names_out()
        model_importance = loaded_model.named_steps["model"].feature_importances_
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": model_importance}).sort_values("Importance", ascending=False)
    except Exception:
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": np.nan})
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    test_perf.set_index("Model")["R2"].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison - Test R-squared")
    ax.set_ylabel("R-squared")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_test_r2_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    test_perf.set_index("Model")["RMSE"].plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison - Test RMSE")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_test_rmse_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, final_pred, s=8, alpha=0.5)
    low = min(y_test.min(), final_pred.min())
    high = max(y_test.max(), final_pred.max())
    ax.plot([low, high], [low, high])
    ax.set_title("Final Model: Actual vs Predicted Sales")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.hist(y_test.values - final_pred, bins=35)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Records")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "residual_distribution.png", dpi=180)
    plt.close(fig)

    if not importance_df["Importance"].isna().all():
        fig, ax = plt.subplots(figsize=(9, 5.2))
        importance_df.head(10).sort_values("Importance").set_index("Feature")["Importance"].plot(kind="barh", ax=ax)
        ax.set_title("Top Model Feature Drivers")
        ax.set_xlabel("Tree-based Feature Importance")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "feature_importance.png", dpi=180)
        plt.close(fig)

    summary = f"""
SuperKart Full-Score Spyder Validation Summary
==============================================
Dataset rows: {modeling_data.shape[0]:,}
Modeling columns after feature engineering: {modeling_data.shape[1]}
Train/test split: 70/30, random_state=1

Critical preprocessing correction:
- Original low-code notebook dropped numerical predictors because remainder defaults to 'drop'.
- Corrected preprocessor uses remainder='passthrough'.
- This keeps Product_Weight, Product_Allocated_Area, Product_MRP, and Store_Age_Years.

Selected final model: {best_model_name}
Test R-squared: {final_r2:.4f}
Test RMSE: {final_rmse:.2f}
Test MAE: {final_mae:.2f}
Test MAPE: {final_mape:.2f}%

The serialized model was loaded successfully and generated predictions on the test set.
""".strip()
    (OUTPUT_DIR / "final_model_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    return loaded_model, X_test


def create_deployment_files(X_test):
    print("Creating deployment files...", flush=True)
    shutil.copy2(OUTPUT_DIR / "superkart_best_model_pipeline.joblib", BACKEND_DIR / "superkart_best_model_pipeline.joblib")
    (BACKEND_DIR / "app.py").write_text(textwrap.dedent('''
        import joblib
        import pandas as pd
        from flask import Flask, request, jsonify

        app = Flask("superkart_sales_api")
        model = joblib.load("superkart_best_model_pipeline.joblib")
        REQUIRED_COLUMNS = [
            "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
            "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
            "Store_Age_Years", "Product_Type_Category"
        ]

        @app.get("/")
        def home():
            return jsonify({"message": "SuperKart sales forecasting API is running."})

        @app.post("/v1/predict")
        def predict_sales():
            try:
                payload = request.get_json(force=True)
                records = [payload] if isinstance(payload, dict) else payload
                df = pd.DataFrame(records)
                missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if missing:
                    return jsonify({"error": "Missing required columns", "missing_columns": missing}), 400
                predictions = model.predict(df[REQUIRED_COLUMNS]).tolist()
                return jsonify({"predictions": predictions})
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        if __name__ == "__main__":
            app.run(host="0.0.0.0", port=7860)
    ''').strip(), encoding="utf-8")
    (BACKEND_DIR / "requirements.txt").write_text("\n".join([
        "Flask>=3.0,<4.0", "gunicorn>=22,<24", "pandas>=2.2,<3.0", "numpy>=1.26,<3.0", "scikit-learn>=1.6,<1.7", "joblib>=1.4,<2.0"
    ]), encoding="utf-8")
    (BACKEND_DIR / "Dockerfile").write_text(textwrap.dedent('''
        FROM python:3.11-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY . .
        EXPOSE 7860
        CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
    ''').strip(), encoding="utf-8")

    (FRONTEND_DIR / "app.py").write_text(textwrap.dedent('''
        import streamlit as st
        import pandas as pd
        import requests

        st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")
        st.title("SuperKart Quarterly Sales Forecast")
        API_URL = st.sidebar.text_input("Backend API URL", "https://<your-backend-space>.hf.space/v1/predict")

        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
                Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
                Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.05, step=0.01)
                Product_MRP = st.number_input("Product MRP", min_value=0.0, value=141.62)
                Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])
            with c2:
                Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
                Store_Location_City_Type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
                Store_Type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
                Store_Age_Years = st.number_input("Store Age Years", min_value=0, max_value=50, value=16)
                Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])
            payload = {
                "Product_Weight": Product_Weight, "Product_Sugar_Content": Product_Sugar_Content,
                "Product_Allocated_Area": Product_Allocated_Area, "Product_MRP": Product_MRP,
                "Store_Size": Store_Size, "Store_Location_City_Type": Store_Location_City_Type,
                "Store_Type": Store_Type, "Product_Id_char": Product_Id_char,
                "Store_Age_Years": Store_Age_Years, "Product_Type_Category": Product_Type_Category,
            }
            if st.button("Predict Sales", type="primary"):
                response = requests.post(API_URL, json=payload, timeout=20)
                if response.ok:
                    pred = response.json()["predictions"][0]
                    st.success(f"Predicted Product Store Sales Total: {pred:,.2f}")
                else:
                    st.error(response.text)
        with tab2:
            uploaded = st.file_uploader("Upload a CSV with the required columns", type="csv")
            if uploaded is not None:
                df = pd.read_csv(uploaded)
                st.dataframe(df.head())
                if st.button("Run Batch Prediction"):
                    response = requests.post(API_URL, json=df.to_dict(orient="records"), timeout=60)
                    if response.ok:
                        out = df.copy(); out["Predicted_Sales"] = response.json()["predictions"]
                        st.dataframe(out.head())
                        st.download_button("Download Predictions", out.to_csv(index=False), "superkart_batch_predictions.csv")
                    else:
                        st.error(response.text)
    ''').strip(), encoding="utf-8")
    (FRONTEND_DIR / "requirements.txt").write_text("\n".join(["streamlit>=1.43,<2.0", "pandas>=2.2,<3.0", "requests>=2.32,<3.0"]), encoding="utf-8")
    (FRONTEND_DIR / "Dockerfile").write_text(textwrap.dedent('''
        FROM python:3.11-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY . .
        EXPOSE 8501
        CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ''').strip(), encoding="utf-8")
    X_test.head(20).to_csv(FRONTEND_DIR / "batch_prediction_sample.csv", index=False)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}. Keep ext-SuperKart.csv in the same folder as this script.")
    make_dirs()
    print("Loading dataset...", flush=True)
    raw = pd.read_csv(DATA_FILE)
    print(f"Loaded dataset shape: {raw.shape}", flush=True)
    run_eda(raw)
    modeling_data = prepare_modeling_data(raw)
    modeling_data.to_csv(OUTPUT_DIR / "modeling_dataset_after_feature_engineering.csv", index=False)
    model, X_test = run_models(modeling_data)
    create_deployment_files(X_test)
    print("Done. Review outputs in:", OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
