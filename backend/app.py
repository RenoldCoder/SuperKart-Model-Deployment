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