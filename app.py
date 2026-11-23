from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "backend_files" / "superkart_model.joblib"
app = Flask(__name__)
CORS(app)
model = joblib.load(MODEL_PATH)

REQUIRED_COLUMNS = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category"
]

@app.get("/")
def index():
    return jsonify({"status": "ok", "service": "SuperKart Sales Forecast API"})

@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": MODEL_PATH.exists()})

@app.post("/v1/predict")
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON payload is required"}), 400
    missing = [col for col in REQUIRED_COLUMNS if col not in payload]
    if missing:
        return jsonify({"error": "Missing fields", "missing_fields": missing}), 400
    frame = pd.DataFrame([{col: payload[col] for col in REQUIRED_COLUMNS}])
    prediction = float(model.predict(frame)[0])
    return jsonify({"predicted_sales": round(prediction, 2), "prediction": round(prediction, 2)})

@app.post("/v1/predictbatch")
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "CSV file is required under the 'file' field"}), 400
    uploaded = request.files["file"]
    try:
        frame = pd.read_csv(BytesIO(uploaded.read()))
    except Exception as exc:
        return jsonify({"error": f"Invalid CSV: {exc}"}), 400
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        return jsonify({"error": "Missing columns", "missing_columns": missing}), 400
    predictions = model.predict(frame[REQUIRED_COLUMNS])
    return jsonify({str(i): round(float(value), 2) for i, value in enumerate(predictions)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
