import math
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
FEATURES_BY_MODE = {
    "pregnancy": [
        "systolicBP_mmHg",
        "diastolicBP_mmHg",
        "heartRate_bpm",
        "temperature_C",
        "bloodSugar_mg_dL",
        "recordWeight_kg",
        "sleepHours",
    ],
    "child": [
        "ageMonths_derived",
        "systolicBP_mmHg",
        "diastolicBP_mmHg",
        "heartRate_bpm",
        "temperature_C",
        "bloodSugar_mg_dL",
        "recordWeight_kg",
        "sleepHours",
    ],
}


def load_model(filename):
    model_path = BASE_DIR / filename
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required model file is missing: {model_path}. "
            "Train the models with `python train.py`."
        )
    return joblib.load(model_path)


app = Flask(__name__)
preg_model = load_model("pregnancy_model.pkl")
child_model = load_model("child_model.pkl")


@app.get("/health")
def health():
    return jsonify({"success": True, "status": "ready"})


@app.post("/predict-alert")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "error": "A JSON object is required."}), 400

    mode = data.get("mode")
    features = data.get("features")
    if mode not in FEATURES_BY_MODE:
        return jsonify({"success": False, "error": "Invalid mode."}), 400
    if not isinstance(features, dict):
        return jsonify({"success": False, "error": "Features must be a JSON object."}), 400

    feature_names = FEATURES_BY_MODE[mode]
    missing = [name for name in feature_names if name not in features]
    if missing:
        return jsonify({
            "success": False,
            "error": "Missing required model features.",
            "missing_features": missing,
        }), 400

    values = {}
    for name in feature_names:
        value = features[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return jsonify({
                "success": False,
                "error": f"{name} must be a finite number.",
            }), 400
        values[name] = value

    model = preg_model if mode == "pregnancy" else child_model
    try:
        row = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
        prediction = model.predict(row)[0]
        if prediction not in (0, 1, False, True):
            raise ValueError("The classifier returned an unsupported label.")
        model_requires_attention = prediction == 1 or prediction is True

        guardrail_triggered = (
            values["systolicBP_mmHg"] > 140
            or values["systolicBP_mmHg"] < 90
            or values["diastolicBP_mmHg"] > 90
            or values["diastolicBP_mmHg"] < 60
            or values["temperature_C"] > 38
            or values["temperature_C"] < 35
            or (
                mode == "pregnancy"
                and (
                    values["heartRate_bpm"] > 110
                    or values["heartRate_bpm"] < 50
                )
            )
        )

        return jsonify({
            "success": True,
            "model_assessed": True,
            "model_requires_attention": bool(model_requires_attention),
            "guardrail_triggered": bool(guardrail_triggered),
            "requires_attention": bool(model_requires_attention or guardrail_triggered),
        })
    except Exception:
        app.logger.exception("Unable to calculate health screening prediction")
        return jsonify({
            "success": False,
            "error": "The screening model could not assess these measurements.",
        }), 500


if __name__ == "__main__":
    host = os.environ.get("ML_HOST", "127.0.0.1")
    port = int(os.environ.get("ML_PORT", "5001"))
    app.run(host=host, port=port, debug=False)
