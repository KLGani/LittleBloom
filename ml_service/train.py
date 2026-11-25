from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
LABEL = "demo_followup_attention"
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


def load_training_data(filename, mode):
    dataset_path = BASE_DIR / filename
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset does not exist: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    required_columns = FEATURES_BY_MODE[mode] + [LABEL]
    missing_columns = [column for column in required_columns if column not in dataset]
    if missing_columns:
        raise ValueError(f"{filename} is missing columns: {', '.join(missing_columns)}")

    training_data = dataset[required_columns]
    if training_data.isnull().any().any():
        raise ValueError(f"{filename} contains missing model inputs or labels.")

    labels = training_data[LABEL]
    if not labels.isin([0, 1]).all() or labels.nunique() < 2:
        raise ValueError(f"{filename} must contain both binary target labels, 0 and 1.")

    return training_data[FEATURES_BY_MODE[mode]], labels


def train():
    print("Loading datasets...")
    training_sets = {
        "pregnancy": load_training_data("pregnancy_mode_1000.csv", "pregnancy"),
        "child": load_training_data("child_mode_1000.csv", "child"),
    }

    for mode, filename in (
        ("pregnancy", "pregnancy_model.pkl"),
        ("child", "child_model.pkl"),
    ):
        features, labels = training_sets[mode]
        print(f"Training {mode.title()} Model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        joblib.dump(model, BASE_DIR / filename)

    print("Training complete! Models saved in ml_service.")


if __name__ == "__main__":
    train()
