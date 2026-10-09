from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from .data import load_data

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def train_and_save():
    x_train, x_val, x_test, y_train, y_val, y_test, names, shape = load_data()
    model = Pipeline([("svc", SVC(kernel="linear", C=0.1))])
    x_final = np.concatenate([x_train, x_val])
    y_final = np.concatenate([y_train, y_val])
    print(f"Training on {len(y_final)} images; testing on {len(y_test)} images")
    model.fit(x_final, y_final)
    pred = model.predict(x_test)
    print(f"Test accuracy: {accuracy_score(y_test, pred):.2%}")
    print(classification_report(y_test, pred, zero_division=0))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "face_recognition_model.pkl")
    joblib.dump({"names": names, "shape": shape}, ARTIFACT_DIR / "face_metadata.pkl")
    print("Model and metadata saved to artifacts/")


if __name__ == "__main__":
    train_and_save()
