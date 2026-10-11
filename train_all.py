

from pathlib import Path
import joblib

from src.data import load_data
from src.classical import ClassicalModel
from src.deep import DeepModel

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def main():
    print("=== Loading data ===")
    X_train, X_val, X_test, y_train, y_val, y_test, names, shape = load_data()

    # Classical model (HOG + SVM)
    # Maureen can change the settings here later
    print("\n=== Training Classical Model (HOG) ===")
    classical = ClassicalModel(feature_type="hog", C=10.0)
    classical.fit(X_train, y_train)
    joblib.dump(classical, MODELS_DIR / "classical_hog.joblib")
    print("Saved models/classical_hog.joblib")

    # Deep model (FaceNet + Logistic Regression)
    # Viraj can change C here later
    print("\n=== Training Deep Model ===")
    deep = DeepModel(C=1.0, batch_size=32)
    deep.fit(X_train, y_train)
    joblib.dump(deep, MODELS_DIR / "deep_facenet.joblib")
    print("Saved models/deep_facenet.joblib")

    print("\nDone. Both models are saved.")


if __name__ == "__main__":
    main()