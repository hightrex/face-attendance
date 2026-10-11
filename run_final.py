from pathlib import Path
import joblib
import pandas as pd

from src.data import load_data
from src.evaluate import (
    evaluate,
    plot_confusion_matrix,
    plot_accuracy_comparison,
    apply_blur,
    add_gaussian_noise,
)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=== Loading data ===")
    X_train, X_val, X_test, y_train, y_val, y_test, names, shape = load_data()

    results = []

    # ----- Classical -----
    print("\n=== Classical Model ===")
    classical = joblib.load(MODELS_DIR / "classical_hog.joblib")
    res_classical = evaluate(classical, X_test, y_test, name="Classical HOG")
    results.append({"model": "Classical HOG", "condition": "clean", **res_classical})

    plot_confusion_matrix(
        y_test, res_classical["y_pred"], classical.classes_,
        "Confusion Matrix - Classical HOG",
        "cm_classical.png"
    )

    # ----- Deep -----
    print("\n=== Deep Model ===")
    deep = joblib.load(MODELS_DIR / "deep_facenet.joblib")
    res_deep = evaluate(deep, X_test, y_test, name="Deep FaceNet")
    results.append({"model": "Deep FaceNet", "condition": "clean", **res_deep})

    plot_confusion_matrix(
        y_test, res_deep["y_pred"], deep.classes_,
        "Confusion Matrix - Deep FaceNet",
        "cm_deep.png"
    )

    # ----- Comparison bar chart -----
    clean_scores = {
        "Classical HOG": res_classical["accuracy"],
        "Deep FaceNet": res_deep["accuracy"]
    }
    plot_accuracy_comparison(clean_scores)

    # ----- Robustness (blur) -----
    print("\n=== Robustness: Blur ===")
    X_blur = apply_blur(X_test, ksize=5)
    res = evaluate(classical, X_blur, y_test, name="Classical (blur)")
    results.append({"model": "Classical HOG", "condition": "blur", **res})
    res = evaluate(deep, X_blur, y_test, name="Deep (blur)")
    results.append({"model": "Deep FaceNet", "condition": "blur", **res})

    # ----- Robustness (noise) -----
    print("\n=== Robustness: Noise ===")
    X_noise = add_gaussian_noise(X_test, std=0.08)
    res = evaluate(classical, X_noise, y_test, name="Classical (noise)")
    results.append({"model": "Classical HOG", "condition": "noise", **res})
    res = evaluate(deep, X_noise, y_test, name="Deep (noise)")
    results.append({"model": "Deep FaceNet", "condition": "noise", **res})

    # ----- Save numbers -----
    df = pd.DataFrame(results)[["model", "condition", "accuracy", "macro_f1"]]
    df.to_csv(RESULTS_DIR / "final_metrics.csv", index=False)
    print("\nSaved results/final_metrics.csv")
    print(df.to_string(index=False))

    

if __name__ == "__main__":
    main()