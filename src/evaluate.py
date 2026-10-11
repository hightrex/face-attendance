"""
src/evaluate.py
Evaluation metrics, robustness tests and simple plots.
Hightrex - CSI6209 Assessment 2
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(model, X, y, name="model"):
    """
    Test a model that follows the team contract
    (has classes_ and predict_proba).
    Returns accuracy and macro-F1.
    """
    proba = model.predict_proba(X)
    y_pred = np.array(model.classes_)[np.argmax(proba, axis=1)]

    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"[{name}] Accuracy: {acc:.3f}  |  Macro-F1: {macro_f1:.3f}")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "y_pred": y_pred
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Make a confusion matrix and save it."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_accuracy_comparison(results_dict, filename="accuracy_comparison.png"):
    """
    Simple bar chart comparing models.
    results_dict example: {"Classical": 0.82, "Deep": 0.91}
    """
    names = list(results_dict.keys())
    scores = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, scores, color=["#4C72B0", "#C44E52", "#55A868"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison (Test Set)")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    out = RESULTS_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def add_gaussian_noise(images, std=0.05):
    """Add random noise to images."""
    rng = np.random.default_rng(42)
    noisy = images.astype(np.float32) + rng.normal(0, std, images.shape)
    if images.max() <= 1.01:
        return np.clip(noisy, 0.0, 1.0)
    return np.clip(noisy, 0.0, 255.0)


def apply_blur(images, ksize=5):
    """Simple blur using scipy (no OpenCV needed)."""
    from scipy.ndimage import uniform_filter

    blurred = np.zeros_like(images, dtype=np.float32)
    for i, img in enumerate(images):
        if img.ndim == 2:
            blurred[i] = uniform_filter(img.astype(np.float32), size=ksize)
        else:
            for c in range(img.shape[-1]):
                blurred[i, ..., c] = uniform_filter(
                    img[..., c].astype(np.float32), size=ksize
                )
    return blurred