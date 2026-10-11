import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Set the project folder and create the figures folder if needed
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Load the model results from the CSV file
df = pd.read_csv(ROOT / "results" / "final_metrics.csv")

# Set the colours and labels for the plots
BLUE, GREY = "#2F6690", "#8FA1A8"
conds = ["clean", "blur", "noise"]
labels = ["Clean", "Blur", "Noise"]

# Create two plots side by side with the same y-axis scale
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)

# Plot accuracy and macro-F1 for both models
for a, metric, title in zip(
    axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro-F1"]
):
    x = np.arange(3)  # Positions for the three test conditions
    w = 0.36  # Width of each bar

    # Get the scores for the classical and deep learning models
    c = df[df.model == "Classical HOG"].set_index("condition").loc[conds, metric].values
    d = df[df.model == "Deep FaceNet"].set_index("condition").loc[conds, metric].values

    # Draw bars for each model
    b1 = a.bar(x - w/2, c, w, color=GREY, label="Classical (HOG + SVM)")
    b2 = a.bar(x + w/2, d, w, color=BLUE, label="Deep (FaceNet + LR)")

    # Add the score above each bar
    for bars in (b1, b2):
        for r in bars:
            a.text(r.get_x() + r.get_width()/2, r.get_height() + 0.015,
                   f"{r.get_height():.3f}", ha="center", fontsize=8.5)

    # Set the x-axis labels, title and y-axis range
    a.set_xticks(x)
    a.set_xticklabels(labels)
    a.set_ylim(0, 1.1)
    a.set_title(title, fontsize=12)

    # Add horizontal grid lines to make scores easier to compare
    a.grid(axis="y", color="#DDE3E5", lw=0.8)
    a.set_axisbelow(True)

# Add the y-axis label and a shared legend for both plots
axes[0].set_ylabel("Score (test set, n = 605)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10)

# Adjust spacing and save the figure as a PNG image
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / "fig_robustness.png", dpi=220)
plt.close(fig)
