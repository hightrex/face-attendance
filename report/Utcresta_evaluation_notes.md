
## What we measure

We use two main metrics:

- **Accuracy** – overall percentage of correct predictions
- **Macro-F1** – average of the F1 score for each person.  
  This is fairer when some people have more photos than others.

## How evaluation works

1. Each model must have:
   - `classes_` (list of person names)
   - `predict_proba(X)` (probability table)

2. We call `predict_proba`, take the highest probability as the guess, then calculate accuracy and Macro-F1.

3. The **test set is only used once** at the end (in `run_final.py`).

## Robustness tests

We also test the models when the images are made worse:

- **Blur** – simple uniform blur
- **Noise** – Gaussian noise added to the pixels

Deep embeddings are usually more robust to these changes than hand-crafted features (HOG/PCA). We should check if that is true in our results.

## Files produced

- `results/cm_classical.png` – confusion matrix for classical model
- `results/cm_deep.png` – confusion matrix for deep model
- `results/accuracy_comparison.png` – bar chart of clean accuracy
- `results/final_metrics.csv` – all numbers in one table