# Project Architecture & Decisions — CSI6209 Assessment 2

## 1. Shared Model Contract
Every model implemented in this project (both classical and deep learning) must adhere to the following contract to ensure compatibility with evaluation and fusion pipelines:

1. **`classes_` Attribute**:
   - An array-like / list of all person names in strictly sorted alphabetical order.
   - Example: `['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', ...]`

2. **`predict_proba(X)` Method**:
   - Accepts an input batch `X` of face images.
   - Returns a 2D numpy array of probabilities with shape `(n_samples, n_classes)`.
   - Each row corresponds to a single image and sums to 1.0 across classes.
   - Column index `j` corresponds to class `classes_[j]`.

---

## 2. Dataset Protocol (Data Freeze)
- **Source**: LFW (Labeled Faces in the Wild) via `sklearn.datasets.fetch_lfw_people`.
- **Target Filter**: `min_faces_per_person` agreed upon by the group (e.g. 20 faces).
- **Split Ratio**: 60% Train, 20% Validation (hyperparameter tuning), 20% Test (evaluation).
- **Stratification**: All splits must be stratified to preserve class proportions.
- **Random Seed**: Fixed at `random_state=42` to guarantee identical splits across all team machines.
- **Rules**:
  - Validation set is used for hyperparameter tuning, model comparison, and threshold selection.
  - Test set remains locked and untouched until the final evaluation run.

---

## 3. Project Milestones & Target Dates
- **T** (Canvas Submission / Presentation Date): *(Insert Date)*
- **T - 10 Days** (Code Freeze & Integration): All models and evaluation scripts functional.
- **T - 7 Days** (Report First Draft): Initial compilation of results, figures, and methodology.
- **T - 4 Days** (Slides & Presentation Ready): Slide deck finalized, rehearsals scheduled.
- **T - 1 Day** (Final Submission): Final report review and submission package ready.
