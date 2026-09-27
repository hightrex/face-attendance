# Face-Based Attendance Identification
**CSI6209 — Assessment 2: Applied Artificial Intelligence**

A comparative study of **Classical Machine Learning** (HOG / PCA + SVM) versus **Deep Learning** (Pretrained FaceNet + Classifier) for automated face-based attendance tracking on the LFW (Labeled Faces in the Wild) dataset.

---

## 👥 Team Roles & Responsibilities

| Name | Role | Primary Responsibilities | Main Files |
| :--- | :--- | :--- | :--- |
| **Tshering** | Data & Demo Lead | Dataset loader, stratified train/val/test splits, demo notebook, slide deck | `src/data.py` |
| **Maureen** | Classical ML Lead | Preprocessing (grayscale/resize), HOG/PCA feature extraction, SVM Pipeline | `src/classical.py`, `report/references.csv` |
| **Viraj** | Deep Learning Lead | Pretrained FaceNet feature extraction (`facenet-pytorch`), Logistic Regression classifier | `src/deep.py`, `requirements.txt` |
| **Hightrex** | Evaluation & Repo Owner | Evaluation metrics (Accuracy, Macro-F1), perturbation tests (blur/noise), model fusion, runner scripts | `src/evaluate.py`, `train_all.py`, `run_final.py` |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/hightrex/face-attendance.git
cd face-attendance
```

### 2. Set Up Virtual Environment
Always work inside an isolated virtual environment to prevent package version conflicts:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```text
face-attendance/
├── models/             # Saved model checkpoints (*.joblib, *.pt) - gitignored
├── report/             # Report drafts, decision records, and references
│   └── decisions.md    # Shared model contract and dataset agreements
├── results/            # Generated figures, confusion matrices, and metrics CSVs
├── src/                # Core implementation source code
│   ├── data.py         # LFW data loader & stratified splits (Tshering)
│   ├── classical.py    # Classical pipeline: HOG/PCA + SVM (Maureen)
│   ├── deep.py         # Deep pipeline: FaceNet + Logistic Regression (Viraj)
│   └── evaluate.py     # Evaluation metrics & robustness tests (Hightrex)
├── train_all.py        # Orchestration script to fit and save all models (Hightrex)
├── run_final.py        # Final evaluation run on locked test set (Hightrex)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation and guide
```

---

## 📋 Team Workflow Rules

1. **Keep `main` Clean**: Never push broken or untested code directly to `main`.
2. **Branching**: Each member creates a branch for their work:
   ```bash
   git checkout -b feature/<your-name>-<feature-description>
   ```
3. **Model Contract**: Every model must implement `classes_` (sorted name list) and `predict_proba(X)` (probability table). See [report/decisions.md](report/decisions.md) for specifications.
4. **Data Freeze**: The test set is locked and must only be evaluated once during the final evaluation phase.
