# Viraj – Experiment Log (Validation only)

I computed the FaceNet embeddings once, then tried different values of C for Logistic Regression.

| What I changed              | Validation accuracy | What I think happened |
|-----------------------------|---------------------|-----------------------|
| C=1 (starting point)        |                     | Baseline              |
| C=0.1                       |                     | Stronger regularisation – maybe underfit |
| C=10                        |                     | Weaker regularisation – maybe overfit |
| C=100                       |                     | Even weaker – risk of overfitting |

**Final choice:** C = ______ (fill after running the experiments)

**Why I chose it:**
(write 1–2 sentences after you see the numbers)