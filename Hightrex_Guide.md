# Hightrex's Guide — CSI6209 Assessment 2

**Your role:** repo owner, evaluation and comparison, fusion (optional extra), and final submission.
**Read first:** `Git_Basics.md`. **Also read:** `CSI6209_Lean_Plan.md` for the big picture and the 30-mark breakdown.

**Your files (nobody else should edit these):** `src/evaluate.py`, `train_all.py` (the list of models — but each person tunes their own settings inside it), `run_final.py`.

Everything below is described so **you write the code yourself**. No code is given — the point is to understand every line you submit, since you may be asked to explain it in the viva.

---

## Phase 1 — Set Up (Week 1)

**Task 1: Create the GitHub repository (Day 1)**
1. On github.com: **New repository** → name it (e.g. `csi6209-face-attendance`) → **Private** → tick "Add a README" and "Add .gitignore: Python" → Create.
2. **Settings → Collaborators → Add people.** Invite your three teammates once you have their GitHub usernames (see the invite script below — it can email them for you).
3. `git clone` it to your computer (see `Git_Basics.md`).
4. Create these empty folders inside it: `src/`, `results/`, `report/`. Add a `.gitignore` line for `.venv/`, `__pycache__/`, `models/`, `test_data/`, `*.joblib`, `*.pt`.
5. Commit and push: "Initial project structure".

**Task 2: Send the invites**
Use the `invite_team.sh` script (see the note at the end of this guide) to invite your three teammates, or invite them manually on GitHub if that's simpler: **Settings → Collaborators → Add people**, type their GitHub username or email, send.

**Task 3: Check the ECU lab PCs (Days 1–3)**
On a lab computer, check that Python, Git, and `pip install` all work, and try installing PyTorch (`pip install torch torchvision`) inside a virtual environment to see if it works and how long it takes. Post the result to the group — if it fails, the group needs a backup plan (train on laptops, demo from a saved model).

**Task 4: Agree the shared "contract" with the team**
Before anyone writes code, get agreement (in the group chat or a meeting) on: every model in the project will have two things — a `classes_` list (the names, sorted) and a `predict_proba(X)` function (returns a table of probabilities, one row per photo, one column per person). This is the "plug" that lets your evaluation code work with anyone's model without you needing to know how it works inside. Write this agreement down in `report/decisions.md`.

**Task 5: Find your dates**
Look up the presentation/submission date on Canvas. Call it **T**. Write the four phase deadlines into `report/decisions.md`: code finished T−10, report draft T−7, slides T−4, submit T−1.

**Done when:** repo exists, teammates are in, the model "contract" is agreed and written down, and the dates are set.

---

## Phase 2 — Build (Weeks 2–3)

Your job this phase is to write the code that **measures** any model, so it's ready the moment Maureen's and Viraj's models exist.

**Task 1: Write `evaluate.py`**
Write a function `evaluate(model, X, y)` that:
1. Calls `model.predict_proba(X)` to get a table of probabilities.
2. For each photo, picks the person with the highest probability as the guess.
3. Calculates **accuracy** (the fraction of guesses that were correct) and **macro-F1** (look this up — in short, it scores each person separately and averages, so someone with only a few photos counts as much as someone with a lot).
4. Look at `sklearn.metrics.accuracy_score`, `sklearn.metrics.precision_recall_fscore_support` (with `average="macro"`), and `sklearn.metrics.ConfusionMatrixDisplay` in the scikit-learn documentation — these do exactly what you need.

**Task 2: Write two "make it worse" functions**
- A blur function: look at `cv2.GaussianBlur` in the OpenCV documentation.
- A noise function: add random values to every pixel (`numpy.random.default_rng().normal(...)`), then clip the result back to 0–255.
Test both by running your evaluate function on the original test photos and on the blurred/noisy ones, and check the accuracy drops as expected.

**Task 3 (optional extra): fusion**
Write a function that combines two models' probability tables, for example a weighted average: `w * probs_a + (1 - w) * probs_b`. Try a few values of `w` (0.0 to 1.0) **on the validation photos** and keep whichever works best; only then check it once on test. Important: before combining, check that both models list people in the **exact same order** (`model_a.classes_` equals `model_b.classes_`) — if they don't match, your combined result will be nonsense.

**Task 4: Test your code before real models exist**
Write a "pretend model" — a small class with a `classes_` list and a `predict_proba` method that just returns random numbers — and run your `evaluate` function on it. If it prints sensible-looking numbers with no red error text, your code works, and Maureen and Viraj can plug their real models in later.

**Task 5: `train_all.py`**
Once Maureen's and Viraj's model files exist, write a short script that: loads the data (Tshering's function), creates one of each model, calls `.fit()` on each with the training photos, and saves each one to a `models/` folder (look up `joblib.dump`). Leave a comment showing each person where to put their own best settings.

**Task 6: Draft the block diagram**
Use draw.io or PowerPoint shapes to draw the whole pipeline: photo → preprocessing → the classical route and the deep route side by side → probabilities → (optional) fusion → predicted name. Save it as a PNG for the report.

**Done when:** `evaluate.py` works correctly on a pretend model, and you understand every metric well enough to explain it.

---

## Phase 3 — Evaluate and Demo (Week 4)

**Task 1: Final run**
Once everyone's final settings are frozen (end of Phase 2), write `run_final.py`: train each model once more with final settings, then run your `evaluate` function on the **test** photos (used only now, for the first time) — clean, then blurred, then noisy. Save every result to a CSV file (one row per model per condition — look up `pandas.DataFrame.to_csv`).

**Task 2: Make the figures**
From your saved CSV, make: a bar chart of clean accuracy per model, a line chart of accuracy vs. blur/noise level per model, and a confusion matrix for the best model. Look up `matplotlib.pyplot` for the first two and `ConfusionMatrixDisplay.from_predictions` for the third.

**Task 3: Measure speed**
Time how long each model takes to predict one photo (`time.perf_counter()` before and after a loop over some test photos, divided by the number of photos). This matters for a real attendance system.

**Task 4: Write up the "why"**
For every result, write in your own words: what happened, why you think it happened, and what would test that idea. This is what earns "insightful interpretation" marks, not just the numbers.

**Task 5: Support the clean-machine test**
Viraj will try to run the whole project on a fresh computer using only your README. Be ready to fix anything in your code that only worked "by accident" on your own machine.

**Task 6: Freeze the code**
Once everything works and is reviewed, this is the point where nobody changes the model code anymore except for real bug fixes.

**Done when:** you have a results CSV, three or more figures, and written interpretations for each.

---

## Phase 4 — Write and Present (Weeks 5–6)

**Your report sections:** the block diagram and overall system description, the evaluation protocol (how splits, metrics, and testing-once work), the results comparison and discussion. Also write one Introduction paragraph explaining what the project compares and why, and outline the report's structure in one sentence.

**Your slides:** how the evaluation works, the results table, the figures, the key insight from comparing the two approaches.

**Your part of the demo:** running the live evaluation on a handful of test photos in front of the class, and reading the results table out loud, explaining what the numbers mean.

**Submission (this is on you):**
1. Copy the finished project to a clean folder, remove `.git`, `.venv`, `__pycache__`, and anything not needed to run it.
2. Make sure the test photos are included (or a script that regenerates them) and that a README explains exactly how to run everything from nothing.
3. Zip it, then **test the zip on a different computer** (or have Viraj do it) before submitting.
4. Submit the report and the zip **one day before the deadline**, so there's time to fix upload problems.

**Done when:** the report sections are written, your slides are ready, and the tested submission zip is uploaded.

---

## Note: the invite script

A separate file, `invite_team.sh`, can send GitHub invites automatically to your teammates' email addresses. It needs the **GitHub CLI** installed and you logged in once. See the comments at the top of that script for exact steps — it only takes a couple of minutes to set up, and after that inviting people is one command.
