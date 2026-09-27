# Maureen's Guide — CSI6209 Assessment 2

**Your role:** the classical machine learning model (hand-made features + SVM), and the reference list.
**Read first:** `Git_Basics.md`. **Also read:** `CSI6209_Lean_Plan.md` for the big picture.

**Your file (nobody else should edit this):** `src/classical.py`. **Also yours:** `report/references.csv`.

Nobody is giving you finished code — this guide tells you *what* to build and which library functions to look up, so you write it and understand every line yourself.

---

## Phase 1 — Set Up (Week 1)

**Task 1:** Follow `Git_Basics.md` to install Git, clone the repo, and make your first small commit (a notes file with your name in it) so you know the routine works.

**Task 2:** Install Python (3.10 or 3.11) and set up a virtual environment for the project (your teammate guide or Hightrex will share the exact commands — it's `python -m venv .venv` then activate it). Install the packages you'll need: `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `joblib`.

**Task 3: Reading, in your own words, in a notes file**
Answer these before you start coding (this becomes your report methodology and your Q&A answers):
- What is a "feature", in one sentence?
- What does HOG (Histogram of Oriented Gradients) measure? (Look up "HOG feature descriptor" — the idea is it describes the direction of edges in small patches of the image, which captures face shape.)
- What is PCA, and what are "eigenfaces"? (Look up "eigenfaces face recognition" — the scikit-learn example "Faces recognition example using eigenfaces and SVMs" is a great starting point to read, though don't copy its code — write your own.)
- What does a Support Vector Machine (SVM) do, and what does its setting `C` control?
- Why should the scaler and PCA be learned only from the training photos, never from validation or test photos? (Hint: think about what would happen if information "leaked" from photos your model is later tested on.)

**Task 4:** Wait for Tshering's data-loading function to be ready, then confirm you can load a batch of training photos on your own computer.

**Done when:** Git works for you, your environment is set up, and your notes answer every question above in your own words.

---

## Phase 2 — Build (Weeks 2–3)

Build `src/classical.py`. It needs to do four things, and you decide how:

**Step 1 — Preprocess.** Turn each colour photo into a small grayscale image (look up `skimage.color.rgb2gray` and `skimage.transform.resize`). A common starting size is 64×64 pixels — smaller is faster, but too small loses detail.

**Step 2 — Two feature extractors.**
- **HOG:** look up `skimage.feature.hog` in the scikit-image documentation. Read what its main settings do: `orientations`, `pixels_per_cell`, `cells_per_block`.
- **PCA (eigenfaces):** flatten each grayscale image into one long row of numbers, then look up `sklearn.decomposition.PCA` — it finds the most important patterns of variation across all your faces (the "eigenfaces") and represents each photo with far fewer numbers.

**Step 3 — Classifier.** Look up `sklearn.svm.SVC` with `probability=True` (so it can output a probability per person, not just a single guess). Its main setting, `C`, controls how strictly it tries to get every training photo right — too high can overfit, too low can underfit.

**Step 4 — Put it together correctly.** This is the most important technical point: your scaler and your PCA must be **fitted only on the training photos**, then applied to validation/test. Look up `sklearn.pipeline.Pipeline` or `make_pipeline` — chaining your steps this way does the fitting correctly for you automatically. Build a small class with `.fit(X, y)` and `.predict_proba(X)`, matching the project's shared "contract" (see Hightrex's guide): after fitting, it should have a `classes_` list (use `np.unique(y)`, sorted, or whatever your pipeline gives you automatically).

**Step 5 — Experiment, one change at a time, on the *validation* photos only** (never look at test photos yet). Keep a table in your notes:

| What I changed | Validation accuracy | What I think happened |
|---|---|---|
| HOG, size 64, C=10 (starting point) | | |
| size 96 | | |
| C=1 / C=100 | | |
| PCA, 100 components | | |

Try both feature types, a couple of image sizes, a couple of values of `C`, and (for PCA) a couple of values for the number of components kept. Log at least 8 rows.

**Step 6:** Once the group agrees to freeze settings (end of Phase 2), pick your best HOG setup and your best PCA setup, and hand the settings to Hightrex for `train_all.py`.

**Done when:** you have a working classical model, an experiment log with 8+ rows, and final settings chosen.

---

## Phase 3 — Evaluate and Demo (Week 4)

**Task 1:** Let Hightrex's final run use your finished, frozen model. Don't change it once results are being generated.

**Task 2:** When the final results come back, write about half a page: which of your two feature types (HOG vs. PCA) did better, on clean photos and on the blurred/noisy ones? Why do you think that is? (Think about what blur physically does to edges, which is what HOG relies on.)

**Task 3:** Prepare a short "walk-through" of your training code for the presentation — pick about 10 lines (your feature extraction and your classifier training) and practise explaining what each line does and why, in under a minute.

**Task 4:** Start the reference list. Go through your Assessment 1 literature review and pull out every reference you already checked. Add them to `report/references.csv` with columns like: full reference, link or DOI, checked (yes/no), which report section it belongs to. You need **15 minimum**, aim for 20 so there's a buffer, and only mark something "checked" once you've opened the actual source.

**Done when:** you understand and can explain your final results, your code walk-through is ready, and the reference collection has started.

---

## Phase 4 — Write and Present (Weeks 5–6)

**Your report sections:**
- Your part of the Methodology: preprocessing → HOG/PCA features → SVM, with your final settings in a small table.
- Your part of the Introduction: one paragraph on why a classical approach is worth including as a comparison point.
- Your results paragraph in the Evaluation section, using what you wrote in Phase 3.
- **References:** collect everyone's checked references, get to 15+ in one consistent style (APA 7th or IEEE — whichever the group picked), make sure every reference cited in the text also appears in the list and vice versa, and cite the dataset and the libraries you used too.
- Check the whole report's page count once everything is combined (max 8 pages).

**Your slides:** how HOG and PCA/eigenfaces work (one simple picture each is enough), your final settings, your code walk-through, and your results.

**Your part of the demo/Q&A:** be ready to explain, live, what HOG and PCA capture and why you chose your final SVM setting, and to answer questions about your model from any teammate.

**Done when:** your report parts are written, references are complete and checked, your slides are ready, and you've rehearsed your explanation twice.
