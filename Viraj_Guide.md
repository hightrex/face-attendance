# Viraj's Guide — CSI6209 Assessment 2

**Your role:** the deep learning model (pretrained face network + a simple classifier), and making sure everything runs on the lab computers.
**Read first:** `Git_Basics.md`. **Also read:** `CSI6209_Lean_Plan.md` for the big picture.

**Your file (nobody else should edit this):** `src/deep.py`. **Also yours:** `requirements.txt` and the "Setup" part of the README.

Nobody is giving you finished code — this guide tells you *what* to build and which library functions to look up, so you write it and understand every line yourself.

---

## Phase 1 — Set Up (Week 1)

**Task 1:** Follow `Git_Basics.md` to install Git, clone the repo, and make your first small commit so you know the routine works.

**Task 2: Check the ECU lab PCs early — this is urgent, do it by Day 3.** On a lab computer: check the Python version, check that `pip install` works, and try `pip install torch torchvision` inside a fresh virtual environment. Time it and note whether it succeeds. Report your findings to the group immediately — if PyTorch cannot be installed on the lab machines, the group needs a backup plan (train and demo from a laptop instead) well before it becomes urgent.

**Task 3:** Write `requirements.txt` — the list of Python packages the project needs (one per line: `numpy`, `scikit-learn`, `torch`, `torchvision`, `facenet-pytorch`, etc.). Test that `pip install -r requirements.txt` works from scratch in a brand-new virtual environment.

**Task 4: Reading, in your own words, in a notes file**
- What is a neural network, in plain language?
- What is a CNN (convolutional neural network)?
- What is an "embedding"? (The idea: a network turns a photo into a list of numbers, positioned so that photos of the *same* person land close together and different people land far apart.)
- What is "transfer learning", and why use a network that's already been trained on millions of faces instead of training from nothing?
- Look up the `facenet-pytorch` project page and find out what dataset its pretrained weights were trained on. Could any of those people also appear in our dataset (LFW)? Note this down — it's a real limitation to mention in the report.

**Done when:** the lab PC check is reported, `requirements.txt` installs cleanly on a fresh environment, and your notes answer every question above.

---

## Phase 2 — Build (Weeks 2–3)

Build `src/deep.py`. It needs to do two things:

**Step 1 — Turn a photo into an embedding.** Look up `facenet_pytorch.InceptionResnetV1` and its `pretrained="vggface2"` option — this loads a network that already knows faces. Key points to research and get right:
- Put the network in evaluation mode (look up `.eval()` on a PyTorch model) and don't compute gradients (`torch.no_grad()`) — you are only *using* it, not training it.
- The network expects photos resized to 160×160 and scaled in a particular way — check the facenet-pytorch documentation or examples for the exact scaling it expects.
- Running many photos through the network at once ("batching") is much faster than one at a time — look up how to process a list of images in batches.

**Step 2 — Train a small classifier on the embeddings.** Look up `sklearn.linear_model.LogisticRegression`. You compute embeddings once for your training photos, then fit a `LogisticRegression` on those embeddings and the names. Its setting `C` controls overfitting, same idea as in the classical model.

**Step 3 — Match the shared "contract".** Build a class with `.fit(X, y)` and `.predict_proba(X)`, and a `classes_` list, exactly like Maureen's model, so Hightrex's evaluation code works on yours without changes.

**Step 4 — Experiment, on the *validation* photos only,** and log each attempt:

| What I changed | Validation accuracy | What I think happened |
|---|---|---|
| Logistic regression, C=1 (starting point) | | |
| C=0.1 / C=10 / C=100 | | |

Try a few values of `C`. Since computing embeddings is the slow part, do it once and re-use it while you try different `C` values.

**Step 5 (optional, only if you finish early):** try fine-tuning a small pretrained image network (for example ResNet-18 from `torchvision.models`) instead of only training a classifier on frozen embeddings. This is harder and involves training with an optimiser and multiple passes ("epochs") over the data — only attempt it once your basic version works well, and ask for a focused explanation of transfer-learning fine-tuning when you get there.

**Step 6:** Once the group freezes settings, pick your best `C` and hand it to Hightrex for `train_all.py`.

**Done when:** your model works, downloads its weights successfully, your experiment log has entries, and final settings are chosen.

---

## Phase 3 — Evaluate and Demo (Week 4)

**Task 1:** Let Hightrex's final run use your finished, frozen model.

**Task 2:** When results come back, write about half a page: how did your model compare to the classical ones, on clean photos and on blurred/noisy ones? Deep embeddings are usually more robust to small image changes than hand-made features — did you see that, and why might that be true (or not)?

**Task 3:** Prepare a short "walk-through" of your training code for the presentation — pick about 10 lines (getting the embedding, training the classifier) and practise explaining them in under a minute.

**Task 4: The clean-machine test.** This is your responsibility. On a lab PC (or any computer that has never run this project), clone the repo fresh into an empty folder, and follow **only the README** step by step — nothing from memory or your own machine. If anything is missing, wrong, or fails, fix the README (or tell Hightrex) until a brand-new computer can run the whole project from nothing. This proves the project you submit will actually work for the markers.

**Done when:** your results write-up is done, your code walk-through is ready, and the clean-machine test passes.

---

## Phase 4 — Write and Present (Weeks 5–6)

**Your report sections:**
- Your part of the Methodology: the embedding model, the classifier on top, and your final settings.
- Your part of the Introduction: one paragraph on why a pretrained deep network is worth trying.
- Draft the Conclusion section (key outcomes, what the group learned, honest limitations — including the pretrained-weights overlap point from Phase 1 — and realistic future work). The rest of the team will edit it with you.

**Your slides:** what an embedding is (one simple picture is enough), why transfer learning, your final settings, your code walk-through, your results.

**Your part of the demo/Q&A:** be ready to explain, live, what an embedding is and why you didn't train the whole network from nothing, and to answer questions about your model from any teammate.

**Submission support:** you already tested the clean-machine setup in Phase 3 — do it again right before submission on the final, frozen code, to make sure nothing broke since then.

**Done when:** your report parts are written, the conclusion is drafted, your slides are ready, and you've rehearsed your explanation twice.
