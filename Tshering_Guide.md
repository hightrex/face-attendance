# Tshering's Guide — CSI6209 Assessment 2

**Your role:** the dataset (loading and splitting), the demo, and the slide deck.
**Read first:** `Git_Basics.md`. **Also read:** `CSI6209_Lean_Plan.md` for the big picture.

**Your file (nobody else should edit this):** `src/data.py`. **Also yours:** the demo notebook and the shared slide deck.

Nobody is giving you finished code — this guide tells you *what* to build and which library functions to look up, so you write it and understand every line yourself.

---

## Phase 1 — Set Up (Week 1)

**Task 1:** Follow `Git_Basics.md` to install Git, clone the repo, and make your first small commit so you know the routine works.

**Task 2: Lead the Team Contract.** Download the Team Contract template from Canvas. Organise a short meeting (about 45 minutes) where the group agrees on: meeting times, how you'll communicate, who does what (see `CSI6209_Lean_Plan.md`), and what happens if someone falls behind or is unwell. Everyone signs and keeps their own copy.

**Task 3: Load the dataset — this is the first big code task, and everyone depends on it.**
- Look up `sklearn.datasets.fetch_lfw_people` in the scikit-learn documentation. Read what its `min_faces_per_person`, `resize`, and `color` settings do.
- Write a function `load_data()` that: downloads the dataset with a chosen `min_faces_per_person` (a good starting point to try is around 20), then splits the photos into three groups — 60% for training, 20% for validation (used for tuning), and 20% for testing (locked away until the very end).
- Look up `sklearn.model_selection.train_test_split`, and its `stratify` setting — this makes sure every person appears in every group, in fair proportions. You will call it twice: once to split off the training group, once to split the remainder into validation and test.
- Use a **fixed random seed** (for example `random_state=42`) so that every teammate who runs your function gets the *exact same* split. This matters a lot: if splits differ between teammates, everyone's results become impossible to compare fairly.
- Print out how many people and how many photos you kept, and save a small picture showing a handful of sample faces (look up `matplotlib.pyplot.subplots` and `imshow`).

**Task 4: Agree the freeze.** Once the group has looked at your numbers and picked `min_faces_per_person` together, announce a **data freeze**: from the middle of Week 2 onward, nobody changes the dataset settings, because every change makes everyone's results different and impossible to compare.

**Done when:** `load_data()` works and gives everyone the same split, the sample-faces picture is saved, and the Team Contract is signed.

---

## Phase 2 — Build (Weeks 2–3)

**Task 1: Support the team.** If Maureen or Viraj get stuck loading data, help them first — this phase can't start for anyone until your function works reliably.

**Task 2: Dataset facts for the report.** Write a short script that prints: how many people, how many photos in total, the smallest/largest/average number of photos per person, and the sizes of your train/validation/test groups. Save this as text and as a small histogram picture (`matplotlib.pyplot.hist` on the photo-per-person counts).

**Task 3: Plan the demo.** You don't need working models yet, but plan out the story: (1) load the data and the trained models, (2) show a handful of test photos next to what each model guesses, (3) run the evaluation live and show the numbers, (4) show a blurred photo and how much accuracy drops, (5) print a mock "attendance list" from a batch of photos. Write this plan down — you'll build it in Phase 3.

**Task 4: Start the slides.** Create one shared slide deck (Google Slides or PowerPoint on OneDrive, so everyone can edit). Add a title slide and one empty section per person. Draft your own opening slides: the team, the problem (why take attendance by face), two or three sentences on related work (from your Assessment 1 literature review), and the dataset facts and sample-faces picture from Task 2.

**Task 5: Write your data notes.** In your own words: where the dataset comes from, how many people and photos, how and why it's split the way it is, what preprocessing happens, and what the dataset's limitations are (for example, these are celebrity photos, not real classroom photos).

**Done when:** the dataset facts and picture are saved, the demo plan is written down, and your first slides exist.

---

## Phase 3 — Evaluate and Demo (Week 4)

**Task 1: Build the demo**, following your Phase 2 plan, once Maureen's and Viraj's trained models exist:
- Load the saved models and a handful of test photos.
- Show the photos next to each model's guess.
- Run the evaluation live (using Hightrex's evaluation code) on a modest number of test photos, so it finishes quickly during a live presentation.
- Apply a blur to one photo and show how the models' answers or confidence change.
- Simulate an "attendance list": run the best model on a batch of test photos and print out the names it recognised.

**Task 2: Time it.** Run through the whole demo and measure how long it takes. It should comfortably fit in about 2–3 minutes, since it's part of a strict 15-minute presentation. If any step is slow, reduce how many photos it processes live.

**Task 3: Record a backup video.** Screen-record the entire demo working, in case something breaks live on presentation day (Windows: Win+G opens a recording tool; PowerPoint also has Insert → Screen Recording).

**Task 4: Save the test photos for submission.** Write a small script that saves your test group's photos and names to a single file (look up `numpy.savez_compressed`), so it can be included in the project zip as the required "test dataset".

**Done when:** the demo runs start to finish, a backup video exists, and the test photos are saved to a file.

---

## Phase 4 — Write and Present (Weeks 5–6)

**Your report sections:**
- Your part of the Methodology: the dataset and preprocessing, with your facts table and sample-faces picture.
- Your part of the Introduction: the context and the problem being solved.
- The **Abstract** — write this last, once every other section and all the results are finished. It should, in a few sentences: make the reader care, say what you set out to do, briefly say how, give the main results with numbers, and say what it means.

**Assemble the final slide deck:** collect everyone's slides into the one shared deck, make sure the look and font sizes are consistent, and that it fits comfortably in 15 minutes including the demo.

**Your part of the demo:** you drive it — sharing your screen, running each step, and narrating what's happening, while Maureen, Viraj and Hightrex jump in to comment on their own model's part.

**Rehearse:** run the whole talk and demo with a timer at least twice, once on a lab computer if possible, and make sure the backup video is easy to reach in case of a live failure.

**Done when:** your report parts and the abstract are written, the assembled deck is finished, and you've timed the full rehearsal at least twice.
