# Git Basics — read this before Phase 1

You will use only **six commands** for this whole project. Nobody expects you to know Git already — this page is everything you need.

**Words explained**
- **Terminal:** a window where you type commands instead of clicking. VS Code: menu **Terminal → New Terminal**.
- **Repository (repo):** the shared project folder that Git keeps track of.
- **Clone:** download your own copy of the repo, once.
- **Commit:** save a snapshot of your changes with a short note.
- **Push / Pull:** send your commits to GitHub / fetch everyone else's commits.

## Install Git (once per computer)

Download from **git-scm.com** and install with the default options. Then tell it who you are:

```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Get the project onto your computer (once)

Hightrex will send you a link to the repo and add you as a collaborator (check your email for a GitHub invite and accept it). Then:

```
git clone <the link Hightrex sent>
cd csi6209-face-attendance
```

This creates a folder with the whole project. You only do this once.

## Your daily routine (every time you work on the project)

**Before you start working — always pull first:**
```
git pull
```
This brings in whatever your teammates added since you last looked.

**After you finish working — save and share your changes:**
```
git add <the files you changed>
git commit -m "a short note about what you did"
git push
```

Example: `git add src/classical.py` then `git commit -m "Add HOG feature extraction"` then `git push`.

**That's it.** Pull → work → add → commit → push. Repeat.

## The one rule that avoids almost every problem

**Only edit your own files.** If two people edit the same file at the same time, Git has to merge their changes and it can get confusing. Stick to your own area of the project (each person's guide says which files are theirs) and this will not come up.

## If something goes wrong

- **`git push` is rejected / says "rejected" or "non-fast-forward":** someone pushed before you. Run `git pull`, then `git push` again.
- **A file shows lines with `<<<<<<<`, `=======`, `>>>>>>>`:** this is a merge conflict — two people changed the same lines. Don't panic: open the file, decide which version (or combination) is correct, delete the `<<<<<<<` / `=======` / `>>>>>>>` markers, save, then `git add`, `git commit`, `git push` as normal. Ask a teammate if you're unsure.
- **`git commit` says "please tell me who you are":** run the two `git config` lines above.
- **You're stuck for more than 15 minutes:** post the exact error message in the group chat. Nobody is expected to figure out Git alone.

## Checking what's going on

```
git status
```
Shows which files you've changed and whether you're up to date. Run it any time you're unsure what to do next — it usually tells you.
