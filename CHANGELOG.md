# Changelog

Newest entry first.

## 2026-09-30
- Push script now uses rebase with autostash, because a crashed run can leave unstaged files that block a rebase
- Launchers use two working copies, code checked out at a tag and results committed on dev, because a tag checkout is detached and cannot push
- Kaggle secret label is kaggle_dpfl-braintumor instead of GH_TOKEN
- .gitignore pattern changed from data/ to /data/, because the unanchored pattern silently ignored src/data
- README expanded with repository layout and data policy
- Scaffold pushed from Kaggle notebook 00-scaffold-repo
- Roadmap created and committed as docs/ROADMAP.md
