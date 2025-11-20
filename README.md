# learning-ml-models

This repository follows the Microsoft Learn path "Create machine learning models" and contains starter notebooks and READMEs for each module.

Modules

- learning_modules/01-explore-analyze-data - Explore and analyze data with Python
  - learning_modules/01-explore-analyze-data/README.md
- learning_modules/02-train-evaluate-regression - Train and evaluate regression models
  - learning_modules/02-train-evaluate-regression/README.md
- learning_modules/03-train-evaluate-classification - Train and evaluate classification models
  - learning_modules/03-train-evaluate-classification/README.md
- learning_modules/04-train-evaluate-clustering - Train and evaluate clustering models
  - learning_modules/04-train-evaluate-clustering/README.md
- learning_modules/05-train-evaluate-deep-learning - Train and evaluate deep learning models
  - learning_modules/05-train-evaluate-deep-learning/README.md

Quick start (Git Bash on Windows)

```bash
# create and activate venv
python -m venv .venv
source .venv/Scripts/activate

# install project and recommended extras (if defined in pyproject.toml)
pip install -e .
# optional: install dev or notebook extras
pip install -e "[dev]"
pip install -e "[notebook]"

# start Jupyter Lab
jupyter lab --port 8888 --no-browser
```

Notes
- This project uses Git Bash for example commands; see `.github/copilot.yaml` for project-local suggestions.
- Jupyter notebooks for each module live under `learning_modules/*`.
