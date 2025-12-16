# Project 3 — Image Classification (PCA+KMeans + CNN)

This repository contains the materials for Project 3 (FYS-STK3155/FYS4155). It includes a small image dataset of cells (Live/Dead) and utilities to run two approaches:

- PCA + KMeans baselines (raw pixels or engineered features)
- A minimal CNN implemented in PyTorch

Core helpers live in [Project3/code/p3_module.py](Project3/code/p3_module.py). A reproducible notebook is in [Project3/code/project3.ipynb](Project3/code/project3.ipynb).


## Repository Layout

- code/: notebooks and reusable helpers
	- [Project3/code/p3_module.py](Project3/code/p3_module.py)
	- [Project3/code/project3.ipynb](Project3/code/project3.ipynb)
- data/: cells dataset with splits
	- [Project3/data/cells](Project3/data/cells) with `train/`, `dev/`, `test/` each containing `Live_resized/` and `Dead_resized/`
- report/: report assets (figures, diagrams)
- [Project3/environment.yml](Project3/environment.yml), [Project3/requirements.txt](Project3/requirements.txt)


## Environment Setup

Choose one of the two options below.

### Option A: Conda (recommended)

From the project root (Project3/):

```bash
conda env create -f environment.yml
conda activate fys-stk-4155-project3
```

### Option B: Python venv + pip

Windows PowerShell (from Project3/):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional (register a Jupyter kernel):

```powershell
python -m ipykernel install --user --name project3
```


## Data

Download the dataset and unzip it to a folder named data/ in the Project3 directory. Here is the link: https://drive.google.com/file/d/1Z8IdHF6iOaJ5Ju3LMlaQQGKq4GCqc79b/view?usp=sharing

Then the dataset is included under [Project3/data/cells](Project3/data/cells) with three splits: `train/`, `dev/`, `test/`, each with folders `Live_resized/` and `Dead_resized/`. Images are RGB 32×32. If images differ in size, the loader will resize to 32×32.


## Quickstart (Notebook)

Open and run [Project3/code/project3.ipynb](Project3/code/project3.ipynb). It demonstrates loading data, running PCA+KMeans, and training the CNN with default settings.


## Reproducibility

- Set seeds with `set_seed(…)` to make data splits and model init stable where possible.
- For PyTorch, deterministic settings are enabled when available; GPU vs CPU may still lead to small differences.
