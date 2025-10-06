Project 1 — Regression & Resampling Methods
Course: FYS-STK3155 / FYS4155 — Data Analysis & ML
University of Oslo, Fall 2025

Group Members:
- Bartosz Trela (bartoszt@uio.no)
- Samson Fekade Badishe (samsonfb@uio.no)
- Theodor Charles Merritt (Wålberg tcwaalbe@uio.no)

Overview:
This project implements and evaluates OLS, Ridge, and LASSO regression methods on the Runge function,
including experiments with polynomial degree, noise, and resampling (bootstrap vs cross-validation).
It also compares optimization methods (gradient descent, Adam, RMSProp, etc.) in the context of regression.

applied-data-analysis-and-ml/
├── Project1/
│   ├── chat-logs/                  # AI chat logs used during the project
│   ├── code/                        # Jupyter notebooks with main project code
│   │   ├── exercisesweek36.ipynb    # Previous exercises (not relevant for main project)
│   │   ├── exercisesweek37.ipynb    # Previous exercises (not relevant for main project)
│   │   └── project1.ipynb           # Main notebook with full project implementation
│   ├── ipynb/                       # Instruction files for the project (for reference)
│   ├── pdf/                         # Final report in PDF and LaTeX formats
│   │   ├── Project1.pdf
│   │   └── Project1.tex
│   └── report/                      # Previous exercises and draft reports (not relevant)
├── exercises/                        # Additional exercises from the course
├── lecture notes/                     # Lecture notes provided for the course
├── .gitignore                         # Git ignore rules
├── README.txt                         # This file
└── requirements.txt                   # Python package requirements

How to Run:
1. Clone the repo:
   git clone <repo-url>
   cd applied-data-analysis-and-ml/Project1

2. Install dependencies:
   pip install -r ../requirements.txt

3. Launch the notebook:
   jupyter notebook Project1.ipynb

4. Run all cells to reproduce experiments, plots, and metrics.

Notes:
- Figures appear inline in the notebook and are exported manually if desired.
- The report (PDF) is in the pdf/ folder.
- Source code is organized by part (a–h) and contains comments and docstrings.
- There is no external dataset; Runge’s function is generated within the notebook.

Report:
The project report is located at:
   Project1/pdf/Project1.pdf
It contains the full discussion, figures, results, and conclusions.