# Project 2 — Neural Networks  
**Course:** FYS-STK3155 / FYS4155 — Data Analysis & Machine Learning  
**University of Oslo, Fall 2025**

## Group Members
- **Bartosz Trela** — bartoszt@uio.no  
- **Samson Fekade Badishe** — samsonfb@uio.no  
- **Theodor Charles Merritt (Wålberg)** — tcwaalbe@uio.no  

---

## Overview

This project implements a fully connected feedforward neural network (FFNN) from scratch using NumPy and applies it to both regression and classification tasks. Key goals included:

- Building a general-purpose FFNN supporting arbitrary depth, width, and activation functions
- Applying the network to the Runge regression task and MNIST/Fashion-MNIST classification
- Investigating the impact of architecture, regularization, optimizer choice, and learning rate
- Comparing performance against classical models (OLS, Ridge, Lasso) and PyTorch baselines
- Reproducing results via clean modular code and a Jupyter notebook pipeline

---

## File Structure

applied-data-analysis-and-ml/
├── requirements.txt
├── Project2/
│ ├── code/
│ │ ├── p2_nn_module.py # Core FFNN implementation
│ │ └── project2.ipynb # Main notebook: data loading, training, evaluation, plots
│ ├── report/
│ │ ├── report.tex # Full LaTeX report
│ │ ├── Data Analysis and Machine Learning FYS-STK4155 - Project 2.pdf # Full pdf report
│ │ └── project2.html # Rendered report (HTML)
│ └── README.txt # This file

---

## How to run

1. Clone the repository
git clone https://github.com/Twaal/applied-data-analysis-and-ml.git
cd applied-data-analysis-and-ml/Project2

2. Install dependencies
pip install -r ../requirements.txt

3. Launch the notebook
jupyter notebook project2.ipynb

4. Run all cells to reproduce experiments, plots, and metrics

---

## Dependencies

All required packages are listed in ../requirements.txt
Includes versions for numpy, matplotlib, scikit-learn, torch, and more
Compatible with Python 3.10+
The file also sets seeds and ensures deterministic training where possible for reproducibility

---

## Module Guide (A short abstract of the code)

This project includes two main code files located in the `code/` directory:

#### `p2_nn_module.py` — Custom Feedforward Neural Network (NumPy)

This file contains a complete NumPy-based implementation of a feedforward neural network. It is fully modular and supports:

- **Network Construction**
  - Arbitrary number of layers and nodes (user-defined depth × width)
  - Weight initialization via `np.random.normal(0, 0.01)`
  - Biases initialized to zero
- **Supported Activation Functions**
  - `Sigmoid`
  - `ReLU`
  - `Leaky ReLU`
- **Loss Functions**
  - Mean Squared Error (for regression)
  - Softmax Cross-Entropy (for classification)
- **Optimizers**
  - Stochastic Gradient Descent (SGD)
  - RMSprop
  - Adam
- **Training Features**
  - Mini-batch gradient descent
  - Early stopping (optional)
  - L1 and L2 regularization
  - Deterministic training via fixed seeds

**Notable Functions and Classes:**
- `NeuralNetwork`: main class for model definition and training
- `forward()`, `backward()`, `update_params()`: core training logic
- `compute_loss()`: returns loss and accuracy/MSE
- `train()`, `predict()`, `evaluate()`: user-facing training API
- Internally structured to separate forward/backward logic per layer

---

#### `project2.ipynb` — Experiments and Analysis

This Jupyter notebook serves as the central script to:
- Load and preprocess datasets (Runge function, MNIST, Fashion-MNIST)
- Define hyperparameters and architectures
- Run training loops for:
  - Regression (Runge function)
  - Classification (MNIST, Fashion-MNIST)
- Perform systematic sweeps over:
  - Learning rate
  - Optimizers
  - Depth, width, and activation functions
  - Regularization strength
- Generate:
  - All figures used in the final report
  - Tabular summaries of test error and generalization gap
  - Confusion matrices and sample inference visualizations

**Notebook Structure:**
1. **Imports & Config** — Ensures reproducibility and layout
2. **Helper Functions** — Data scaling, visualizations, and formatting
3. **Regression** — Full suite of training, evaluation, and plotting on the Runge function
4. **Classification** — Similar pipeline for MNIST and Fashion-MNIST
5. **Appendix (Optional)** — Additional heatmaps, sweep visualizations, diagnostics

