# BIBD4ML: Balanced Incomplete Block Design for Machine Learning Model Comparison

This repository contains a Jupyter Notebook (`BIBD4ML.ipynb`) showcasing how to apply **Balanced Incomplete Block Design (BIBD)** in evaluating and comparing classical machine learning models across a variety of public datasets.

## 📊 Overview

The notebook implements model comparisons on ten well-known datasets using BIBD principles, with classical machine learning models such as:

- **Logistic Regression**
- **Linear Discriminant Analysis (LDA)**
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (k-NN)**
- **Decision Tree**

Each dataset is analyzed by selecting a subset of models (incomplete block), and comparisons are made using pairwise t-tests and visual diagnostics (e.g., residual plots and Box-Cox transformation effects).

## 📁 Datasets Used

The following datasets are included in the analysis:

1. **Iris** - LDA, SVM, Logistic Regression  
2. **Wine** - Decision Tree, LDA, k-NN  
3. **Wine Quality** - SVM, Logistic Regression, k-NN  
4. **Adult Census Income** - SVM, LDA, k-NN  
5. **Bank Marketing** - SVM, LDA, Decision Tree  
6. **Breast Cancer Wisconsin** - Decision Tree, k-NN, LDA  
7. **Mushroom** - SVM, Logistic Regression, k-NN  
8. **Car Evaluation** - Logistic Regression, LDA, k-NN  
9. **Heart Disease** - Logistic Regression, LDA, Decision Tree  
10. **Spambase** - SVM, k-NN, Decision Tree

## 📈 Analysis Techniques

- **Box-Cox Transformation** to stabilize variance.
- **Pairwise t-statistics** to compare model performance.
- **Tukey's HSD** to determine significant differences.
- **Residual Diagnostics** for model fit validation.

## 🧪 Statistical Methodology

This project leverages the structure of a BIBD (with parameters \(v, b, r, k, \lambda\)) to reduce the number of model comparisons while maintaining balanced inference power. It’s particularly helpful in experimental settings where exhaustive pairwise model evaluation is computationally expensive.

## 📦 Requirements

The full notebook was executed with Python 3.12 and the versions in
`requirements.txt` on 2026-10-20 (141 code cells, no cell execution errors).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook BIBD4ML.ipynb
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
Run the cells in order; UCI data retrieval requires network access. The repository
also includes the project report, result table, and diagnostic plots.

An unrestricted installation using pandas 3 failed in the Car Evaluation
classification section with `Unknown label type: unknown`. The tested environment
uses pandas 2.3.3. The pinned direct dependencies record this working environment;
they are not a lock of every transitive dependency.

## Reproducibility limits

- The final analyses use numeric matrices recorded in cells 138–140, rather than
  automatically collecting the model-training results above them. Re-running the
  notebook does not automatically replace those historical measurements.
- Several cross-validation calls standardize the full feature matrix before
  splitting folds. Preprocessing needs to be fitted within each fold before
  those scores can be interpreted as validation without preprocessing leakage.
- The shuffled cross-validation splitter has no fixed random seed. Measured
  training/prediction times depend on the machine and software environment.
- Successful execution verifies that the notebook runs in the tested environment;
  it is not an independent validation of the statistical conclusions.
