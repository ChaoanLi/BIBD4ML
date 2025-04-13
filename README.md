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

To run the notebook, install the following Python packages:

```bash
pip install ucimlrepo scikit-learn scipy numpy pandas matplotlib seaborn
