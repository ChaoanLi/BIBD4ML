# BIBD-Based Diagnostic and Pairwise Analysis for Machine Learning Models

This module provides a diagnostic tool for visualizing the effect of **Box–Cox transformations** on **pairwise model comparisons** under a **Balanced Incomplete Block Design (BIBD)** framework. It also includes **residual analysis before and after transformation** to guide transformation decisions.

## Features

- 📈 Plot pairwise **t-statistics vs. Box–Cox lambda**.
- 🔍 Compute and display **Tukey’s significance threshold**.
- 📉 Show **residuals vs. fitted values**:
  - Before transformation.
  - After transformation using the **optimal Box–Cox lambda (MLE)**.

## Function

### `plot_pairwise_t_vs_lambda_with_residuals(...)`

**Description:**  
Generates diagnostic plots for BIBD-based analysis of machine learning model performance, with optional Box–Cox transformation of response variables.

**Parameters:**

| Parameter         | Type          | Description                                                                 |
|------------------|---------------|-----------------------------------------------------------------------------|
| `response_matrix`| `np.ndarray`  | Response matrix (treatments × blocks), use `np.nan` for missing entries    |
| `treatment_names`| `list[str]`   | Names of the treatments (default: `["M1", ..., "Mt"]`)                     |
| `k`              | `int`         | Number of treatments per block in BIBD (default: `3`)                      |
| `lambda_bibd`    | `float`       | Lambda value for BIBD ANOVA (default: `3`)                                 |
| `alpha`          | `float`       | Significance level for Tukey threshold (default: `0.05`)                   |

**Example Usage:**

```python
plot_pairwise_t_vs_lambda_with_residuals(
    response_matrix=response_matrix,
    treatment_names=['SVM', 'LR', 'LDA', 'DT', 'kNN'],
    k=3,
    lambda_bibd=3,
    alpha=0.05
)
