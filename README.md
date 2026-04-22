# Customer Propensity & Segmentation Analytics

> **Live Demo →** [customerprofiling-5khq3rytojlz5y5fczxpv2.streamlit.app](https://customerprofiling-5khq3rytojlz5y5fczxpv2.streamlit.app/)

An end-to-end machine-learning project that answers two core marketing questions:

1. **Who will respond to our next campaign?** — Propensity modelling (binary classification).
2. **What kind of customers do we have?** — Customer segmentation (unsupervised clustering).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
  - [1. Data Cleaning](#1-data-cleaning)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Preprocessing](#3-preprocessing)
  - [4. Propensity Modelling (Classification)](#4-propensity-modelling-classification)
  - [5. Customer Segmentation (Clustering)](#5-customer-segmentation-clustering)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)

---

## Project Overview

This project uses a real-world retail dataset to build models that help marketing teams:

- **Predict campaign response** — identify customers most likely to accept a marketing offer, enabling targeted outreach and higher ROI.
- **Segment the customer base** — group customers by behaviour and demographics so that messaging can be personalised per segment.

A Streamlit web application exposes both capabilities interactively.

---

## Dataset

**File:** `Customer_Profiling.csv`

| Category | Columns |
|---|---|
| Demographics | `ID`, `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Purchase history | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Purchase channels | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Campaign history | `AcceptedCmp1` – `AcceptedCmp5`, `Response` (target) |
| Other | `Dt_Customer`, `Recency`, `Complain`, `Z_CostContact`, `Z_Revenue` |

- **Rows:** ~2 240 customers
- **Target variable:** `Response` (1 = accepted the last campaign offer, 0 = did not)
- **Class imbalance:** the positive class (`Response = 1`) is a minority (~15 %)

---

## Project Pipeline

### 1. Data Cleaning

- Identified and imputed **24 missing values** in `Income` with the median.
- Removed extreme income outliers (top 0.5 % by value).
- Verified zero duplicate rows.

### 2. Feature Engineering

New columns derived from raw data:

| Feature | Description |
|---|---|
| `Age` | `current_year − Year_Birth` |
| `Customer_Tenure_Days / Months` | Days/months since the customer enrolled (`Dt_Customer`) |
| `Total_Expenditure` | Sum of all product category spend columns |
| `Average_Monthly_Spend` | `Total_Expenditure / Customer_Tenure_Months` |
| `Dependents` | `Kidhome + Teenhome` |
| `Dependency_Ratio` | `Dependents / total_customers` |
| `Engagement_Score` | Weighted mix of web visits (×0.4) and store purchases (×0.6) |
| `Campaign_Response` | 1 if the customer accepted *any* of the five prior campaigns |
| `Has Kids` / `Has Teens` | Binary flags |

Categorical encoding:
- `Education` and `Marital_Status` → one-hot encoded (rare values such as `'Absurd'`, `'Alone'`, `'YOLO'` merged into `'Single'`; `'Married'` and `'Together'` merged into `'Committed'`).

### 3. Preprocessing

- **Min-Max scaling** applied to `Income`, `Total_Expenditure`, and `Customer_Tenure_Months`.
- Redundant and identifier columns (`ID`, `Year_Birth`, `Dt_Customer`, `Z_CostContact`, `Z_Revenue`, individual campaign flags, etc.) dropped before modelling.
- **80 / 20 train-test split** with `random_state=42`.
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied to the training set to address class imbalance; models are evaluated both with and without SMOTE.

### 4. Propensity Modelling (Classification)

Three classifiers were trained and evaluated using **3-fold Stratified K-Fold cross-validation**:

| Model | Variant |
|---|---|
| Support Vector Machine (SVM) | Without SMOTE / With SMOTE |
| Decision Tree | Without SMOTE / With SMOTE |
| Random Forest | Without SMOTE / With SMOTE |

Evaluation metrics per model: **Accuracy, Precision, Recall, F1-Score** (all reported for the positive class).

Feature importances were extracted for the tree-based models (top 15 features visualised for each variant).

### 5. Customer Segmentation (Clustering)

- **Features used:** Income, Age, Recency, Total Expenditure, Average Monthly Spend, Engagement Score, Dependents, Campaign Response, and encoded education/marital-status columns.
- Additional **Min-Max scaling** applied to unscaled clustering features.
- **Elbow Method** (inertia / WCSS) and **Silhouette Score** used to determine the optimal number of clusters → **k = 3** selected.
- **K-Means (k-means++ initialisation, n_init=10)** used for the final clustering.
- **PCA** used to reduce dimensionality to 2 components (and separately to PC1 vs PC3) for scatter-plot visualisation.
- Animated GIFs generated to illustrate how clusters evolve from k=1 → k=2 → k=3.
- Cluster summary (mean feature values per cluster) computed to interpret each segment.

---

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| SVM (No SMOTE) | ~87 % | High accuracy, low recall on minority class |
| SVM (With SMOTE) | ~83 % | Improved recall on positive class |
| Decision Tree (No SMOTE) | ~88 % | Fast; prone to overfitting |
| Decision Tree (With SMOTE) | ~84 % | Better balance |
| Random Forest (No SMOTE) | ~90 % | Best raw accuracy |
| Random Forest (With SMOTE) | ~87 % | Best balance of precision & recall |

> Exact figures will vary with the current date because `Customer_Tenure_Months` and `Age` are computed dynamically.

**Segmentation:** Three distinct customer segments identified, roughly characterised by:
- **High-value, high-spend** customers (wine & meat buyers, high income, low recency).
- **Mid-value, deal-seekers** (moderate spend, active web visitors, families with kids).
- **Low-engagement, low-spend** customers (young, lower income, rarely purchase).

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `imageio` |
| Machine learning | `scikit-learn` (SVM, Decision Tree, Random Forest, KMeans, PCA, MinMaxScaler, SMOTE via `imbalanced-learn`) |
| Web app | `streamlit` |
| Notebook environment | Google Colab |

---

## Repository Structure

```
Customer-Propensity-Segmentation-Analytics/
├── Customer_Profiling.csv                  # Raw dataset
├── Customer_Profiling_Colab file.ipynb     # Full analysis notebook (EDA → modelling → clustering)
├── Customer Propensity modelling.pdf       # Project report / presentation
└── README.md
```

---

## How to Run

### Notebook (Google Colab)

1. Upload `Customer_Profiling.csv` to `/content/` in your Colab session.
2. Open `Customer_Profiling_Colab file.ipynb` in Google Colab.
3. Run all cells in order (`Runtime → Run all`).

### Streamlit App (locally)

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn streamlit imageio

# Launch the app
streamlit run app.py
```

Or visit the hosted version directly:  
🔗 **<https://customerprofiling-5khq3rytojlz5y5fczxpv2.streamlit.app/>**

