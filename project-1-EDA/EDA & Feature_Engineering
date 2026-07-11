# Project 1: Advanced EDA & Feature Engineering
### Decode Labs — Data Science Internship 2026

---

## Overview
This project focuses on transforming a raw, messy real estate dataset into a clean, ML-ready dataset using advanced data preprocessing techniques including missing value imputation, outlier detection, and feature engineering.

---

## Dataset
| Detail | Info |
|--------|------|
| **Name** | House Price India |
| **Source** | Kaggle |
| **Original Shape** | 187,531 rows × 21 columns |
| **Final Shape** | 175,813 rows × 24 columns |

---

## What I Did

### 1. Missing Value Handling
Applied a Decision Matrix approach based on missingness percentage:

| Rule | Columns | Method |
|------|---------|--------|
| 100% missing | Dimensions, Plot Area | Dropped columns |
| < 5% missing | Description, Status, Floor, Bathroom, Transaction, Furnishing | Dropped rows (dropna) |
| ~9% missing | Price (in rupees) | Median imputation |
| > 20% missing | Carpet Area, facing, overlooking, Society, Balcony, Car Parking, Ownership, Super Area | Filled with "Unknown" |

### 2. KNN Imputation
- Applied **KNNImputer (n_neighbors=5)** on Carpet Area numeric values
- Demonstrated on 10,000 row sample (KNN is O(N²) — too slow on 187k rows)
- Missing values reduced from **4,474 → 0**

### 3. Outlier Detection & Neutralization
Used **IQR (Interquartile Range)** method on Price column:
```
Q1: 4,500 | Q3: 8,410 | IQR: 3,910
Lower Bound: -1,365 | Upper Bound: 14,275
Outliers Detected: 9,881 rows
```
Applied **Winsorization** (`numpy.clip`) — capped values instead of deleting rows to preserve data volume.

### 4. Feature Engineering (3 New Features)
| Feature | Logic | Purpose |
|---------|-------|---------|
| `Price_Category` | Low / Medium / High based on quantiles | Categorical price segmentation |
| `Has_Parking` | 1 if Car Parking available, else 0 | Binary signal for ML models |
| `Is_Furnished` | 1 if Furnished, else 0 | Binary signal for ML models |

### 5. One-Hot Encoding
- Applied `pd.get_dummies()` on `Furnishing` and `Transaction` columns
- Used `drop_first=True` to avoid multicollinearity (dummy variable trap)
- Shape after encoding: **175,813 × 24**

---

## Results
```
Original Dataset:  187,531 rows × 21 columns
After Cleaning:    175,813 rows × 24 columns
Missing Values:    0 remaining in all key columns
New Features:      3 engineered + encoding columns added
```

---

## Tools & Libraries
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24-orange)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-1.3-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-purple)

---

## Project Structure
```
📁 data-science-internship-decodelabs/
│
├── 📓 Project1_EDA_FeatureEngineering.ipynb   # Main notebook
├── 📊 house_prices.csv                         # Raw dataset
├── 📊 house_prices_cleaned.csv                 # Cleaned output
├── 🖼️ graph1_missing.png                       # Missing values chart
├── 🖼️ graph2_price.png                         # Price distribution
├── 🖼️ graph3_category.png                      # Price categories
└── 📄 README.md
```

---

## Key Learnings
- Real-world datasets are never clean — missingness itself can be a meaningful signal
- KNN Imputation is powerful but computationally expensive on large datasets (O(N²))
- Winsorization preserves data volume better than row deletion for outliers
- One-Hot Encoding must be used over Label Encoding for nominal categories to avoid false ordinal relationships
