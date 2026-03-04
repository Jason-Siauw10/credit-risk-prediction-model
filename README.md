# 💳 Credit Risk Prediction Model

A machine learning project built to predict credit card default risk using Random Forest classification, achieving **81.7% accuracy** on real-world banking data.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Dashboard](#dashboard)

---

## Project Overview

Credit score models are a critical tool for financial institutions to assess the risk of lending. Older logistic regression-based models were losing predictive accuracy under large economic fluctuations, prompting the need for a more robust machine learning approach.

This project identifies the **key features that determine credit default risk** among credit card applicants and builds a predictive model with target accuracy ≥85%, while also profiling the demographic and income characteristics of good credit clients.

---

## Problem Statement

> *What are the key features when deciding on a good or bad credit card client? What demographics and income profiles tend to belong to reliable clients?*

**Hypotheses going in:**
- Higher yearly income → lower default risk
- Stable employment → lower default risk
- Older clients → lower default risk

---

## Dataset

Two datasets were merged for this project:

| Dataset | Description |
|---|---|
| **Account Records** | Demographics, income, employment, and personal attributes of bank clients |
| **Credit Records** | Historical monthly credit status of credit card users |

- Only records with matching IDs across both datasets were used
- Final merged dataset after cleaning: ~6,000 records

**Features used:**

| Feature | Type |
|---|---|
| `AMT_INCOME_TOTAL` | Yearly income amount |
| `Age` | Derived from `DAYS_BIRTH` |
| `work_yr` | Derived from `DAYS_EMPLOYED` |
| `NAME_INCOME_TYPE` | Employment category |
| `FLAG_OWN_REALTY` | Owns property (Y/N) |
| `FLAG_OWN_CAR` | Owns a car (Y/N) |
| `NAME_EDUCATION_TYPE` | Highest education level |
| `CODE_GENDER` | Gender |
| `NAME_FAMILY_STATUS` | Marital status |
| `CNT_CHILDREN` | Number of children |

---

## Methodology

```
Define Problem → Data Cleaning → EDA → ML Modeling → Insights → Recommendations
```

### 1. Data Cleaning (Python)
- Removed duplicate IDs from account records
- Dropped `OCCUPATION_TYPE` column (134,000+ missing values)
- Fixed outliers using IQR (1st–99th percentile) for income and children count
- Converted `DAYS_BIRTH` and `DAYS_EMPLOYED` into `Age` and `work_yr` in years
- Label encoded all categorical columns

### 2. Credit Label Engineering
- Credit statuses `C`, `X`, `0` → labeled **0 (Good Credit)**
- Credit statuses `1`–`5` (late/overdue) → labeled **1 (Bad Credit)**
- Grouped by client ID using `max()` — if any month was bad, client flagged as bad credit

### 3. Class Imbalance Handling
- Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance good vs. bad credit classes before training

### 4. Model Training & Evaluation
- 70/30 train-test split
- StandardScaler applied to features
- Five models benchmarked

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 53.1% | 53.1% | 53.1% | 53.1% |
| K-Neighbors Classifier | 70.3% | 72.4% | 70.3% | 69.6% |
| Decision Tree Classifier | 81.7% | 83.1% | 81.7% | 81.5% |
| **Random Forest Classifier** | **81.7%** | **83.4%** | **81.7%** | **81.5%** |
| XGBoost Classifier | 88.2% | 88.7% | 88.2% | 88.1% |

**Why Random Forest was chosen over XGBoost:**
Although XGBoost achieved higher accuracy (~88%), its ensemble boosting mechanism makes it harder to interpret. Random Forest was selected for production use because it provides **transparent feature importances** — crucial for a banking context where explainability and regulatory transparency matter.

---

## Key Findings

### 🔑 Top 3 Key Features (Random Forest Feature Importance)
1. **Work Year** — Longer employment history = lower default risk
2. **Age** — Older clients tend to have greater financial maturity
3. **Yearly Income** — Higher income = greater repayment capacity

### 📊 Supporting Features
- Income Type (working vs. pensioner vs. state servant)
- Owns Realty / Car (signals financial capacity & collateral)
- Education Level

### 👥 Client Demographics (EDA)
- **Age**: Majority of clients fall in the 29–40 age group, followed by 41–58
- **Income**: Most clients earn between 100K–300K (middle-class bracket)
- **Work Experience**: Majority are in the 0–5 year and 5–10 year brackets (early-to-mid career)

---

## Recommendations

**1. Focus on Key Features**
Prioritize income amount, age, and years of employment when evaluating applications. These three factors collectively signal payment capability, financial stability, and career stability.

**2. Check Supporting Features**
Use income type, property/car ownership, and education level as secondary screening criteria. Property ownership in particular signals both financial capacity and potential collateral.

**3. Cater to the Actual Client Base**
Since the majority of applicants are middle-income, early-career adults (ages 29–40), the bank should offer:
- **Mid-tier credit cards** for established but not yet high-income clients
- **Credit builder cards** for those with little to no credit history

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189fc4?style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-76b7b2?style=flat)
![Tableau](https://img.shields.io/badge/Tableau-E97627?style=flat&logo=tableau&logoColor=white)

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn (Logistic Regression, KNN, Decision Tree, Random Forest), XGBoost
- **Imbalanced Data**: imbalanced-learn (SMOTE)
- **Visualization**: Matplotlib, Seaborn, Tableau
- **Environment**: Google Colab

---

## Project Structure

```
credit-risk-prediction-model/
│
├── README.md
├── DEEPP_Data_Cleaning_and_Analysis.ipynb   # Full pipeline: cleaning → EDA → ML
│
├── data/
│   └── (datasets linked via Google Drive — see notebook)
│
├── presentation/
│   └── DEEPP_Project_Customer_Credit_Profile_Prediction_Model.pdf
│
└── dashboard/
    └── (Tableau dashboard — see link below)
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jason-Siauw10/credit-risk-prediction-model.git
   cd credit-risk-prediction-model
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn gdown
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook DEEPP_Data_Cleaning_and_Analysis.ipynb
   ```
   Or open directly in [Google Colab](https://colab.research.google.com/drive/1rFWKqX39QxeJpDtLt2ZEihu8mY1rczj0?usp=sharing).

> ⚠️ The datasets are hosted on Google Drive and downloaded automatically within the notebook via `gdown`.

---

## Dashboard

An interactive Tableau dashboard was created to explore client demographics (age group, income bracket, work year, credit status, education, and asset ownership).

🔗 **[View Dashboard on Tableau Public](https://public.tableau.com/app/profile/jason.suciptono/viz/DEEPPProjectDashboard/AODABankCreditCardClientDemographicDashboard)**

---

## Acknowledgements

This project was completed as part of the **DEEPP Data Analytics Program**.

Data source: Internal bank records (Account Records + Credit Records datasets).
