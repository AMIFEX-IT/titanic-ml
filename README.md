# 🚢 Titanic Survival Prediction — Machine Learning Project

A complete end-to-end machine learning project predicting passenger survival on the Titanic. Built as my first ML project, covering the full data science workflow from raw data to trained model.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Feature Engineering](#feature-engineering)
- [Models & Results](#models--results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [What I Learned](#what-i-learned)

---

## Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster based on features like age, sex, ticket class, and fare. This is a classic **binary classification** problem — the model outputs either `1` (survived) or `0` (did not survive).

This project follows a complete ML pipeline:
- Loading and exploring real-world data
- Cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Comparing multiple algorithms

---

## Dataset

- **Source:** [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Size:** 891 passengers, 12 columns
- **Target variable:** `Survived` (0 = died, 1 = survived)

### Raw Features

| Feature | Description | Type |
|---|---|---|
| `Survived` | Whether the passenger survived | Target (0/1) |
| `Pclass` | Ticket class (1st, 2nd, 3rd) | Categorical |
| `Sex` | Gender of passenger | Categorical |
| `Age` | Age in years | Numerical |
| `SibSp` | Number of siblings/spouses aboard | Numerical |
| `Parch` | Number of parents/children aboard | Numerical |
| `Fare` | Ticket fare paid | Numerical |
| `Embarked` | Port of embarkation (C, Q, S) | Categorical |
| `Cabin` | Cabin number | Categorical (dropped) |
| `Name` | Passenger name | Text (dropped) |
| `Ticket` | Ticket number | Text (dropped) |

### Missing Data Found
| Column | Missing Values | Action Taken |
|---|---|---|
| `Age` | 177 (19.9%) | Filled with median age |
| `Cabin` | 687 (77.1%) | Dropped — too many missing |
| `Embarked` | 2 (0.2%) | Filled with most common port |

---

## Project Workflow

```
Raw Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Engineering
   ↓
Train / Test Split (80% / 20%)
   ↓
Model Training
   ↓
Evaluation & Comparison
   ↓
Predictions on New Data
```

---

## Feature Engineering

Beyond the raw features, I created new features to give the model more signal:

| New Feature | How it was created | Why it helps |
|---|---|---|
| `FamilySize` | `SibSp + Parch + 1` | Larger families had different survival dynamics |
| `IsAlone` | 1 if FamilySize == 1 | Solo travellers behaved differently |
| `Title` | Extracted from Name (Mr, Mrs, Miss, Master) | Title captures social status and gender better than raw name |

**Title encoding:**

Rare titles (Dr, Rev, Col, Lady, etc.) were grouped into a single `Rare` category, then all titles were encoded as numbers:

| Title | Encoded Value |
|---|---|
| Mr | 0 |
| Miss | 1 |
| Mrs | 2 |
| Master | 3 |
| Rare | 4 |

**Text to numbers:** `Sex` was encoded as `male=0, female=1`. `Embarked` was one-hot encoded into `Embarked_Q` and `Embarked_S`.

---

## Models & Results

### Model 1 — Random Forest Classifier (Baseline)

A Random Forest builds many decision trees independently and combines their votes.

```
n_estimators = 100
random_state = 42
```

| Metric | Score |
|---|---|
| Test Accuracy | **79.89%** |

### Model 2 — XGBoost Classifier (Improved)

XGBoost (Extreme Gradient Boosting) builds trees **sequentially**, where each new tree focuses on correcting the mistakes of the previous one. This makes it significantly more powerful on structured tabular data.

```
n_estimators    = 200
max_depth       = 4
learning_rate   = 0.05
subsample       = 0.8
colsample_bytree = 0.8
random_state    = 42
```

| Metric | Score |
|---|---|
| Test Accuracy | **83.24%** |
| Cross-Validation Accuracy (5-fold) | **83.84% ± 2.46%** |

### Improvement Summary

| Model | Test Accuracy | Notes |
|---|---|---|
| Random Forest | 79.89% | Baseline, no feature engineering |
| XGBoost | 83.24% | Better algorithm + feature engineering |
| **Improvement** | **+3.35%** | From algorithm switch + new features |

> Cross-validation (CV) is more reliable than a single test split because it tests the model on 5 different subsets of the data and averages the results — removing the element of luck from which 20% happened to be the test set.

---

## Key Findings

### Most Important Features (by model importance score)

1. **Fare** — Wealthier passengers had better access to lifeboats (higher decks, better position)
2. **Sex** — "Women and children first" was strictly followed by the crew
3. **Age** — Younger passengers, especially children, were prioritized

The model discovered these patterns **entirely on its own** from the numbers — it was never told any historical context about the Titanic evacuation. This is a powerful demonstration of what machine learning can extract from data.

### Sample Predictions on New Passengers

| Passenger | Profile | Prediction |
|---|---|---|
| Passenger 1 | Female, 1st class, £100 fare | ✅ SURVIVED |
| Passenger 2 | Male, 3rd class, £7.5 fare | ❌ DID NOT SURVIVE |

These results align perfectly with historical records — 1st class women had a survival rate of nearly 97%, while 3rd class men had one of the lowest survival rates on the ship.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3 | Primary programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization and charts |
| Scikit-learn | Train/test split, Random Forest, evaluation metrics |
| XGBoost | Gradient boosting model |
| Google Colab | Cloud-based notebook environment (free GPU) |

---

## How to Run

### Option 1 — Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `.ipynb` notebook file
3. Run all cells from top to bottom (`Runtime → Run all`)
---

No manual dataset download needed — the notebook fetches the data directly from the web.

---



## Future Improvements

- Try additional feature engineering (age bands, fare per person, deck from cabin number)
- Experiment with other algorithms (LightGBM, SVM, Logistic Regression ensemble)
- Use `GridSearchCV` or `Optuna` for more thorough hyperparameter tuning
- Submit to the [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic) to benchmark against thousands of other solutions

---

*First ML Project*
