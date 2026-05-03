# Heart Disease Prediction

**Author:** Egor Liu  
**Project type:** Machine Learning portfolio project

Binary classification project — predicting heart disease from patient clinical data using Logistic Regression, Decision Tree, and Random Forest.

The goal was to build a full ML workflow, not just train one model: data quality check, EDA, preprocessing, baseline comparison, overfitting analysis, cross-validation, and final model selection.

---

## My Focus

In this project, I focused on building the workflow step by step instead of only training one final model.

The most important parts for me were:

- checking data quality and removing duplicated rows
- comparing models against a baseline
- understanding overfitting in Decision Trees
- using cross-validation after seeing that one train/test split may be unstable
- explaining why the final model changed from Decision Tree to Random Forest `max_depth=3`

---

## Dataset

UCI Heart Disease dataset:

| Item | Value |
|---|---:|
| Original rows | 1025 |
| Features + target | 14 |
| Duplicate rows removed | 723 |
| Final rows | 302 |

Target:

| Target | Meaning | Count |
|---|---|---:|
| 0 | No heart disease | 138 |
| 1 | Heart disease | 164 |

The classes are fairly balanced, so accuracy is useful.

But since this is a medical-style classification task, I also focus on recall and F1-score.

A false negative is worse here: the model says “no heart disease”, but the patient actually has heart disease.

---

## Workflow

Main steps:

1. Load and inspect the dataset
2. Remove duplicated rows
3. Rename unclear medical abbreviations
4. Analyze target distribution
5. Run EDA with plots and correlations
6. Split data into train/test sets
7. Scale features for Logistic Regression
8. Train baseline, Logistic Regression, Decision Tree, and Random Forest models
9. Check overfitting with train/test accuracy
10. Use cross-validation for a more stable comparison
11. Select final model
12. Analyze confusion matrix and feature importance

---

## Models

Models tested:

- Dummy Classifier baseline
- Logistic Regression
- Decision Tree
- Decision Tree `max_depth=3`
- Random Forest
- Random Forest `max_depth=3`

---

## Test Set Results

On one train/test split, Decision Tree `max_depth=3` ranked first.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline | 0.539 | 0.539 | 1.000 | 0.701 |
| Logistic Regression | 0.803 | 0.783 | 0.878 | 0.828 |
| Decision Tree max_depth=3 | 0.816 | 0.814 | 0.854 | 0.833 |
| Random Forest | 0.803 | 0.810 | 0.829 | 0.819 |
| Random Forest max_depth=3 | 0.789 | 0.778 | 0.854 | 0.814 |

At this point, Decision Tree looked like the best model.

But with only ~300 rows, one split can be noisy.

---

## Cross-Validation Results

To avoid trusting one split too much, I used 5-fold stratified cross-validation.

| Model | CV Accuracy | CV Precision | CV Recall | CV F1-score |
|---|---:|---:|---:|---:|
| Baseline | 0.544 | 0.544 | 1.000 | 0.705 |
| Logistic Regression | 0.814 | 0.810 | 0.862 | 0.835 |
| Decision Tree max_depth=3 | 0.747 | 0.750 | 0.806 | 0.774 |
| Random Forest | 0.836 | 0.833 | 0.879 | 0.854 |
| Random Forest max_depth=3 | 0.845 | 0.824 | 0.911 | 0.865 |

Cross-validation changed the ranking.

The best average F1-score came from **Random Forest `max_depth=3`**.

---

## Final Model

Final model:

RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

I selected this model because it had the best cross-validation F1-score and the highest CV recall.

The normal Random Forest also performed well, but the limited-depth version gave a slightly better and more stable result.

---

## Feature Importance

The final Random Forest mostly relied on features such as:

- chest pain type
- number of major vessels
- thalassemia-related category
- ST depression
- maximum heart rate

This does **not** mean these features directly cause heart disease.

It only means they were useful for this model on this dataset.

---

## What I Learned

Main takeaways:

- Accuracy alone is not enough.
- Baseline models are useful for sanity checks.
- Decision Trees can overfit hard.
- Limiting depth can help, but too much restriction can hurt.
- One train/test split can give a misleading winner.
- Cross-validation gives a better view when the dataset is small.
- Feature importance helps interpret the model, but it is not medical proof.

---

## Limitations

This is not a medical diagnostic tool.

Main limitations:

- small dataset after duplicate removal
- only one dataset used
- no external validation
- no systematic GridSearchCV yet
- feature importance is model-based, not causal

---

## Possible Improvements

Next improvements could be:

- tune hyperparameters with GridSearchCV
- test on another heart disease dataset
- compare feature importance across models
- add SHAP or permutation importance
- calibrate predicted probabilities

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Open the notebook:

jupyter notebook notebook/heart_disease_classification.ipynb

Dataset path:

data/heart.csv

---

## Project Structure

project/
│
├── data/
│   └── heart.csv
│
├── notebook/
│   └── heart_disease_classification.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore