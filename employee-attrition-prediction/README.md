# Employee Attrition Prediction Using Machine Learning

**Predictive Analytics – CA3 Project (B.Tech Computer Science Engineering)**

An end-to-end, reproducible machine-learning project that predicts whether an employee is likely to leave an organisation. It follows the Knowledge Discovery in Databases (KDD) process, from problem definition to a reusable prediction script, and compares **Logistic Regression**, **Random Forest** and **XGBoost** on the IBM HR Analytics Employee Attrition dataset.

> **Responsible use.** The dataset is fictional and the model is an academic exercise. Predictions are statistical estimates, not facts about a person, and must not be used to hire, fire, promote, or discipline anyone.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement and Objectives](#2-problem-statement-and-objectives)
3. [Dataset](#3-dataset)
4. [Data Cleaning and Preprocessing](#4-data-cleaning-and-preprocessing)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis-eda)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine-Learning Algorithms](#7-machine-learning-algorithms)
8. [Model Evaluation and Results](#8-model-evaluation-and-results)
9. [Threshold Selection: Recall vs Precision](#9-threshold-selection-recall-vs-precision)
10. [Key Findings and Feature Importance](#10-key-findings-and-feature-importance)
11. [Installation and How to Run](#11-installation-and-how-to-run)
12. [Repository Structure](#12-repository-structure)
13. [Limitations](#13-limitations)
14. [Future Improvements](#14-future-improvements)
15. [References and Licence](#15-references-and-licence)

---

## 1. Project Overview

| Item | Detail |
|---|---|
| Task | Supervised binary classification (`Attrition`: 1 = Left, 0 = Stayed) |
| Data | 1,470 employee records, 35 columns, 16.1% attrition |
| Models | Logistic Regression (baseline), Random Forest, XGBoost |
| Final model | Logistic Regression with balanced class weights, decision threshold 0.40 |
| Test ROC-AUC | **0.803** (95% bootstrap CI 0.724–0.880) |
| Test recall / precision | **0.766 / 0.313** at threshold 0.40 (36 of the 47 test-set leavers identified) |

Every number in this README comes from the executed notebook `notebooks/Employee_Attrition_Prediction.ipynb` and the tables in `results/`.

## 2. Problem Statement and Objectives

Employee attrition increases recruitment and onboarding costs, disrupts teams, and causes a loss of organisational knowledge. HR teams often learn about a departure only when a resignation is submitted. This project builds a classification model that estimates, from attributes such as age, job role, monthly income, job satisfaction, overtime, years at the company and work-life balance, how likely an employee is to leave. It also explains which attributes the model relies on and where its limits are.

**Objectives**

1. Understand and analyse employee-related data.
2. Identify patterns and factors associated with attrition.
3. Clean and preprocess the dataset.
4. Perform exploratory data analysis (EDA) with appropriate visualisations.
5. Engineer predictive features and measure whether they help.
6. Train Logistic Regression, Random Forest and XGBoost classifiers.
7. Evaluate and compare the models with appropriate metrics.
8. Select an operating threshold that suits the HR use case.
9. Interpret predictions and identify important features.
10. Provide a reusable, validated prediction function.

## 3. Dataset

| Property | Value |
|---|---|
| Name | IBM HR Analytics Employee Attrition & Performance |
| Source | Kaggle, uploaded by `pavansubhasht` – <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset> |
| Nature | Described by the source as a fictional data set created by IBM data scientists |
| Licence | Database: ODbL v1.0; Contents: DbCL v1.0 (redistribution allowed with attribution) |
| Size | 1,470 rows × 35 columns; no missing values; no duplicate rows |
| Target | `Attrition` – No: 1,233 (83.9%), Yes: 237 (16.1%) |
| File in this repo | `data/IBM_HR_Analytics_Employee_Attrition.csv` (**tab-delimited**; the loader detects the delimiter automatically) |

**Column groups:** demographics (age, gender, marital status), job attributes (department, role, level, overtime, business travel), compensation (monthly income, salary hike, stock options), tenure (years at company / in role / since last promotion / with manager), survey ratings (job, environment and relationship satisfaction, work-life balance, job involvement) and the target `Attrition`.

A full data dictionary is saved as `results/data_dictionary.csv`. Details of the data folder are in `data/README.md`. If you prefer not to redistribute the data, delete the file and download the original from the Kaggle link (the comma-separated original works without code changes).

## 4. Data Cleaning and Preprocessing

Results of each check are saved in `results/cleaning_log.csv`.

| Check | Finding | Action |
|---|---|---|
| Missing values | None | No imputation needed (imputers are still kept in the pipeline for unseen input) |
| Duplicates | None | Nothing removed |
| Invalid values / consistency rules | All coded scales in range; 0 violations of 5 rules | Nothing changed |
| `NumCompaniesWorked = 0` | 197 rows, all with `TotalWorkingYears = YearsAtCompany + 1` | Kept; documented as an undocumented coding convention |
| Outliers (IQR rule) | 713 flagged values in 8 columns | **Kept** – plausible values (senior levels, long tenure), not errors |
| Constant columns | `EmployeeCount`, `Over18`, `StandardHours` | Dropped (zero variance) |
| Identifier | `EmployeeNumber` | Dropped |
| Target | Yes / No | Encoded as 1 = Left, 0 = Stayed |

**Encoding and scaling.** A scikit-learn `ColumnTransformer` one-hot encodes the 7 categorical columns (`handle_unknown="ignore"`) and median-imputes numeric columns. `StandardScaler` is applied for Logistic Regression only, because tree models do not need scaling.

**Leakage prevention.** All learned steps live inside `Pipeline` objects and are fitted on the training split only (or on the training folds during cross-validation).

**Split.** 80% training (1,176 rows) / 20% test (294 rows), stratified on `Attrition`, `random_state=42`. All tuning and threshold choices use 5-fold stratified cross-validation on the training set only; the test set is used once for the final evaluation.

![Outlier box plots](visualizations/fig_4_1_outliers.png)

## 5. Exploratory Data Analysis (EDA)

The EDA covers 13 figures (all in `visualizations/`). The attrition-rate comparisons are saved in `results/eda_attrition_rate_summary.csv`.

**Class balance.** Only 16.1% of employees left, so the data is imbalanced. A model that always predicts "Stayed" already scores 84% accuracy, which is why accuracy is not the headline metric.

![Class distribution](visualizations/fig_5_1_class_distribution.png)

**Attrition rate by group (highest vs lowest group):**

| Attribute | Highest-rate group | Rate | Lowest-rate group | Rate |
|---|---|---|---|---|
| Job role | Sales Representative | 39.8% | Research Director | 2.5% |
| Tenure at company | 0–1 years (first two years) | 34.9% | 11–20 years | 6.7% |
| Age group | 18–25 | 35.8% | 36–45 | 9.2% |
| Job involvement | Level 1 (low) | 33.7% | Level 4 (very high) | 9.0% |
| Income quintile | Q1 (lowest) | 31.3% | Q5 (highest) | 9.2% |
| Job level | Level 1 | 26.3% | Level 4 | 4.7% |
| Overtime | Yes | 30.5% | No | 10.4% |
| Work-life balance | 1 (bad) | 31.3% | 3 (better) | 14.2% |
| Business travel | Travel_Frequently | 24.9% | Non-Travel | 8.0% |
| Stock option level | 0 | 24.4% | 2 | 7.6% |
| Marital status | Single | 25.5% | Divorced | 10.1% |
| Department | Sales | 20.6% | Research & Development | 13.8% |

<table>
<tr>
<td><img src="visualizations/fig_5_6_5_8_overtime_satisfaction_wlb.png" alt="Overtime, satisfaction and work-life balance"></td>
<td><img src="visualizations/fig_5_9_5_10_tenure_promotion.png" alt="Tenure and promotion"></td>
</tr>
<tr>
<td><img src="visualizations/fig_5_5_income.png" alt="Income"></td>
<td><img src="visualizations/fig_5_3_5_4_department_jobrole.png" alt="Department and job role"></td>
</tr>
</table>

![Correlation heat map](visualizations/fig_5_11_correlation.png)

Income and job level are almost perfectly correlated (r = 0.95), so their effects are difficult to separate. Other EDA figures: `fig_5_2_age.png`, `fig_5_12_additional.png`, `fig_5_13_distance_experience.png`.

> EDA shows associations only. These patterns are found in fictional data and do not establish causes.

## 6. Feature Engineering

Five features were derived from existing columns (`src/attrition_features.py`):

| Feature | Formula | Idea |
|---|---|---|
| `TenureRatio` | `YearsAtCompany / TotalWorkingYears` | Share of career spent at this company |
| `AvgYearsPerCompany` | `TotalWorkingYears / (NumCompaniesWorked + 1)` | Job-hopping tendency |
| `PromotionGapRatio` | `YearsSinceLastPromotion / (YearsAtCompany + 1)` | Time stuck without promotion |
| `MeanSatisfaction` | mean of environment, job, relationship satisfaction and work-life balance | Overall satisfaction |
| `IncomePerJobLevel` | `MonthlyIncome / JobLevel` | Pay relative to seniority |

**Did they help?** Under a rule fixed in advance, the features were tested with 5-fold cross-validation. They raised mean ROC-AUC for all three models (Logistic Regression +0.0031, Random Forest +0.0108, XGBoost +0.0027; average +0.0056) and were adopted. The gain is **smaller than the fold-to-fold variation (≈ 0.02–0.035)**, so it is a minor refinement, not a significant improvement. No columns were removed based on their relationship with the target, to avoid leaking test information into the design.

## 7. Machine-Learning Algorithms

| Model | Role | Imbalance handling | Tuned hyperparameters (5-fold CV, ROC-AUC) |
|---|---|---|---|
| Logistic Regression | Interpretable linear baseline | `class_weight="balanced"` | `C=0.1` (grid of 9 values) |
| Random Forest | Bagged ensemble of decision trees | `class_weight="balanced_subsample"` | 300 trees, `max_depth=None`, `max_features="sqrt"`, `min_samples_leaf=1` (grid of 32) |
| XGBoost | Gradient-boosted trees | `scale_pos_weight ≈ 5.19` (986 stayed / 190 left in training) | 200 trees, `learning_rate=0.1`, `max_depth=2`, `min_child_weight=10`, `subsample=0.6`, `colsample_bytree=0.4`, `reg_lambda=1`, `gamma=1` (randomised search, 40 draws) |

A `DummyClassifier` (always "Stayed") is included as the majority-class baseline. Saved in `results/tuned_hyperparameters.csv`.

## 8. Model Evaluation and Results

**Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average precision), confusion matrices, ROC and precision–recall curves, and 95% bootstrap confidence intervals on the test set.

### 8.1 Cross-validation on the training set (mean ± std, threshold 0.5)

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Majority-class baseline | 0.500 ± 0.000 | 0.162 ± 0.000 | 0.838 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | **0.833 ± 0.028** | **0.625 ± 0.071** | 0.760 ± 0.040 | 0.383 ± 0.049 | **0.737 ± 0.055** | 0.501 ± 0.040 |
| Random Forest | 0.818 ± 0.035 | 0.576 ± 0.058 | **0.860 ± 0.008** | **0.861 ± 0.134** | 0.168 ± 0.068 | 0.272 ± 0.090 |
| XGBoost | 0.829 ± 0.020 | 0.618 ± 0.051 | 0.833 ± 0.013 | 0.492 ± 0.029 | 0.679 ± 0.051 | **0.568 ± 0.014** |

### 8.2 Held-out test set (294 employees, 47 leavers, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC [95% CI] | PR-AUC |
|---|---|---|---|---|---|---|
| Majority-class baseline | 0.840 | 0.000 | 0.000 | 0.000 | 0.500 | 0.160 |
| Logistic Regression | 0.762 | 0.361 | **0.638** | 0.462 | **0.803** [0.724, 0.880] | **0.582** |
| Random Forest | **0.850** | **0.714** | 0.106 | 0.185 | 0.790 [0.709, 0.859] | 0.455 |
| XGBoost | 0.786 | 0.389 | 0.596 | **0.471** | 0.786 [0.702, 0.865] | 0.524 |

Random Forest has the best accuracy and precision only because it rarely predicts "Left" (recall 0.106), which makes it unsuitable when finding leavers is the goal.

![Confusion matrices](visualizations/fig_10_1_confusion_matrices.png)
![ROC and PR curves](visualizations/fig_10_2_10_3_roc_pr_curves.png)
![Metric comparison](visualizations/fig_10_4_metric_comparison.png)

### 8.3 Final model selection

The selection rule was fixed before the test set was used: highest cross-validated ROC-AUC, preferring the simpler model when alternatives are within one standard deviation. All three models were within one standard deviation, so the rule picked **Logistic Regression**. It also had the highest test ROC-AUC and PR-AUC, although the differences between the models are within the bootstrap uncertainty (`results/test_bootstrap_ci.csv`).

## 9. Threshold Selection: Recall vs Precision

A classifier outputs a risk probability; the **threshold** decides how high it must be to flag an employee. Lowering the threshold catches more leavers (higher recall) but flags more people who would have stayed (lower precision).

The threshold was chosen on **out-of-fold training predictions only**, never on the test set. Recall is prioritised because the intended action for a flagged employee is a low-cost, supportive conversation, so missing a leaver costs more than a false alarm.

| Threshold | Accuracy | Precision | Recall | F1 | TN / FP / FN / TP (test) |
|---|---|---|---|---|---|
| 0.50 (default) | 0.762 | 0.361 | 0.638 | 0.462 | 194 / 53 / 17 / 30 |
| 0.70 (F1-optimal) | 0.861 | 0.575 | 0.489 | 0.529 | 230 / 17 / 24 / 23 |
| **0.40 (F2-optimal, selected)** | 0.694 | 0.313 | **0.766** | 0.444 | 168 / 79 / 11 / 36 |

- **0.70** maximises F1 (balance of precision and recall): fewer false alarms (17), but 24 of 47 leavers are missed.
- **0.40** maximises F2 (recall counts twice as much as precision): it finds 36 of 47 leavers (76.6%) at the price of 79 false alarms.
- At the selected threshold, the model flags 115 of 294 test employees, and about two out of three flags are false alarms. This is acceptable only for low-cost, supportive follow-up. If interventions were expensive, a higher threshold would be more appropriate.
- ROC-AUC (0.803) does not change with the threshold; only the operating point moves along the curve.

![Threshold trade-off](visualizations/fig_10_5_threshold_tradeoff.png)

The comparison is saved in `results/final_model_threshold_comparison.csv`.

## 10. Key Findings and Feature Importance

Importance was measured by **permutation importance** (drop in ROC-AUC when a feature is shuffled) for all three models, which is comparable across model types. Ranks are in `results/permutation_importance_ranks.csv`.

| Feature | Rank in Logistic Regression | Rank in Random Forest | Rank in XGBoost |
|---|---|---|---|
| **OverTime** | 1 | 1 | 1 |
| JobSatisfaction | 7 | 2 | 5 |
| BusinessTravel | 8 | 5 | 4 |
| StockOptionLevel | 15 | 3 | 2 |
| MaritalStatus | 9 | 4 | 8 |
| MonthlyIncome | 2 | 28 | 3 |

For the final Logistic Regression model, the five most important features are `OverTime` (mean AUC drop 0.105), `MonthlyIncome` (0.061), `TotalWorkingYears` (0.036), `NumCompaniesWorked` (0.028) and `YearsAtCompany` (0.024).

![Permutation importance for the final model](visualizations/fig_11_2_permutation_importance_final.png)
![Model-specific importance](visualizations/fig_11_1_model_specific_importance.png)

**Key findings**

- **Overtime** is the strongest and most consistent predictor: attrition is 30.5% with overtime versus 10.4% without, and it ranks #1 for all three models.
- Attrition is highest early in a career: 34.9% in the first two years at the company (tenure band 0–1), 35.8% for ages 18–25, and 26.3% at job level 1.
- Lower income is associated with leaving (31.3% in the lowest income quintile versus 9–13% in the top three), but income and job level are almost perfectly correlated, so their effects overlap.
- Frequent business travel (24.9%), being single (25.5%), no stock options (24.4%), and low job involvement or satisfaction are also associated with higher attrition.
- Logistic Regression ranks employees about as well as the tree ensembles on this dataset (ROC-AUC 0.80–0.83) and is easier to explain.

Feature importance shows what the model uses, not what causes attrition, and correlated features share importance (which is why `MonthlyIncome` ranks high for two models and low for Random Forest).

## 11. Installation and How to Run

**Requirements:** Python 3.10+ (developed with Python 3.13).

```bash
git clone https://github.com/Ankit-afk-prog/employee-attrition-prediction.git
cd employee-attrition-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install notebook               # only needed to open the notebook
```

**Run the notebook** (use *Kernel → Restart & Run All*):

```bash
jupyter notebook notebooks/Employee_Attrition_Prediction.ipynb
```

The notebook works whether Jupyter is started from the repository root or from `notebooks/`. It regenerates the files in `visualizations/`, `results/` and `models/`. To run it non-interactively (a few minutes on a laptop, longer on slow machines):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/Employee_Attrition_Prediction.ipynb
```

**Score employees with the saved model** (from the repository root):

```bash
python src/predict.py                                  # scores the bundled dataset
python src/predict.py my_employees.csv scored.csv      # your own file -> output file
```

The script prints how many employees were flagged and adds `RiskScore` and `FlaggedAtRisk` columns. The input must contain the same feature columns as the training data.

Or in Python:

```python
import sys, joblib, pandas as pd
sys.path.insert(0, "src")                      # lets joblib find attrition_features.py

bundle = joblib.load("models/final_model.joblib")
employees = pd.read_csv("data/IBM_HR_Analytics_Employee_Attrition.csv", sep="\t")
risk = bundle["pipeline"].predict_proba(employees[bundle["schema"]["columns"]])[:, 1]
flagged = risk >= bundle["threshold"]
```

Use the same scikit-learn / XGBoost versions as the saved model (see the `library_versions` entry in the model bundle), or re-run the notebook to regenerate it.

**Google Colab:** upload the notebook, `src/attrition_features.py` and the CSV, set `DATA_PATH` in Section 0 to the uploaded file, and add the folder containing `attrition_features.py` to `sys.path`.

## 12. Repository Structure

```
employee-attrition-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── IBM_HR_Analytics_Employee_Attrition.csv
│   └── README.md                         # source, licence and download instructions
├── notebooks/
│   └── Employee_Attrition_Prediction.ipynb   # complete executed notebook (13 KDD stages)
├── src/
│   ├── attrition_features.py             # feature-engineering function (needed to load the saved model)
│   └── predict.py                        # command-line scoring script
├── models/
│   └── final_model.joblib                # fitted pipeline + threshold + input schema
├── visualizations/                       # 16 PNG charts generated by the notebook
├── results/                              # 10 CSV tables generated by the notebook
└── docs/
    ├── Project_Report.md                 # academic report
    ├── Presentation_Content.md           # slide plan, speaker notes, viva Q&A
    └── GitHub_Upload_Guide.md            # publishing instructions
```

## 13. Limitations

- **Fictional data.** The dataset is synthetic, so results may not reflect any real organisation.
- **Small test set.** With 1,470 records and only 47 test-set leavers, metrics are uncertain (ROC-AUC CI width ≈ 0.16).
- **Snapshot data.** A single cross-sectional snapshot with undocumented timing means there is a possible risk of tenure-related leakage, and no time-based validation was possible.
- **Undocumented columns.** `DailyRate`, `HourlyRate`, `MonthlyRate`, distance units and currency are not defined by the source.
- **Uncalibrated probabilities.** Class weighting shifts predicted probabilities, so they should be read as relative risk scores, not true probabilities.
- **Low precision at the chosen threshold.** About two out of three flags are false alarms.
- **Sensitive attributes.** The model uses gender and marital status, and no fairness audit has been performed.
- **Importance is not causation.** Feature importance shows what the model uses, not what causes attrition.

## 14. Future Improvements

Proposed, not yet done:

- Validate on newer, real and more representative data with time-based splits, and on another organisation's data.
- Calibrate the predicted probabilities.
- Add a fairness and subgroup-performance analysis, and compare with a model that excludes protected attributes.
- Choose the threshold using real intervention costs and benefits.
- Add model-explanation tools for individual predictions (e.g. SHAP).
- Monitor for data drift and test whether additional data sources improve performance.

## 15. References and Licence

1. U. Fayyad, G. Piatetsky-Shapiro, P. Smyth, "From Data Mining to Knowledge Discovery in Databases," *AI Magazine*, 17(3), 1996.
2. pavansubhasht, "IBM HR Analytics Employee Attrition & Performance," Kaggle, 2017. <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>
3. L. Breiman, "Random Forests," *Machine Learning*, 45(1), 5–32, 2001.
4. T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proc. 22nd ACM SIGKDD*, 785–794, 2016.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, 12, 2825–2830, 2011.
6. F. Fallucchi, M. Coladangelo, R. Giuliano, E. W. De Luca, "Predicting Employee Attrition Using Machine Learning Techniques," *Computers*, 9(4), 86, 2020.
7. C. Strobl, A.-L. Boulesteix, A. Zeileis, T. Hothorn, "Bias in random forest variable importance measures," *BMC Bioinformatics*, 8, 25, 2007.

The complete reference list is in `docs/Project_Report.md`.

**Licence and attribution.** Code and documentation were written for an academic CA3 submission. Dataset: *IBM HR Analytics Employee Attrition & Performance* (Kaggle, `pavansubhasht`), available under ODbL v1.0 / DbCL v1.0.
