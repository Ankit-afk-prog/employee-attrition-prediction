# Presentation Content – Employee Attrition Prediction Using Machine Learning

**Individual CA3 presentation, Predictive Analytics (October 2026)** · about 10 slides, 8–10 minutes

Every number below comes from the executed notebook. Each "Visual" line names a file in `visualizations/` that you can drop straight into the slide.

---

## Part 1 – Slide-by-slide plan

### Slide 1 – Title

**Bullets**
- Employee Attrition Prediction Using Machine Learning
- `[Student Name]` · `[Registration No.]` · `[Section]`
- Course: Predictive Analytics · Assessment: CA3
- `[University]` · `[Faculty name]` · `[Date]`

**Visual:** clean title slide; optionally a small icon of people leaving a building.

**Speaker notes:**
> "Good morning. My project is about predicting employee attrition, which means predicting which employees are likely to leave a company, using machine learning. I followed the complete KDD process, from understanding the data to building and explaining a final model."

**Possible viva question**
- *Q: Why did you choose this topic?*
  A: Attrition is expensive for companies, and the problem is a realistic example of imbalanced binary classification. That let me practise the whole predictive-analytics workflow, including the parts that are easy to get wrong, such as choosing metrics and thresholds.

---

### Slide 2 – Introduction and Problem Statement

**Bullets**
- Attrition = employees leaving the organisation
- Costs: recruitment, training, lost productivity, lost knowledge
- **Problem:** HR usually finds out only when the resignation arrives
- **Goal:** predict the probability that an employee will leave, and explain the model
- 10 objectives: understand data → EDA → preprocess → 3 models → evaluate → interpret → prediction function

**Visual:** a simple icon row: *Data → Model → Risk score → Supportive HR action*.

**Speaker notes:**
> "When an employee leaves, the company pays to hire and train a replacement and loses that person's experience. My aim was to build a classification model that gives each employee a risk score based on attributes like overtime, income, job role and satisfaction. It's important to say up front: this is a support tool for group-level planning, not a tool to make decisions about individual people."

**Possible viva questions**
- *Q: Is this classification or regression?*
  A: Classification. The target `Attrition` has two values, Yes and No, so it is binary classification. The model outputs a probability, and a threshold turns that probability into a class.
- *Q: Who would use this model?*
  A: An HR analytics team, to see where attrition risk is concentrated and to prioritise supportive actions. It should never be used to fire or penalise anyone.

---

### Slide 3 – Dataset Description

**Bullets**
- IBM HR Analytics Employee Attrition & Performance (Kaggle, `pavansubhasht`)
- **Fictional** data created by IBM data scientists · licence ODbL/DbCL
- **1,470 employees × 35 columns** (26 numeric, 9 text)
- Target: `Attrition`, with **237 Yes (16.1%)** and **1,233 No (83.9%)**, about 5.2 : 1 imbalance
- Key features: OverTime, MonthlyIncome, JobRole, JobSatisfaction, YearsAtCompany, WorkLifeBalance

**Visual:** `fig_5_1_class_distribution.png`

**Speaker notes:**
> "The dataset has 1,470 employees and 35 columns. Only 16% of them left, so the classes are imbalanced. That matters a lot. A useless model that always says 'stays' would already be 84% accurate. One more thing: the dataset is fictional, so my results describe this data, not a real company."

**Possible viva questions**
- *Q: Why is class imbalance a problem?*
  A: The model can get high accuracy by ignoring the minority class. I handled it with stratified splitting, class weights, and metrics such as recall, F1 and ROC-AUC instead of accuracy alone.
- *Q: Is the data real?*
  A: No. The Kaggle page describes it as fictional data created by IBM data scientists. That is a limitation, and I mention it in my conclusions.

---

### Slide 4 – Data Understanding and Cleaning

**Bullets**
- Missing values: **0** · Duplicate rows: **0**
- Removed **3 constant columns** (EmployeeCount, Over18, StandardHours) and **1 ID** (EmployeeNumber) → 1,470 × 31
- Outliers (IQR rule): 713 values flagged, **all kept**. For example, all high incomes belong to job levels 4–5, so they are real senior staff.
- Consistency checks (e.g. YearsAtCompany ≤ TotalWorkingYears): 0 violations
- File was tab-separated → loader detects the delimiter automatically

**Visual:** the cleaning-log table from the notebook, or `fig_4_1_outliers.png`

**Speaker notes:**
> "The data was clean: no missing values and no duplicates. I removed three columns that had the same value for everyone and the employee ID, because an ID can't generalise. I did not delete outliers automatically. When I checked them, the very high incomes all belonged to senior job levels, so they are genuine employees, and deleting them would bias the model."

**Possible viva questions**
- *Q: Why not remove outliers?*
  A: An outlier is only a problem if it is an error. These were plausible values, such as senior staff with high salaries or long-serving employees. Removing them would throw away real information.
- *Q: Why drop EmployeeNumber?*
  A: It is a unique identifier. It has no meaning for a new employee and could only teach the model spurious patterns.

---

### Slide 5 – Exploratory Data Analysis

**Bullets**
- **Overtime:** 30.5% attrition with overtime vs 10.4% without
- **Early career:** 34.9% in the first two years at the company · 35.8% for ages 18–25
- **Income:** 31.3% in the lowest income fifth vs 9–13% in the top three
- **Role:** Sales Representatives 39.8% vs Research Directors 2.5%
- Also higher with frequent travel (24.9%), single (25.5%), low job involvement (33.7%)
- These are associations, not causes

**Visual:** `fig_5_6_5_8_overtime_satisfaction_wlb.png` plus `fig_5_3_5_4_department_jobrole.png` (or `fig_5_9_5_10_tenure_promotion.png`)

**Speaker notes:**
> "In each chart, the bar is the attrition rate inside that group, and the error bar is a 95% confidence interval. The biggest single difference is overtime: people working overtime left about three times as often. Attrition is also highest early in a career: young, new, low-level, lower-paid employees. Many of these attributes overlap, and this is an association, not proof of cause."

**Possible viva questions**
- *Q: Why did you add confidence intervals?*
  A: Some groups are small. For example, only 80 employees rated work-life balance as "Bad". A rate from a small group is unreliable, and the interval shows how uncertain it is.
- *Q: Does overtime cause attrition?*
  A: We can't say that. Overtime may reflect workload, role or team, which could be the real reasons. Proving cause would need an experiment or causal analysis.

---

### Slide 6 – KDD Workflow and Preprocessing

**Bullets**
- Problem → Data → Understanding → Cleaning → EDA → Features → Encoding/Scaling → Split → Models → Evaluation → Interpretation → Prediction
- **Feature engineering:** 5 new features (e.g. AvgYearsPerCompany, MeanSatisfaction). Average gain +0.006 CV ROC-AUC, which is small.
- **One-hot encoding** for 7 categorical columns; **StandardScaler** for Logistic Regression only
- **Pipeline + ColumnTransformer** → preprocessing fitted on training data only (no leakage)
- **80/20 stratified split** (1,176 train / 294 test), seed 42 · 5-fold CV on training data for all tuning

**Visual:** the workflow diagram (Figure 3.1 in the report)

**Speaker notes:**
> "Everything is inside a scikit-learn pipeline, so the scaler and encoder learn only from the training data. That prevents data leakage. I split 80/20 with stratification so both sets keep about 16% leavers. All tuning used cross-validation on the training set, and I touched the test set only once at the end. I also measured my engineered features instead of assuming they help: the gain was positive but small."

**Possible viva questions**
- *Q: What is data leakage, and how did you prevent it?*
  A: Leakage is when information from the test set, or information unavailable at prediction time, influences training. I fitted all preprocessing inside pipelines on training data only, and never used the test set for tuning.
- *Q: Why scale only for Logistic Regression?*
  A: Logistic Regression with regularisation is sensitive to feature scale. Tree models split on one feature's threshold at a time, so scaling doesn't change them.

---

### Slide 7 – Model Development

**Bullets**
- **Logistic Regression:** linear, interpretable baseline · class_weight="balanced" · C = 0.1
- **Random Forest:** 300 decision trees on bootstrap samples, averaged · class_weight="balanced_subsample"
- **XGBoost:** boosted trees, each correcting earlier errors · scale_pos_weight ≈ 5.19 · depth 2, 200 trees
- All tuned with 5-fold stratified CV on training data, optimising ROC-AUC
- Why these three: a simple baseline, a bagging ensemble, and a boosting ensemble

**Visual:** a three-column comparison table with an icon per model

**Speaker notes:**
> "I compared three types of model. Logistic Regression is the simple baseline and easy to explain. Random Forest averages many decision trees trained on random samples. XGBoost builds small trees one after another, where each new tree fixes the mistakes of the previous ones. All three got class weights to handle imbalance and were tuned with cross-validation."

**Possible viva questions**
- *Q: Difference between Random Forest and XGBoost?*
  A: Random Forest builds trees independently, in parallel, on bootstrap samples, and averages them, which reduces variance. XGBoost builds trees sequentially, each fitting the remaining errors, which reduces bias, and it adds regularisation.
- *Q: Why did you expect XGBoost might not win?*
  A: With only 1,470 rows and mostly simple relationships, a well-regularised linear model can perform just as well. That is what happened here.

---

### Slide 8 – Model Evaluation

**Bullets**

Test set: 294 employees, 47 leavers, threshold 0.5.

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | **0.803** | **0.638** | 0.361 | 0.462 | 0.762 |
| Random Forest | 0.790 | 0.106 | **0.714** | 0.185 | **0.850** |
| XGBoost | 0.786 | 0.596 | 0.389 | **0.471** | 0.786 |

- RF has the best accuracy but catches only 5 of 47 leavers: **accuracy is misleading**
- ROC-AUC 95% intervals overlap (≈ 0.70–0.88), so the models are statistically very close
- **Final model: Logistic Regression** (best CV ROC-AUC 0.833, simplest)
- Threshold **0.40**, chosen on training CV with F2 (recall-weighted) → **recall 0.766** (36/47 leavers), precision 0.313

**Visual:** `fig_10_1_confusion_matrices.png` and `fig_10_2_10_3_roc_pr_curves.png` (split across two slides if needed)

**Speaker notes:**
> "This slide shows why accuracy is not enough. Random Forest has the highest accuracy, 85%, but it finds only 5 of the 47 people who actually left. Logistic Regression has the best ROC-AUC, 0.80, but the confidence intervals overlap, so honestly the three models rank employees about equally well. I chose Logistic Regression using a rule I fixed in advance: best cross-validation score, and prefer the simpler model if they're close. Then I lowered the threshold to 0.40 because missing a leaver is worse than a false alarm. It now finds 77% of leavers, but about two out of three flagged people are false alarms."

**Possible viva questions**
- *Q: Why lower the threshold to 0.40?*
  A: Our priority is recall. I chose the threshold that maximises F2, which weights recall twice as much as precision, using out-of-fold predictions on the training data only, not the test set.
- *Q: Isn't 31% precision bad?*
  A: It's a trade-off. If the follow-up is a cheap, friendly check-in conversation, false alarms cost little. If HR capacity is limited, the 0.70 threshold gives 57.5% precision but only 49% recall.

---

### Slide 9 – Feature Importance and Interpretation

**Bullets**
- **OverTime is #1 in all three models** (permutation importance)
- Final model's top features: OverTime (0.105 drop in AUC), MonthlyIncome, TotalWorkingYears, NumCompaniesWorked, YearsAtCompany, YearsSinceLastPromotion, JobSatisfaction
- Logistic Regression: overtime ≈ **4× higher odds** of leaving; frequent travel, Lab Technician, Sales Rep and Single raise risk
- Correlated features share importance (income ranks #2 for LR but #28 for RF)
- **Importance ≠ cause.** Gender and marital status are used, so a fairness audit is needed before any real use.

**Visual:** `fig_11_2_permutation_importance_final.png`

**Speaker notes:**
> "To explain the model, I used permutation importance. I shuffle one column and see how much the model's ROC-AUC drops. Overtime matters most in every model. For Logistic Regression, working overtime is linked to about four times higher odds of leaving, other things equal. But importance tells us what the model uses, not what causes people to leave. The model also uses gender and marital status, which is ethically sensitive, so a real deployment would need a fairness check."

**Possible viva questions**
- *Q: What is permutation importance?*
  A: Shuffle one feature's values in the test set and measure how much performance drops. A big drop means the model relies on that feature. It works for any model.
- *Q: Why does income rank so differently between models?*
  A: Income is highly correlated with job level (r = 0.95) and experience. Random Forest can get the same information from those columns, so shuffling income alone barely hurts it.

---

### Slide 10 – Conclusion and Future Scope

**Bullets**
- Built a complete, reproducible KDD pipeline; all 10 objectives met
- Best ranking: Logistic Regression, test ROC-AUC 0.803; finds 77% of leavers at threshold 0.40
- Key finding: overtime, early-career pay and tenure, and satisfaction are associated with attrition
- **Limitations:** fictional and small data, single snapshot, no fairness audit, uncalibrated probabilities
- **Future:** real and newer data, calibration, fairness analysis, external validation, drift monitoring
- Use it to *support* humane HR decisions, never to replace them

**Visual:** a summary graphic with three boxes: *What I built · What I found · What's next*

**Speaker notes:**
> "To conclude, I built and documented the full predictive-analytics workflow. The final Logistic Regression model ranks employees by risk much better than chance and catches about three-quarters of leavers, at the cost of false alarms. The strongest signal is overtime. The main limitations are that the data is fictional and small. The next steps would be real data, calibrated probabilities and a fairness analysis. Thank you, and I'm happy to take questions."

**Possible viva question**
- *Q: What would you do differently with more time?*
  A: Validate on real data with a time-based split, calibrate the probabilities, check performance separately for different groups such as gender and age, and compare a version of the model without sensitive attributes.

---

## Part 2 – Two-minute project explanation ("Explain your project")

> "My project predicts employee attrition, which means whether an employee is likely to leave the company, using machine learning. I used the IBM HR Analytics dataset from Kaggle, which has 1,470 fictional employees and 35 attributes such as age, job role, income, overtime, satisfaction ratings and years at the company. Only 16% of employees left, so the data is imbalanced.
>
> I followed the KDD process. First I checked the data: there were no missing values or duplicates, and I removed three constant columns and the employee ID. In the exploratory analysis I found that attrition is much higher for people working overtime, 30% versus 10%, and for employees who are early in their careers, low-paid or in roles like Sales Representative.
>
> For modelling, I put all preprocessing, meaning one-hot encoding and scaling, inside scikit-learn pipelines, so nothing was learned from the test data. I split the data 80/20 with stratification and trained three models: Logistic Regression, Random Forest and XGBoost. I tuned them with 5-fold cross-validation and used class weights for the imbalance.
>
> On the test set all three had ROC-AUC around 0.79 to 0.80, and their confidence intervals overlap. I chose Logistic Regression because it had the best cross-validation score and is the simplest to explain. Random Forest had the highest accuracy but caught only 5 of 47 leavers, which shows why accuracy is misleading. Because missing a leaver matters more than a false alarm, I set the threshold to 0.40 using training data. The model then finds about 77% of leavers with about 31% precision.
>
> Feature importance showed that overtime is the strongest predictor in every model, followed by income, experience, tenure and satisfaction. These are associations, not causes, and the data is fictional. So this model should only support group-level HR decisions, never decisions about individuals. Future work would be real data, probability calibration and a fairness analysis."

---

## Part 3 – Viva questions and answers (30)

**1. What is predictive analytics?**
Using historical data with statistical and machine-learning models to estimate unknown or future outcomes, such as whether an employee will leave. Descriptive analytics asks "what happened", predictive analytics asks "what is likely to happen", and prescriptive analytics asks "what should we do".

**2. What is the KDD process?**
Knowledge Discovery in Databases (Fayyad et al., 1996) is the overall process of turning data into useful knowledge: selection, preprocessing, transformation, data mining and interpretation/evaluation. In my project it became 13 stages, from problem definition to final prediction.

**3. What is employee attrition?**
Employees leaving an organisation, whether voluntarily (resignation) or involuntarily. In this dataset it is the `Attrition` column, with values Yes or No.

**4. Why is this a classification problem?**
The target is a category (Left/Stayed), not a continuous number. With two classes it is *binary* classification. The model outputs a probability, and a threshold turns it into a class.

**5. What is the target variable?**
`Attrition`. I verified that it exists and has exactly two values, then encoded Yes = 1 (Left) and No = 0 (Stayed).

**6. Why was this dataset used?**
It is public, well documented for its coded scales, openly licensed (ODbL), complete (no missing values), and contains exactly the kinds of variables named in the problem statement. Its main weakness is that it is fictional.

**7. What is exploratory data analysis?**
Summarising and visualising data before modelling to understand distributions, spot problems and find relationships. I plotted attrition *rates* by group with confidence intervals, plus a correlation heatmap.

**8. Why check missing values and duplicates?**
Missing values can break models or bias results, and duplicates can over-weight some records and leak between train and test sets. Here there were none, but checking is necessary before trusting the data.

**9. What is feature engineering?**
Creating new input variables from existing ones to expose useful information. I created five, such as AvgYearsPerCompany = TotalWorkingYears / (NumCompaniesWorked + 1), and measured their effect with cross-validation: a small average gain of +0.006 ROC-AUC.

**10. Why is categorical encoding necessary?**
ML models need numbers. One-hot encoding turns each category into its own 0/1 column without implying a false order (Sales is not "greater than" R&D). `handle_unknown="ignore"` handles new categories safely.

**11. Why is feature scaling useful for Logistic Regression?**
Regularisation penalises all coefficients equally, so features must be on comparable scales, and the solver converges faster. Standardisation gives each feature mean 0 and standard deviation 1. Tree models don't need it.

**12. What is stratified sampling?**
Splitting so that each subset keeps the same class proportions as the full data. My training and test sets have 16.2% and 16.0% leavers respectively.

**13. What is overfitting?**
When a model learns noise in the training data and performs worse on new data. My Random Forest had training-fold ROC-AUC 1.000 but cross-validated ROC-AUC 0.818, a clear sign of overfitting in the individual trees.

**14. What is cross-validation?**
Splitting the training data into k folds (I used 5), training on k−1 folds and validating on the remaining one, rotating through all folds, and averaging the scores. It gives a more reliable estimate than a single split and lets me tune without touching the test set.

**15. How does Logistic Regression work?**
It computes a weighted sum of the features, z = β₀ + β₁x₁ + …, and passes it through the sigmoid 1/(1+e^(−z)) to get a probability. Coefficients are learned by minimising log-loss. e^β is the odds ratio for a one-unit increase.

**16. How does Random Forest work?**
It trains many decision trees, each on a bootstrap sample of the rows and considering a random subset of features at each split, then averages their votes. Averaging decorrelated trees reduces variance (Breiman, 2001).

**17. How does XGBoost work?**
It builds shallow trees one after another. Each new tree is fitted to the errors (gradients) of the current ensemble and added with a small learning rate. It adds regularisation on tree complexity, plus row and column subsampling, to limit overfitting (Chen & Guestrin, 2016).

**18. What is class imbalance, and how did you handle it?**
One class is much rarer than the other (16% vs 84%). I used a stratified split, class weights (balanced / balanced_subsample / scale_pos_weight ≈ 5.19), metrics that focus on the minority class, and a recall-oriented threshold.

**19. What is a confusion matrix?**
A 2×2 table of actual versus predicted classes: TN, FP, FN, TP. For my final model at threshold 0.40 on the test set it is TN = 168, FP = 79, FN = 11, TP = 36.

**20. Define precision, recall, F1 and ROC-AUC.**
- Precision = TP/(TP+FP): of the employees flagged, how many really left.
- Recall = TP/(TP+FN): of those who left, how many we flagged.
- F1 = the harmonic mean of precision and recall.
- ROC-AUC = the probability that a random leaver gets a higher score than a random stayer; 0.5 is random and 1.0 is perfect.

**21. Why might accuracy be misleading?**
With 84% stayers, predicting "stays" for everyone gives about 84% accuracy and zero recall. In my results, Random Forest had the highest accuracy (85.0%) but found only 5 of 47 leavers.

**22. How was the final model selected?**
By a rule fixed before using the test set: highest cross-validated ROC-AUC, preferring the simpler model if the others are within one standard deviation. All three were within one SD, so Logistic Regression was chosen. The test set then confirmed it had the best ROC-AUC (0.803) and PR-AUC (0.582).

**23. What does feature importance mean?**
How much the model relies on a feature for its predictions. Permutation importance measures the drop in ROC-AUC when the feature is shuffled. It is *not* causal importance, and correlated features share importance.

**24. What is data leakage?**
Using information during training that wouldn't be available at prediction time, or that comes from the test set. That makes scores look better than reality. I avoided it with pipelines fitted on training data only, by never tuning on the test set, and by dropping the ID. I also flagged that tenure values for leavers may have been recorded at exit.

**25. What are the ethical limitations of using ML for HR decisions?**
- Predictions can be wrong: 79 false alarms in my test set.
- Models can encode bias, and mine uses gender and marital status.
- Employees may react badly to being scored.
- Decisions about people need human accountability and transparency (Tambe et al., 2019).

So the model should only support supportive, group-level actions, never hiring, firing or penalising individuals.

**26. What is ROC-AUC versus PR-AUC, and why report both?**
ROC-AUC uses the false-positive rate, which can look optimistic when negatives dominate. PR-AUC focuses on the positive class. A random model has PR-AUC equal to the prevalence (0.16), and my final model reached 0.582 (Saito & Rehmsmeier, 2015).

**27. Why did you choose the threshold with F2 instead of F1?**
F2 weights recall twice as much as precision, matching my stated priority that missing a leaver is costlier than a false alarm. I chose it on out-of-fold *training* predictions, not on the test set.

**28. What are bootstrap confidence intervals, and what did they show?**
I resampled the 294 test rows with replacement 1,000 times and recomputed the metrics each time. The 95% ROC-AUC intervals (LR 0.724–0.880, RF 0.709–0.859, XGB 0.702–0.865) overlap heavily, so the test set cannot reliably separate the models.

**29. Why did Random Forest have such low recall at 0.5?**
Its trees are fully grown with pure leaves, so class weights barely shift the averaged votes, and few leavers reach above 0.5. Its *ranking* (ROC-AUC 0.79) is fine. It just needs a lower threshold.

**30. What would you improve?**
Real and newer data with a time-based split, probability calibration, a fairness and subgroup analysis, a version without sensitive attributes, external validation, and choosing the threshold from real costs.
