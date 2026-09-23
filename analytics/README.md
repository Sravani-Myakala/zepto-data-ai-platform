\# Module 2 - Part A: Exploratory Data Analysis



\## Dataset



The Titanic dataset was loaded using `sns.load\_dataset("titanic")` and saved as `analytics/titanic.csv`. The original dataset contains 891 rows and 15 columns.



\## Missing Value Handling



\* `age`: 19.87% missing values. Missing values were replaced using the median value of 28.00.

\* `embarked`: 0.22% missing values. Rows with missing values were dropped because the percentage was below 5%.

\* `deck`: 77.22% missing values. The column was dropped because the percentage of missing values was very high.

\* `embark\_town`: 0.22% missing values. The same rows removed for missing `embarked` also removed these missing values.

\* Final cleaned dataset contains 889 rows and 14 columns.

\* No missing values remain in the cleaned dataset.



\## Age Analysis



The age distribution was analyzed using a histogram and box plot. The IQR method identified 65 age outliers using the bounds 2.50 and 54.50. Most passengers were concentrated around the younger and middle-age groups.



\## Fare Analysis



The fare distribution was analyzed using a histogram and box plot. The IQR method identified 114 fare outliers using the upper bound of 65.66. Fare is positively/right skewed because the mean (32.10) is greater than the median (14.45), which is greater than the mode (8.05).



\## Survival Analysis



Female passengers had a survival rate of 74.04%, while male passengers had a survival rate of 18.89%. First-class passengers had the highest survival rate at 62.62%, followed by second class at 47.28% and third class at 24.24%.



\## Correlation Analysis



The correlation matrix was calculated using `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`.



The two strongest correlations were `pclass` and `fare` with a correlation of -0.548, and `sibsp` and `parch` with a correlation of 0.415. The negative correlation between class and fare indicates that passenger class and fare are related, while the positive correlation between siblings/spouses and parents/children indicates that passengers traveling with family members in one group tended to have related family counts.



\# Multivariate Data Story



\## 1. Survival by Sex and Passenger Class



The chart shows that survival varied considerably according to both sex and passenger class. Female passengers had higher survival rates than male passengers across all three classes. First-class females had the highest survival rate at about 96.74%, while third-class males had one of the lowest rates at about 13.54%.



\## 2. Age Distribution by Survival



The age distribution shows differences between passengers who survived and those who did not. Survivors and non-survivors were present across several age groups, showing that age alone did not completely determine survival. The chart helps identify how the age distribution differs between the two survival groups.



\## 3. Fare by Passenger Class and Survival



The box plot shows the relationship between fare, passenger class, and survival. First-class passengers generally paid higher fares than passengers in second and third class. The distribution also shows that fare values vary substantially within classes, particularly among passengers who paid higher prices.



\## 4. Family Size and Survival



The family-size chart shows how the number of family members traveling with a passenger relates to survival. Passengers traveling alone or with smaller family groups showed different survival patterns compared with passengers traveling in larger groups. This suggests that family size can provide additional information when studying passenger survival.



\## EDA-Only Standardization



The `age` and `fare` columns were standardized using `StandardScaler` on the full cleaned dataset for EDA purposes only. After standardization, their means were approximately 0 and standard deviations were approximately 1. These standardized values are not used as inputs for the machine-learning models.



\## Generated Plot Files



The following plots were generated:



\* `age\_histogram.png`

\* `age\_boxplot.png`

\* `fare\_histogram.png`

\* `fare\_boxplot.png`

\* `correlation\_heatmap.png`

\* `survival\_by\_sex\_class.png`

\* `age\_by\_survival.png`

\* `fare\_class\_survival.png`

\* `family\_size\_survival.png`



\## Part A Status



\*\*Module 2 - Part A: Exploratory Data Analysis COMPLETED\*\*



# Module 2 - Part B: Machine Learning

## Dataset and Target

The Titanic dataset saved in `analytics/titanic.csv` was used for machine-learning experiments. The target variable is `survived`.

The dataset contains 891 rows. A stratified 80/20 train-test split was used with `random_state=42`, producing 712 training rows and 179 test rows.

The class distribution was:

* Class 0 (Not Survived): 549 passengers (61.62%)
* Class 1 (Survived): 342 passengers (38.38%)

Stratification was used to maintain approximately the same class proportions in the training and test sets.

---

## Preprocessing

The following features were used for classification:

* `pclass`
* `sex`
* `age`
* `sibsp`
* `parch`
* `fare`
* `embarked`

The `alive` column was excluded because it directly represents the survival outcome. The `deck` column was excluded because it contains a very high percentage of missing values.

Numeric features:

* `pclass`
* `age`
* `sibsp`
* `parch`
* `fare`

Categorical features:

* `sex`
* `embarked`

Numeric missing values were handled using median imputation followed by `StandardScaler`.

Categorical missing values were handled using most-frequent imputation followed by one-hot encoding.

The preprocessing was fitted only on the training data and then applied to the test data.

---

## Classification Models

Three classification algorithms were trained using the same stratified train-test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Decision Tree was limited to `max_depth=5`.

A decision-tree visualization was generated using `plot_tree`.

---

## Classification Evaluation

The three models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

### Results

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |   0.8045 |    0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree       |   0.7654 |    0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest       |   0.8156 |    0.8000 | 0.6957 | 0.7442 | 0.8287 |

The Random Forest achieved the highest observed F1 score (0.7442) and accuracy (0.8156) in this test-set comparison. Logistic Regression produced the highest ROC-AUC value (0.8437). Therefore, the final classifier selection was based specifically on the observed F1 criterion rather than assuming that one model performed highest on every metric.

---

## Class Imbalance Comparison

The original dataset contained:

* Class 0: 549 samples (61.62%)
* Class 1: 342 samples (38.38%)

Logistic Regression was compared using three approaches:

1. Baseline
2. `class_weight="balanced"`
3. SMOTE applied only to the training data

### Results

| Method                | Precision | Recall |     F1 |
| --------------------- | --------: | -----: | -----: |
| Baseline              |    0.7931 | 0.6667 | 0.7244 |
| Class Weight Balanced |    0.7297 | 0.7826 | 0.7552 |
| SMOTE                 |    0.7397 | 0.7826 | 0.7606 |

The balanced and SMOTE approaches increased recall compared with the baseline. SMOTE produced the highest F1 score among these three imbalance treatments at 0.7606.

SMOTE was applied only to the training data so that synthetic samples did not enter the test set.

---

## Random Forest GridSearchCV

GridSearchCV with 5-fold cross-validation was applied to the Random Forest.

The searched parameters were:

* `n_estimators`: 100, 200
* `max_depth`: None, 5, 10
* `max_features`: `sqrt`, `log2`

The search used F1 score as the optimization metric.

### Best Parameters

```text
max_depth = 5
max_features = sqrt
n_estimators = 100
```

Best cross-validation F1 score:

```text
0.7459
```

Out-of-bag (OOB) score:

```text
0.8272
```

The tuned Random Forest was evaluated separately on the test set.

---

## Regression Side-Task

A multivariate Linear Regression model was used to predict `fare`.

Features used:

* `pclass`
* `age`
* `sibsp`
* `parch`
* `sex`
* `embarked`

The model was evaluated using:

* MAE
* RMSE
* R²
* Adjusted R²

### Regression Results

| Metric      |  Result |
| ----------- | ------: |
| MAE         | 20.8094 |
| RMSE        | 30.4731 |
| R²          |  0.3999 |
| Adjusted R² |  0.3679 |

The residual plot showed that residual spread increased as predicted fare increased. This suggests heteroscedasticity, meaning that the variance of prediction errors was not constant across the range of predicted fare values.

The residual plot is saved as:

`plots/fare_regression_residuals.png`

---

## Final Model Selection

The final classifier was selected using the observed test-set F1 score from the three classifier comparison.

The selected model was:

**Random Forest**

Test-set metrics:

* Accuracy: 0.8156
* Precision: 0.8000
* Recall: 0.6957
* F1: 0.7442
* ROC-AUC: 0.8287

The Random Forest was selected based on the highest observed F1 score among the three directly compared classifiers. Logistic Regression had a higher ROC-AUC, so the metrics should be interpreted separately.

---

## Saved Model Pipeline

The complete fitted preprocessing and Random Forest classifier pipeline was saved using Joblib:

```text
models/best_classifier_pipeline.joblib
```

The saved pipeline includes both preprocessing and the classifier.

The pipeline was reloaded using `joblib.load()` and tested using raw, unprocessed input. The test successfully generated a prediction.

Example raw input:

```text
pclass = 3
sex = female
age = 25
sibsp = 0
parch = 0
fare = 15.0
embarked = S
```

The reloaded pipeline generated:

```text
Predicted class: 1
Survival probability: 0.5500
```

This confirms that the complete saved pipeline can accept raw input without requiring separate manual preprocessing.

---

## Generated Machine Learning Files

Important generated files include:

```text
model_comparison.csv
model_evaluation_results.txt
imbalance_comparison.csv
imbalance_comparison.txt
gridsearch_results.txt
regression_results.csv
regression_results.txt
final_model_comparison.csv
final_model_recommendation.txt
models/best_classifier_pipeline.joblib
```

Important plots include:

```text
decision_tree.png
fare_regression_residuals.png
```

along with the confusion-matrix and EDA plots stored in the `plots` directory.

---

## Module 2 Status

**Module 2 - Part A: Exploratory Data Analysis COMPLETED**

**Module 2 - Part B: Machine Learning COMPLETED**
