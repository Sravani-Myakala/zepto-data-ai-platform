# ============================================================
# MODULE 2 - PART B: STEP 7
# REGRESSION SIDE-TASK - PREDICT FARE
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "analytics/titanic.csv"
PLOTS_DIR = "analytics/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("MODULE 2 - PART B: STEP 7")
print("REGRESSION SIDE-TASK - PREDICT FARE")
print("=" * 70)

print("\nDATASET LOADED")
print("Shape:", df.shape)


# ============================================================
# 2. SELECT FEATURES AND TARGET
# ============================================================

# Target variable
y = df["fare"]

# Features used to predict fare
features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

X = df[features].copy()

print("\nFEATURES USED:")
for feature in features:
    print("-", feature)

print("\nTARGET:")
print("- fare")


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMISSING VALUES BEFORE PREPROCESSING:")
print(X.isnull().sum())

print("\nMissing target values:", y.isnull().sum())


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN / TEST SPLIT")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 5. DEFINE PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]


# Numeric preprocessing
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 6. CREATE LINEAR REGRESSION PIPELINE
# ============================================================

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTRAINING MULTIVARIATE LINEAR REGRESSION...")

regression_pipeline.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_pred = regression_pipeline.predict(X_test)

print("\nPREDICTIONS GENERATED")
print("Number of predictions:", len(y_pred))


# ============================================================
# 9. CALCULATE REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


# Number of observations
n = len(y_test)

# Number of predictors after preprocessing
X_test_processed = regression_pipeline.named_steps[
    "preprocessor"
].transform(X_test)

p = X_test_processed.shape[1]

# Adjusted R-squared
adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)


# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION RESULTS")
print("=" * 70)

print(f"\nMAE          : {mae:.4f}")
print(f"RMSE         : {rmse:.4f}")
print(f"R²           : {r2:.4f}")
print(f"Adjusted R²  : {adjusted_r2:.4f}")

print("\nNumber of test observations:", n)
print("Number of predictors after encoding:", p)


# ============================================================
# 11. CREATE RESIDUALS
# ============================================================

residuals = y_test - y_pred


print("\nRESIDUAL INFORMATION")
print("Mean residual:", residuals.mean())
print("Residual standard deviation:", residuals.std())


# ============================================================
# 12. CHECK RESIDUAL SPREAD
# ============================================================

results_df = pd.DataFrame({
    "actual_fare": y_test.values,
    "predicted_fare": y_pred,
    "residual": residuals.values
})

# Divide predictions into four groups
results_df["prediction_group"] = pd.qcut(
    results_df["predicted_fare"],
    q=4,
    duplicates="drop"
)

residual_spread = (
    results_df
    .groupby("prediction_group", observed=True)["residual"]
    .std()
)

print("\nRESIDUAL SPREAD BY PREDICTED-FARE GROUP:")
print(residual_spread)


# ============================================================
# 13. HETEROSCEDASTICITY CONCLUSION
# ============================================================

spread_values = residual_spread.values

if len(spread_values) >= 2:

    first_spread = spread_values[0]
    last_spread = spread_values[-1]

    if first_spread > 0 and last_spread > first_spread * 1.5:
        hetero_conclusion = (
            "Residual spread increases noticeably as predicted fare "
            "increases, suggesting heteroscedasticity."
        )

    elif first_spread > 0 and last_spread < first_spread / 1.5:
        hetero_conclusion = (
            "Residual spread decreases across predicted fare groups. "
            "The residual pattern is not constant, indicating possible "
            "heteroscedasticity."
        )

    else:
        hetero_conclusion = (
            "Residual spread is broadly similar across predicted fare "
            "groups, so there is no strong evidence of heteroscedasticity "
            "from this residual analysis."
        )

else:
    hetero_conclusion = (
        "There are not enough residual groups to assess "
        "heteroscedasticity."
    )


print("\nHETEROSCEDASTICITY CONCLUSION:")
print(hetero_conclusion)


# ============================================================
# 14. RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Prediction")

plt.tight_layout()

residual_plot_path = os.path.join(
    PLOTS_DIR,
    "fare_regression_residuals.png"
)

plt.savefig(
    residual_plot_path,
    dpi=300
)

plt.close()

print("\nRESIDUAL PLOT SAVED:")
print(residual_plot_path)


# ============================================================
# 15. SAVE REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame({
    "Model": ["Multivariate Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "Adjusted_R2": [adjusted_r2]
})

results_csv_path = "analytics/regression_results.csv"

regression_results.to_csv(
    results_csv_path,
    index=False
)


# ============================================================
# 16. SAVE DETAILED TEXT REPORT
# ============================================================

report_path = "analytics/regression_results.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("MODULE 2 - PART B - STEP 7\n")
    file.write("REGRESSION SIDE-TASK - PREDICT FARE\n")
    file.write("=" * 70 + "\n\n")

    file.write("Target: fare\n\n")

    file.write("Features used:\n")

    for feature in features:
        file.write(f"- {feature}\n")

    file.write("\nRegression Metrics:\n")
    file.write(f"MAE: {mae:.4f}\n")
    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"R2: {r2:.4f}\n")
    file.write(f"Adjusted R2: {adjusted_r2:.4f}\n")

    file.write("\n")
    file.write(f"Test observations: {n}\n")
    file.write(f"Predictors after encoding: {p}\n")

    file.write("\nResidual Analysis:\n")
    file.write(f"Mean residual: {residuals.mean():.6f}\n")
    file.write(
        f"Residual standard deviation: {residuals.std():.6f}\n"
    )

    file.write("\nHeteroscedasticity conclusion:\n")
    file.write(hetero_conclusion + "\n")


print("\nRESULT FILES SAVED:")
print(results_csv_path)
print(report_path)


# ============================================================
# 17. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)
# ============================================================
# MODULE 2 - PART B: STEP 8
# FINAL MODEL COMPARISON AND COMPLETE PIPELINE
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "analytics/titanic.csv"
MODEL_DIR = "analytics/models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n")
print("=" * 70)
print("MODULE 2 - PART B: STEP 8")
print("FINAL MODEL COMPARISON AND COMPLETE PIPELINE")
print("=" * 70)


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

# We do not use 'alive' because it directly represents the target.
# We also do not use 'deck' because it has very high missing values.

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features].copy()
y = df["survived"]


print("\nFEATURES USED:")
for feature in features:
    print("-", feature)

print("\nTARGET:")
print("- survived")


# ============================================================
# 3. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAIN / TEST DATA")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# 4. DEFINE PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 5. CREATE THREE CLASSIFIERS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}


# ============================================================
# 6. TRAIN AND EVALUATE ALL THREE
# ============================================================

results = []

fitted_pipelines = {}

print("\n")
print("=" * 70)
print("FINAL CLASSIFIER COMPARISON")
print("=" * 70)

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    print(f"\nTraining: {model_name}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    })

    fitted_pipelines[model_name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")


# ============================================================
# 7. CREATE COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("FINAL CLASSIFIER COMPARISON TABLE")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. IDENTIFY BEST MODEL USING F1 SCORE
# ============================================================

best_index = comparison_df["F1"].idxmax()

best_model_name = comparison_df.loc[
    best_index,
    "Model"
]

best_f1 = comparison_df.loc[
    best_index,
    "F1"
]

best_accuracy = comparison_df.loc[
    best_index,
    "Accuracy"
]

best_precision = comparison_df.loc[
    best_index,
    "Precision"
]

best_recall = comparison_df.loc[
    best_index,
    "Recall"
]

best_auc = comparison_df.loc[
    best_index,
    "AUC"
]


print("\n")
print("=" * 70)
print("MODEL SELECTED USING TEST-SET F1")
print("=" * 70)

print("Selected model:", best_model_name)
print(f"Accuracy : {best_accuracy:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall   : {best_recall:.4f}")
print(f"F1       : {best_f1:.4f}")
print(f"AUC      : {best_auc:.4f}")


# ============================================================
# 9. SAVE FINAL COMPARISON TABLE
# ============================================================

comparison_path = "analytics/final_model_comparison.csv"

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\nComparison saved to:")
print(comparison_path)


# ============================================================
# 10. GET BEST PIPELINE
# ============================================================

best_pipeline = fitted_pipelines[
    best_model_name
]


# ============================================================
# 11. SAVE COMPLETE FITTED PIPELINE
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "best_classifier_pipeline.joblib"
)

joblib.dump(
    best_pipeline,
    model_path
)

print("\nCOMPLETE PIPELINE SAVED:")
print(model_path)


# ============================================================
# 12. RELOAD PIPELINE
# ============================================================

loaded_pipeline = joblib.load(
    model_path
)

print("\nPIPELINE RELOADED SUCCESSFULLY.")


# ============================================================
# 13. RAW INPUT PREDICTION
# ============================================================

raw_input = pd.DataFrame([
    {
        "pclass": 3,
        "sex": "female",
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 15.0,
        "embarked": "S"
    }
])


prediction = loaded_pipeline.predict(
    raw_input
)

prediction_probability = loaded_pipeline.predict_proba(
    raw_input
)[:, 1]


print("\n")
print("=" * 70)
print("RAW INPUT PREDICTION TEST")
print("=" * 70)

print("\nRAW INPUT:")
print(raw_input.to_string(index=False))

print("\nPredicted class:", int(prediction[0]))
print(
    f"Survival probability: "
    f"{prediction_probability[0]:.4f}"
)

print(
    "\nSUCCESS: The saved pipeline accepts raw input "
    "without manually preprocessing the data."
)


# ============================================================
# 14. CREATE FINAL WRITTEN RECOMMENDATION
# ============================================================

recommendation = f"""
FINAL MODEL RECOMMENDATION
===========================

The three classification models were evaluated using the same
stratified train/test split.

Logistic Regression:
Accuracy = {comparison_df.loc[
    comparison_df["Model"] == "Logistic Regression",
    "Accuracy"
].iloc[0]:.4f}
Precision = {comparison_df.loc[
    comparison_df["Model"] == "Logistic Regression",
    "Precision"
].iloc[0]:.4f}
Recall = {comparison_df.loc[
    comparison_df["Model"] == "Logistic Regression",
    "Recall"
].iloc[0]:.4f}
F1 = {comparison_df.loc[
    comparison_df["Model"] == "Logistic Regression",
    "F1"
].iloc[0]:.4f}
AUC = {comparison_df.loc[
    comparison_df["Model"] == "Logistic Regression",
    "AUC"
].iloc[0]:.4f}

Decision Tree:
Accuracy = {comparison_df.loc[
    comparison_df["Model"] == "Decision Tree",
    "Accuracy"
].iloc[0]:.4f}
Precision = {comparison_df.loc[
    comparison_df["Model"] == "Decision Tree",
    "Precision"
].iloc[0]:.4f}
Recall = {comparison_df.loc[
    comparison_df["Model"] == "Decision Tree",
    "Recall"
].iloc[0]:.4f}
F1 = {comparison_df.loc[
    comparison_df["Model"] == "Decision Tree",
    "F1"
].iloc[0]:.4f}
AUC = {comparison_df.loc[
    comparison_df["Model"] == "Decision Tree",
    "AUC"
].iloc[0]:.4f}

Random Forest:
Accuracy = {comparison_df.loc[
    comparison_df["Model"] == "Random Forest",
    "Accuracy"
].iloc[0]:.4f}
Precision = {comparison_df.loc[
    comparison_df["Model"] == "Random Forest",
    "Precision"
].iloc[0]:.4f}
Recall = {comparison_df.loc[
    comparison_df["Model"] == "Random Forest",
    "Recall"
].iloc[0]:.4f}
F1 = {comparison_df.loc[
    comparison_df["Model"] == "Random Forest",
    "F1"
].iloc[0]:.4f}
AUC = {comparison_df.loc[
    comparison_df["Model"] == "Random Forest",
    "AUC"
].iloc[0]:.4f}

Based on the measured test-set F1 score, {best_model_name} has the
highest F1 score among the three models in this comparison.

Its measured test-set metrics are:
Accuracy = {best_accuracy:.4f}
Precision = {best_precision:.4f}
Recall = {best_recall:.4f}
F1 = {best_f1:.4f}
AUC = {best_auc:.4f}

Therefore, {best_model_name} is selected as the final classifier
for this experiment based on the observed F1 score. The complete
preprocessing and classifier pipeline has been saved as a single
Joblib file so that raw input can be passed directly to the model.
"""


recommendation_path = "analytics/final_model_recommendation.txt"

with open(
    recommendation_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(recommendation)


print("\nFINAL RECOMMENDATION SAVED:")
print(recommendation_path)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 70)
