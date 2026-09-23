# ============================================================
# MODULE 2 - ANALYTICS & MACHINE LEARNING
# Titanic Classification + Fare Regression
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATA_FILE = "analytics/titanic.csv"
PLOTS_DIR = "analytics/plots"
MODELS_DIR = "analytics/models"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# LOAD TITANIC DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("TITANIC DATASET")
print("=" * 70)

print("Dataset shape:", df.shape)
print()


# ============================================================
# BASIC INFORMATION
# ============================================================

print("Columns:")
print(df.columns.tolist())
print()

print("Missing values:")
print(df.isnull().sum())
print()


# ============================================================
# PART B - CLASSIFICATION
# ============================================================

print("=" * 70)
print("PART B - CLASSIFICATION")
print("=" * 70)


# ------------------------------------------------------------
# FEATURES AND TARGET
# ------------------------------------------------------------

classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[classification_features].copy()
y = df["survived"].copy()


# ------------------------------------------------------------
# STRATIFIED TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))
print()

print("Training class distribution:")
print(y_train.value_counts())
print()

print("Testing class distribution:")
print(y_test.value_counts())
print()


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

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

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# THREE CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


classification_results = []
fitted_pipelines = {}


# ------------------------------------------------------------
# TRAIN AND EVALUATE ALL THREE MODELS
# ------------------------------------------------------------

for model_name, model in models.items():

    print("-" * 70)
    print(model_name)
    print("-" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probability)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("AUC-ROC  :", round(auc, 4))
    print()

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC_ROC": auc
    })

    fitted_pipelines[model_name] = pipeline


# ------------------------------------------------------------
# SAVE CLASSIFIER COMPARISON
# ------------------------------------------------------------

classification_df = pd.DataFrame(classification_results)

classification_df.to_csv(
    "analytics/classification_results.csv",
    index=False
)

print("Classification results saved.")


# ============================================================
# CONFUSION MATRICES
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

for model_name, pipeline in fitted_pipelines.items():

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print()
    print(model_name)
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Did not survive", "Survived"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(
        f"{PLOTS_DIR}/confusion_matrix_{filename}.png"
    )

    plt.close()


print()
print("Confusion matrix plots saved.")


# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

print()
print("=" * 70)
print("DECISION TREE VISUALIZATION")
print("=" * 70)

decision_tree_pipeline = fitted_pipelines["Decision Tree"]

decision_tree_model = decision_tree_pipeline.named_steps["classifier"]

tree_preprocessor = decision_tree_pipeline.named_steps["preprocessor"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(24, 12))

plot_tree(
    decision_tree_model,
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True,
    rounded=True,
    fontsize=7
)

plt.title("Decision Tree - Titanic Survival Prediction")
plt.tight_layout()

plt.savefig(
    f"{PLOTS_DIR}/decision_tree.png",
    dpi=150
)

plt.close()

print("Decision tree plot saved.")


# ============================================================
# CLASS IMBALANCE COMPARISON
# ============================================================

print()
print("=" * 70)
print("CLASS IMBALANCE COMPARISON")
print("=" * 70)

imbalance_results = []


# ------------------------------------------------------------
# 1. BASELINE RANDOM FOREST
# ------------------------------------------------------------

baseline_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

baseline_pipeline.fit(X_train, y_train)

baseline_pred = baseline_pipeline.predict(X_test)

imbalance_results.append({
    "Method": "Baseline Random Forest",
    "Accuracy": accuracy_score(y_test, baseline_pred),
    "Precision": precision_score(y_test, baseline_pred),
    "Recall": recall_score(y_test, baseline_pred),
    "F1": f1_score(y_test, baseline_pred)
})


# ------------------------------------------------------------
# 2. CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

balanced_pipeline.fit(X_train, y_train)

balanced_pred = balanced_pipeline.predict(X_test)

imbalance_results.append({
    "Method": "Random Forest class_weight=balanced",
    "Accuracy": accuracy_score(y_test, balanced_pred),
    "Precision": precision_score(y_test, balanced_pred),
    "Recall": recall_score(y_test, balanced_pred),
    "F1": f1_score(y_test, balanced_pred)
})


# ------------------------------------------------------------
# 3. SMOTE
# ------------------------------------------------------------

smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

smote_pipeline.fit(X_train, y_train)

smote_pred = smote_pipeline.predict(X_test)

imbalance_results.append({
    "Method": "Random Forest + SMOTE",
    "Accuracy": accuracy_score(y_test, smote_pred),
    "Precision": precision_score(y_test, smote_pred),
    "Recall": recall_score(y_test, smote_pred),
    "F1": f1_score(y_test, smote_pred)
})


imbalance_df = pd.DataFrame(imbalance_results)

print(imbalance_df)

imbalance_df.to_csv(
    "analytics/imbalance_comparison.csv",
    index=False
)

print()
print("Imbalance comparison saved.")


# ============================================================
# GRID SEARCH - RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("GRID SEARCH - RANDOM FOREST")
print("=" * 70)

grid_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)


param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"]
}


grid_search = GridSearchCV(
    estimator=grid_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


print("Best parameters:")
print(grid_search.best_params_)

print()
print("Best cross-validation F1:")
print(round(grid_search.best_score_, 4))


# ------------------------------------------------------------
# OOB SCORE
# ------------------------------------------------------------

best_params = grid_search.best_params_

best_rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=best_params["classifier__n_estimators"],
                max_depth=best_params["classifier__max_depth"],
                max_features=best_params["classifier__max_features"],
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

best_rf_pipeline.fit(X_train, y_train)

best_rf_model = best_rf_pipeline.named_steps["classifier"]

print()
print("OOB Score:")
print(round(best_rf_model.oob_score_, 4))


# ============================================================
# TUNED RANDOM FOREST TEST METRICS
# ============================================================

tuned_pred = best_rf_pipeline.predict(X_test)

tuned_probability = best_rf_pipeline.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(y_test, tuned_pred)
tuned_precision = precision_score(y_test, tuned_pred)
tuned_recall = recall_score(y_test, tuned_pred)
tuned_f1 = f1_score(y_test, tuned_pred)
tuned_auc = roc_auc_score(y_test, tuned_probability)


print()
print("Tuned Random Forest test results:")
print("Accuracy :", round(tuned_accuracy, 4))
print("Precision:", round(tuned_precision, 4))
print("Recall   :", round(tuned_recall, 4))
print("F1 Score :", round(tuned_f1, 4))
print("AUC-ROC  :", round(tuned_auc, 4))


# ============================================================
# FINAL CLASSIFIER COMPARISON
# ============================================================

final_classifier_results = classification_df.copy()

final_classifier_results = final_classifier_results[
    final_classifier_results["Model"] != "Random Forest"
]

final_classifier_results = pd.concat(
    [
        final_classifier_results,
        pd.DataFrame([
            {
                "Model": "Random Forest (Tuned)",
                "Accuracy": tuned_accuracy,
                "Precision": tuned_precision,
                "Recall": tuned_recall,
                "F1": tuned_f1,
                "AUC_ROC": tuned_auc
            }
        ])
    ],
    ignore_index=True
)

final_classifier_results.to_csv(
    "analytics/final_classifier_comparison.csv",
    index=False
)

print()
print("=" * 70)
print("FINAL CLASSIFIER COMPARISON")
print("=" * 70)

print(final_classifier_results)


# ============================================================
# SAVE BEST CLASSIFIER PIPELINE
# ============================================================

MODEL_FILE = f"{MODELS_DIR}/best_classifier_pipeline.joblib"

joblib.dump(
    best_rf_pipeline,
    MODEL_FILE
)

print()
print("Best classifier pipeline saved to:")
print(MODEL_FILE)


# ============================================================
# RELOAD MODEL AND PREDICT RAW INPUT
# ============================================================

print()
print("=" * 70)
print("RELOAD SAVED MODEL - RAW INPUT PREDICTION")
print("=" * 70)

loaded_model = joblib.load(MODEL_FILE)


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


prediction = loaded_model.predict(raw_input)

probability = loaded_model.predict_proba(raw_input)[0][1]

print("Raw input:")
print(raw_input)
print()

print("Predicted survival class:", int(prediction[0]))

print(
    "Survival probability:",
    round(probability, 4)
)


# ============================================================
# PART C - REGRESSION
# ============================================================

print()
print("=" * 70)
print("PART C - FARE REGRESSION")
print("=" * 70)


regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

regression_target = "fare"

X_reg = df[regression_features].copy()
y_reg = df[regression_target].copy()


# ------------------------------------------------------------
# REGRESSION TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# REGRESSION PREPROCESSING
# ------------------------------------------------------------

reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]


reg_numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            reg_numeric_transformer,
            reg_numeric_features
        ),
        (
            "cat",
            reg_categorical_transformer,
            reg_categorical_features
        )
    ]
)


# ------------------------------------------------------------
# LINEAR REGRESSION
# ------------------------------------------------------------

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("regressor", LinearRegression())
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


# ------------------------------------------------------------
# REGRESSION METRICS
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)


reg_feature_names = (
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

p = len(reg_feature_names)
n = len(y_reg_test)


adjusted_r2 = (
    1
    - ((1 - r2) * (n - 1) / (n - p - 1))
)


print()
print("Regression metrics:")
print("MAE         :", round(mae, 4))
print("RMSE        :", round(rmse, 4))
print("R2          :", round(r2, 4))
print("Adjusted R2 :", round(adjusted_r2, 4))


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

residuals = y_reg_test - y_reg_pred

residual_mean = residuals.mean()
residual_std = residuals.std()

print()
print("Residual mean:", round(residual_mean, 4))
print("Residual std :", round(residual_std, 4))


absolute_residuals = np.abs(residuals)

correlation = np.corrcoef(
    y_reg_pred,
    absolute_residuals
)[0, 1]

print()
print(
    "Correlation between predicted fare and absolute residual:",
    round(correlation, 4)
)


if abs(correlation) > 0.30:
    hetero_conclusion = (
        "The residual spread changes with predicted fare, "
        "suggesting possible heteroscedasticity."
    )
else:
    hetero_conclusion = (
        "The residual spread does not show a strong systematic "
        "change with predicted fare."
    )

print()
print("Heteroscedasticity conclusion:")
print(hetero_conclusion)


# ------------------------------------------------------------
# RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")

plt.tight_layout()

plt.savefig(
    f"{PLOTS_DIR}/fare_regression_residuals.png",
    dpi=150
)

plt.close()

print()
print("Residual plot saved.")


# ============================================================
# SAVE REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame([
    {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }
])

regression_results.to_csv(
    "analytics/regression_results.csv",
    index=False
)


# ============================================================
# SAVE REGRESSION TEXT REPORT
# ============================================================

with open(
    "analytics/regression_results.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("TITANIC FARE REGRESSION RESULTS\n")
    file.write("=" * 50 + "\n")

    file.write(f"MAE: {mae:.4f}\n")
    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"R2: {r2:.4f}\n")
    file.write(f"Adjusted R2: {adjusted_r2:.4f}\n")
    file.write(f"Predictor count: {p}\n")
    file.write(f"Test rows: {n}\n")
    file.write("\n")
    file.write("Heteroscedasticity conclusion:\n")
    file.write(hetero_conclusion + "\n")


# ============================================================
# FINAL MODELING SUMMARY
# ============================================================

recommendation = f"""
FINAL MODELING SUMMARY
======================

Three classification models were evaluated using the same stratified
80/20 train-test split: Logistic Regression, Decision Tree, and
Random Forest.

GridSearchCV was used to tune the Random Forest using 5-fold
cross-validation and F1 score. The selected Random Forest parameters
were {best_params}, with a cross-validation F1 score of
{grid_search.best_score_:.4f} and an OOB score of
{best_rf_model.oob_score_:.4f}.

The tuned Random Forest was then evaluated on the held-out test set,
achieving accuracy={tuned_accuracy:.4f}, precision={tuned_precision:.4f},
recall={tuned_recall:.4f}, F1={tuned_f1:.4f}, and
AUC-ROC={tuned_auc:.4f}.

For class imbalance, baseline Random Forest, class_weight='balanced',
and SMOTE were compared using the same test set. SMOTE was applied
only to the training data through an imbalanced-learn pipeline.

For fare regression, Linear Regression achieved MAE={mae:.4f},
RMSE={rmse:.4f}, R2={r2:.4f}, and Adjusted R2={adjusted_r2:.4f}.
The residual analysis was used to assess whether the error spread
changes with predicted fare.
"""

with open(
    "analytics/final_modeling_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(recommendation)


print()
print("=" * 70)
print("ALL MODELING TASKS COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("Generated files:")
print("- analytics/classification_results.csv")
print("- analytics/final_classifier_comparison.csv")
print("- analytics/imbalance_comparison.csv")
print("- analytics/regression_results.csv")
print("- analytics/regression_results.txt")
print("- analytics/final_modeling_summary.txt")
print("- analytics/models/best_classifier_pipeline.joblib")
print("- analytics/plots/confusion_matrix_logistic_regression.png")
print("- analytics/plots/confusion_matrix_decision_tree.png")
print("- analytics/plots/confusion_matrix_random_forest.png")
print("- analytics/plots/decision_tree.png")
print("- analytics/plots/fare_regression_residuals.png")