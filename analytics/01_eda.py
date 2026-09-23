import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================================
# MODULE 2 - PART A: EXPLORATORY DATA ANALYSIS
# ============================================================

print("=" * 70)
print("MODULE 2 - PART A: EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD TITANIC DATASET ONCE
# ------------------------------------------------------------

df = sns.load_dataset("titanic")

print("\nDATASET LOADED SUCCESSFULLY")

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- DATASET INFO ---")
df.info()

print("\n--- DATASET DESCRIPTION ---")
print(df.describe(include="all"))


# ------------------------------------------------------------
# SAVE RAW DATASET IMMEDIATELY
# ------------------------------------------------------------

os.makedirs("analytics/plots", exist_ok=True)

df.to_csv("analytics/titanic.csv", index=False)

print("\nRaw dataset saved as:")
print("analytics/titanic.csv")


# ------------------------------------------------------------
# 2. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage.round(2)
})

print(missing_table[missing_table["Missing Count"] > 0])


# ------------------------------------------------------------
# 3. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)

cleaned_df = df.copy()

# AGE: 5%-30% missing -> median imputation
age_missing = cleaned_df["age"].isnull().mean() * 100

print(f"\nAge missing percentage: {age_missing:.2f}%")

if 5 <= age_missing <= 30:
    age_median = cleaned_df["age"].median()
    cleaned_df["age"] = cleaned_df["age"].fillna(age_median)
    print(f"Age median imputation applied: {age_median:.2f}")


# EMBARKED: <5% missing -> drop rows
embarked_missing = cleaned_df["embarked"].isnull().mean() * 100

print(f"Embarked missing percentage: {embarked_missing:.2f}%")

if embarked_missing < 5:
    cleaned_df = cleaned_df.dropna(subset=["embarked"])
    print("Rows with missing Embarked values were dropped.")


# DECK: >30% missing -> drop column
deck_missing = df["deck"].isnull().mean() * 100

print(f"Deck missing percentage: {deck_missing:.2f}%")

if deck_missing > 30:
    cleaned_df = cleaned_df.drop(columns=["deck"])
    print("Deck column dropped because missing percentage is very high.")


print("\nCleaned dataset shape:")
print(cleaned_df.shape)

print("\nRemaining missing values:")
print(cleaned_df.isnull().sum())


# ------------------------------------------------------------
# 4. AGE UNIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AGE ANALYSIS")
print("=" * 70)

age_q1 = cleaned_df["age"].quantile(0.25)
age_q3 = cleaned_df["age"].quantile(0.75)
age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = (
    (cleaned_df["age"] < age_lower) |
    (cleaned_df["age"] > age_upper)
).sum()

print(f"Age Q1: {age_q1:.2f}")
print(f"Age Q3: {age_q3:.2f}")
print(f"Age IQR: {age_iqr:.2f}")
print(f"Age lower bound: {age_lower:.2f}")
print(f"Age upper bound: {age_upper:.2f}")
print(f"Age outlier count: {age_outliers}")


# Age histogram
plt.figure(figsize=(8, 5))
plt.hist(cleaned_df["age"], bins=20)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analytics/plots/age_histogram.png")
plt.show()


# Age box plot
plt.figure(figsize=(8, 5))
plt.boxplot(cleaned_df["age"])
plt.title("Box Plot of Age")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("analytics/plots/age_boxplot.png")
plt.show()


# ------------------------------------------------------------
# 5. FARE UNIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FARE ANALYSIS")
print("=" * 70)

fare_q1 = cleaned_df["fare"].quantile(0.25)
fare_q3 = cleaned_df["fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = (
    (cleaned_df["fare"] < fare_lower) |
    (cleaned_df["fare"] > fare_upper)
).sum()

fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode().iloc[0]

print(f"Fare mean: {fare_mean:.2f}")
print(f"Fare median: {fare_median:.2f}")
print(f"Fare mode: {fare_mode:.2f}")

print(f"\nFare Q1: {fare_q1:.2f}")
print(f"Fare Q3: {fare_q3:.2f}")
print(f"Fare IQR: {fare_iqr:.2f}")
print(f"Fare lower bound: {fare_lower:.2f}")
print(f"Fare upper bound: {fare_upper:.2f}")
print(f"Fare outlier count: {fare_outliers}")


# Fare histogram
plt.figure(figsize=(8, 5))
plt.hist(cleaned_df["fare"], bins=30)
plt.title("Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("analytics/plots/fare_histogram.png")
plt.show()


# Fare box plot
plt.figure(figsize=(8, 5))
plt.boxplot(cleaned_df["fare"])
plt.title("Box Plot of Fare")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("analytics/plots/fare_boxplot.png")
plt.show()


# ------------------------------------------------------------
# 6. FARE SKEWNESS
# ------------------------------------------------------------

print("\n--- FARE SKEWNESS ---")

if fare_mean > fare_median > fare_mode:
    print("Fare is positively/right skewed.")
    print("Reason: Mean > Median > Mode.")
elif fare_mean < fare_median < fare_mode:
    print("Fare is negatively/left skewed.")
    print("Reason: Mean < Median < Mode.")
else:
    print("Fare does not follow a simple Mean-Median-Mode ordering.")


# ------------------------------------------------------------
# 7. SURVIVAL RATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SURVIVAL RATE ANALYSIS")
print("=" * 70)


# By sex
female_rate = cleaned_df.loc[
    cleaned_df["sex"] == "female", "survived"
].mean()

male_rate = cleaned_df.loc[
    cleaned_df["sex"] == "male", "survived"
].mean()

print(f"\nFemale survival rate: {female_rate:.2%}")
print(f"Male survival rate: {male_rate:.2%}")


# By passenger class
print("\nSurvival rate by passenger class:")

for pclass in sorted(cleaned_df["pclass"].unique()):
    rate = cleaned_df.loc[
        cleaned_df["pclass"] == pclass, "survived"
    ].mean()

    print(f"Class {pclass}: {rate:.2%}")


# By sex + passenger class
print("\nSurvival rate by sex and passenger class:")

for sex in ["female", "male"]:
    for pclass in sorted(cleaned_df["pclass"].unique()):

        mask = (
            (cleaned_df["sex"] == sex) &
            (cleaned_df["pclass"] == pclass)
        )

        rate = cleaned_df.loc[mask, "survived"].mean()

        print(f"{sex}, Class {pclass}: {rate:.2%}")


# ------------------------------------------------------------
# 8. REQUIRED CORRELATION MATRIX
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = cleaned_df[correlation_columns].corr()

print(correlation_matrix.round(3))


# Heatmap
plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Titanic Correlation Heatmap")
plt.tight_layout()
plt.savefig("analytics/plots/correlation_heatmap.png")
plt.show()


# ------------------------------------------------------------
# TWO STRONGEST CORRELATIONS
# ------------------------------------------------------------

pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):

        column1 = correlation_columns[i]
        column2 = correlation_columns[j]

        value = correlation_matrix.loc[column1, column2]

        pairs.append(
            (column1, column2, value, abs(value))
        )

pairs.sort(key=lambda x: x[3], reverse=True)

print("\nTwo strongest correlations:")

for pair in pairs[:2]:
    print(
        f"{pair[0]} <-> {pair[1]} = {pair[2]:.3f}"
    )


# ------------------------------------------------------------
# 9. MULTIVARIATE DATA STORY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MULTIVARIATE DATA STORY")
print("=" * 70)


# Chart 1
plt.figure(figsize=(8, 5))

sns.barplot(
    data=cleaned_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("analytics/plots/survival_by_sex_class.png")
plt.show()


# Chart 2
plt.figure(figsize=(8, 5))

sns.histplot(
    data=cleaned_df,
    x="age",
    hue="survived",
    bins=20,
    kde=True
)

plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("analytics/plots/age_by_survival.png")
plt.show()


# Chart 3
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=cleaned_df,
    x="pclass",
    y="fare",
    hue="survived"
)

plt.title("Fare by Passenger Class and Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.savefig("analytics/plots/fare_class_survival.png")
plt.show()


# Chart 4
cleaned_df["family_size"] = (
    cleaned_df["sibsp"] +
    cleaned_df["parch"] +
    1
)

family_survival = (
    cleaned_df
    .groupby("family_size")["survived"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))

sns.lineplot(
    data=family_survival,
    x="family_size",
    y="survived",
    marker="o"
)

plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("analytics/plots/family_size_survival.png")
plt.show()


# ------------------------------------------------------------
# 10. EDA-ONLY STANDARDIZATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA-ONLY STANDARDIZATION")
print("=" * 70)

standardized_df = cleaned_df[["age", "fare"]].copy()

scaler = StandardScaler()

standardized_df[["age", "fare"]] = scaler.fit_transform(
    standardized_df[["age", "fare"]]
)

print("\nBefore standardization:")
print(
    cleaned_df[["age", "fare"]]
    .agg(["mean", "std"])
)

print("\nAfter standardization:")
print(
    standardized_df[["age", "fare"]]
    .agg(["mean", "std"])
)

print("\nStandardization completed for EDA only.")
print("These values will NOT be used for the ML model.")


# ------------------------------------------------------------
# FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODULE 2 PART A COMPLETED")
print("=" * 70)

print("\nCreated files:")
print("analytics/titanic.csv")
print("analytics/plots/age_histogram.png")
print("analytics/plots/age_boxplot.png")
print("analytics/plots/fare_histogram.png")
print("analytics/plots/fare_boxplot.png")
print("analytics/plots/correlation_heatmap.png")
print("analytics/plots/survival_by_sex_class.png")
print("analytics/plots/age_by_survival.png")
print("analytics/plots/fare_class_survival.png")
print("analytics/plots/family_size_survival.png")