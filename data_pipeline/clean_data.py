import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "books_raw.csv"
output_file = DATA_DIR / "books_cleaned.csv"

# Fixed project conversion rate
GBP_TO_INR = 105.50

print("Reading raw data...")

# Read CSV
df = pd.read_csv(input_file, encoding="utf-8")

print("\n========== RAW PRICE SAMPLE ==========")
print(df["price"].head())

# Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicate rows removed: {before - len(df)}")


# =========================================================
# CLEAN PRICE
# =========================================================

# Convert values such as:
# Â£45.17
# £45.17
# £49.43
# into numeric values

df["price_gbp"] = (
    df["price"]
    .astype(str)
    .str.replace("Â", "", regex=False)
    .str.replace("£", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

# Convert to numeric
df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

# Median imputation for invalid/missing prices
price_median = df["price_gbp"].median()

df["price_gbp"] = df["price_gbp"].fillna(price_median)


# =========================================================
# CLEAN STAR RATING
# =========================================================

rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)

# Median imputation if an unexpected rating occurs
rating_median = df["rating"].median()

df["rating"] = df["rating"].fillna(rating_median)

df["rating"] = df["rating"].round().astype(int)


# =========================================================
# CLEAN AVAILABILITY
# =========================================================

df["availability"] = (
    df["availability"]
    .astype(str)
    .str.strip()
)

df["in_stock"] = (
    df["availability"]
    .str.lower()
    .str.contains("in stock", na=False)
)


# =========================================================
# CLEAN TEXT
# =========================================================

df["title"] = df["title"].astype(str).str.strip()

df["category"] = df["category"].astype(str).str.strip()

# Remove rows where essential fields are empty
df = df[
    (df["title"] != "") &
    (df["category"] != "")
]


# =========================================================
# GBP → INR
# =========================================================

df["price_inr"] = df["price_gbp"] * GBP_TO_INR


# =========================================================
# FINAL COLUMN ORDER
# =========================================================

df = df[
    [
        "title",
        "price_gbp",
        "star_rating",
        "rating",
        "availability",
        "in_stock",
        "category",
        "price_inr"
    ]
]

df = df.reset_index(drop=True)


# =========================================================
# SAVE CLEANED DATA
# =========================================================

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n========== CLEANED DATA ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== CATEGORY COUNTS ==========")
print(df["category"].value_counts())

print("\n========== PRICE CHECK ==========")
print(df[["price_gbp", "price_inr"]].head())

print("\n========== FINAL RESULT ==========")
print(f"Total books: {len(df)}")
print(f"Total categories: {df['category'].nunique()}")

print("\n========== REQUIRED COLUMNS ==========")
print(df.columns.tolist())

print(f"\nCleaned data saved to: {output_file}")

print("\nFixed conversion rate used:")
print("1 GBP = 105.50 INR")