import sqlite3
import pandas as pd
from pathlib import Path

# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "books_cleaned.csv"
database_file = DATA_DIR / "books.db"


# =========================================================
# READ CLEANED DATA
# =========================================================

print("Reading cleaned data...")

df = pd.read_csv(input_file, encoding="utf-8-sig")

print(f"Books loaded from CSV: {len(df)}")


# =========================================================
# CREATE SQLITE DATABASE
# =========================================================

connection = sqlite3.connect(database_file)

# Enable foreign key support
connection.execute("PRAGMA foreign_keys = ON")


# =========================================================
# CREATE TABLES
# =========================================================

cursor = connection.cursor()

# Remove old tables if they exist
cursor.execute("DROP TABLE IF EXISTS books")
cursor.execute("DROP TABLE IF EXISTS categories")


# ---------------------------------------------------------
# Categories table
# ---------------------------------------------------------

cursor.execute("""
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
)
""")


# ---------------------------------------------------------
# Books table
# ---------------------------------------------------------

cursor.execute("""
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id)
        REFERENCES categories(category_id)
)
""")


# =========================================================
# INSERT CATEGORIES
# =========================================================

categories = sorted(df["category"].dropna().unique())

for category in categories:
    cursor.execute(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        (category,)
    )


# Create category → ID mapping
cursor.execute("SELECT category_id, category_name FROM categories")

category_map = {
    category_name: category_id
    for category_id, category_name in cursor.fetchall()
}


# =========================================================
# INSERT BOOKS
# =========================================================

for _, row in df.iterrows():

    cursor.execute(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(row["in_stock"]),
            category_map[row["category"]]
        )
    )


# Save changes
connection.commit()


# =========================================================
# VERIFY DATABASE
# =========================================================

print("\n========== DATABASE CREATED ==========")

# Count categories
cursor.execute("SELECT COUNT(*) FROM categories")
category_count = cursor.fetchone()[0]

# Count books
cursor.execute("SELECT COUNT(*) FROM books")
book_count = cursor.fetchone()[0]

print(f"Categories inserted: {category_count}")
print(f"Books inserted: {book_count}")


# =========================================================
# DISPLAY CATEGORIES
# =========================================================

print("\n========== CATEGORIES ==========")

cursor.execute("""
SELECT category_id, category_name
FROM categories
ORDER BY category_id
""")

for row in cursor.fetchall():
    print(row)


# =========================================================
# DISPLAY FIRST 5 BOOKS
# =========================================================

print("\n========== FIRST 5 BOOKS ==========")

cursor.execute("""
SELECT
    books.book_id,
    books.title,
    books.price_gbp,
    books.price_inr,
    books.rating,
    books.in_stock,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
ORDER BY books.book_id
LIMIT 5
""")

for row in cursor.fetchall():
    print(row)


# =========================================================
# CLOSE DATABASE
# =========================================================

connection.close()

print("\n====================================")
print("DATABASE LOADING COMPLETED")
print("====================================")
print(f"Database saved to: {database_file}")
