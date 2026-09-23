import sqlite3
import pandas as pd

# =========================================================
# DATABASE CONNECTION
# =========================================================

database_file = "data/books.db"

conn = sqlite3.connect(database_file)

print("=" * 70)
print("PANDAS SQL ANALYSIS")
print("=" * 70)


# =========================================================
# QUERY 1 - pd.read_sql()
# =========================================================

query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC, title ASC;
"""

df_high_rated = pd.read_sql(query1, conn)

print("\n========== RESULT 1: pd.read_sql() ==========")
print(df_high_rated.head(10))


# =========================================================
# QUERY 2 - pd.read_sql()
# =========================================================

query2 = """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
"""

df_expensive = pd.read_sql(query2, conn)

print("\n========== RESULT 2: pd.read_sql() ==========")
print(df_expensive)


# =========================================================
# LOAD TABLES INTO PANDAS
# =========================================================

books_df = pd.read_sql(
    "SELECT * FROM books",
    conn
)

categories_df = pd.read_sql(
    "SELECT * FROM categories",
    conn
)

print("\n========== BOOKS DATAFRAME ==========")
print(books_df.head())

print("\n========== CATEGORIES DATAFRAME ==========")
print(categories_df)


# =========================================================
# SQL JOIN
# =========================================================

sql_join_query = """
SELECT
    books.title,
    books.price_gbp,
    books.rating,
    categories.category_name
FROM books
JOIN categories
    ON books.category_id = categories.category_id
ORDER BY books.rating DESC, books.title ASC
LIMIT 10;
"""

sql_join_df = pd.read_sql(
    sql_join_query,
    conn
)

print("\n========== SQL JOIN RESULT ==========")
print(sql_join_df)


# =========================================================
# PANDAS JOIN USING pd.merge()
# =========================================================

pandas_join_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

# Select exactly the same columns
pandas_join_df = pandas_join_df[
    [
        "title",
        "price_gbp",
        "rating",
        "category_name"
    ]
]

# Apply SAME sorting as SQL
pandas_join_df = (
    pandas_join_df
    .sort_values(
        by=["rating", "title"],
        ascending=[False, True]
    )
    .head(10)
    .reset_index(drop=True)
)

print("\n========== PANDAS pd.merge() RESULT ==========")
print(pandas_join_df)


# =========================================================
# COMPARE RESULTS
# =========================================================

sql_join_df = sql_join_df.reset_index(drop=True)

sql_join_df["price_gbp"] = sql_join_df["price_gbp"].round(2)
pandas_join_df["price_gbp"] = pandas_join_df["price_gbp"].round(2)

match = sql_join_df.equals(pandas_join_df)

print("\n========== JOIN COMPARISON ==========")

print(f"SQL JOIN rows: {len(sql_join_df)}")
print(f"pd.merge() rows: {len(pandas_join_df)}")

print(
    f"\nDo the SQL JOIN and pd.merge() results match? {match}"
)

if match:
    print(
        "SUCCESS: SQL JOIN and pandas merge produce equivalent results."
    )
else:
    print(
        "WARNING: Results do not match. Check columns and sorting."
    )


# =========================================================
# SAVE OUTPUT
# =========================================================

output_file = "data/pandas_outputs.txt"

with open(output_file, "w", encoding="utf-8") as file:

    file.write("PANDAS SQL ANALYSIS\n")
    file.write("=" * 70 + "\n\n")

    file.write("RESULT 1 - pd.read_sql()\n")
    file.write(str(df_high_rated))
    file.write("\n\n")

    file.write("RESULT 2 - pd.read_sql()\n")
    file.write(str(df_expensive))
    file.write("\n\n")

    file.write("SQL JOIN RESULT\n")
    file.write(str(sql_join_df))
    file.write("\n\n")

    file.write("PANDAS pd.merge() RESULT\n")
    file.write(str(pandas_join_df))
    file.write("\n\n")

    file.write(
        f"SQL JOIN and pd.merge() match: {match}\n"
    )

conn.close()

print(f"\nPandas outputs saved to: {output_file}")