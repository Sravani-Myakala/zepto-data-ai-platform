import sqlite3
from pathlib import Path

# =========================================================
# FILE PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

database_file = DATA_DIR / "books.db"
output_file = DATA_DIR / "query_outputs.txt"


# =========================================================
# CONNECT TO DATABASE
# =========================================================

connection = sqlite3.connect(database_file)


# =========================================================
# SQL QUERIES
# =========================================================

queries = {

    "QUERY 1 - SELECT + WHERE": """
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4;
""",

    "QUERY 2 - ORDER BY + LIMIT": """
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;
""",

    "QUERY 3 - DISTINCT": """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
""",

    "QUERY 4 - IN": """
SELECT title, category_id
FROM books
WHERE category_id IN (1, 2);
""",

    "QUERY 5 - BETWEEN": """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 30
ORDER BY price_gbp;
""",

    "QUERY 6 - JOIN": """
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
}


# =========================================================
# EXECUTE QUERIES
# =========================================================

all_output = []

for query_name, query in queries.items():

    print("\n" + "=" * 70)
    print(query_name)
    print("=" * 70)

    print("\nSQL:")
    print(query.strip())

    cursor = connection.execute(query)
    rows = cursor.fetchall()

    print("\nOUTPUT:")

    for row in rows:
        print(row)

    all_output.append("=" * 70)
    all_output.append(query_name)
    all_output.append("=" * 70)
    all_output.append("\nSQL:")
    all_output.append(query.strip())
    all_output.append("\nOUTPUT:")

    for row in rows:
        all_output.append(str(row))

    all_output.append("")


# =========================================================
# SAVE OUTPUT
# =========================================================

with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(all_output))


# =========================================================
# CLOSE DATABASE
# =========================================================

connection.close()


print("\n" + "=" * 70)
print("ALL SQL QUERIES COMPLETED")
print("=" * 70)
print(f"{len(queries)} SQL queries executed successfully.")
print(f"Query outputs saved to: {output_file}")