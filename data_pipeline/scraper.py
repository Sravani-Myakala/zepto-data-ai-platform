import requests
from bs4 import BeautifulSoup
import csv
from pathlib import Path
import time
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"

# 3+ categories
CATEGORIES = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Science": "catalogue/category/books/science_22/index.html",
    "Fantasy": "catalogue/category/books/fantasy_19/index.html"
}

# Create data folder inside data_pipeline
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

books = []

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Scrape all available books from the selected categories
for category, category_url in CATEGORIES.items():

    print(f"\nScraping category: {category}")

    url = urljoin(BASE_URL, category_url)
    category_count = 0

    while url:

        print(f"Scraping: {url}")

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()

        except requests.RequestException as e:
            print(f"Failed to access page: {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        book_items = soup.select("article.product_pod")

        if not book_items:
            print("No books found on this page.")
            break

        for book in book_items:

            title_element = book.select_one("h3 a")
            price_element = book.select_one(".price_color")
            rating_element = book.select_one("p.star-rating")
            availability_element = book.select_one(".availability")

            title = (
                title_element.get("title", "").strip()
                if title_element
                else "Unknown"
            )

            price = (
                price_element.get_text(strip=True)
                if price_element
                else "Unknown"
            )

            if rating_element:
                rating_classes = rating_element.get("class", [])

                if len(rating_classes) > 1:
                    rating = rating_classes[1]
                else:
                    rating = "Unknown"
            else:
                rating = "Unknown"

            availability = (
                availability_element.get_text(" ", strip=True)
                if availability_element
                else "Unknown"
            )

            books.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": category
            })

            category_count += 1

        # Find next page
        next_button = soup.select_one("li.next a")

        if next_button:
            next_page = next_button.get("href")
            url = urljoin(url, next_page)
            time.sleep(1)
        else:
            url = None

    print(f"{category_count} books scraped from {category}")

# Remove duplicate books
unique_books = []
seen_titles = set()

for book in books:

    if book["title"] not in seen_titles:
        unique_books.append(book)
        seen_titles.add(book["title"])

books = unique_books

# Save CSV
file_path = DATA_DIR / "books_raw.csv"

with open(
    file_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "title",
        "price",
        "star_rating",
        "availability",
        "category"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(books)

print("\n--------------------------------")
print("SCRAPING COMPLETED")
print("--------------------------------")
print(f"Total unique books scraped: {len(books)}")
print(f"Data saved to: {file_path}")

# Check requirement
if len(books) >= 60:
    print("SUCCESS: 60+ books collected!")
else:
    print("WARNING: Less than 60 books collected.")