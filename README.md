\# Zepto Data \& AI Platform



A complete data and AI project containing three connected modules:



1\. \*\*Module 1 – Data Pipeline\*\*

2\. \*\*Module 2 – Analytics \& Machine Learning\*\*

3\. \*\*Module 3 – GenAI Support Assistant\*\*



The project demonstrates web scraping, data cleaning, relational database design, SQL analysis, Pandas analysis, exploratory data analysis, machine learning, RAG-based question answering, LangGraph workflows, FastAPI, and Docker.



\---



\# Project Structure



```text

zepto-data-ai-platform/

│

├── data\_pipeline/

│   ├── scraper.py

│   ├── clean\_data.py

│   ├── database.py

│   ├── queries.py

│   ├── pandas\_analysis.py

│   └── data/

│       ├── books\_raw.csv

│       ├── books\_cleaned.csv

│       ├── books.db

│       ├── query\_outputs.txt

│       └── pandas\_outputs.txt

│

├── analytics/

│   ├── 01\_eda.py

│   ├── 02\_modeling.py

│   ├── titanic.csv

│   ├── README.md

│   ├── models/

│   └── plots/

│

├── support\_assistant/

│   ├── assistant.py

│   ├── ingest.py

│   ├── main.py

│   ├── Dockerfile

│   ├── requirements.txt

│   ├── docs/

│   │   ├── doc\_01.txt

│   │   ├── doc\_02.txt

│   │   ├── doc\_03.txt

│   │   ├── doc\_04.txt

│   │   ├── doc\_05.txt

│   │   ├── doc\_06.txt

│   │   ├── doc\_07.txt

│   │   └── doc\_08.txt

│   └── chroma\_db/

│

├── .gitignore

└── README.md

```



\---



\# Requirements



\* Python 3.12

\* pip

\* Git

\* Docker Desktop for the containerized Support Assistant

\* Internet connection for the initial book scraping and embedding-model download



The project uses free/open-source Python libraries. No paid API is required for the default Support Assistant because `MOCK\_LLM=1` is used by default.



\---



\# Setup



Open PowerShell in the project root:



```powershell

cd "C:\\Users\\myaka\\OneDrive\\Desktop\\zepto-data-ai-platform"

```



Create and activate the virtual environment:



```powershell

python -m venv venv

```



```powershell

.\\venv\\Scripts\\Activate.ps1

```



Install the required packages:



```powershell

pip install requests beautifulsoup4 pandas

```



For the Analytics module:



```powershell

pip install seaborn matplotlib scikit-learn imbalanced-learn joblib

```



For the Support Assistant:



```powershell

pip install fastapi uvicorn pydantic chromadb sentence-transformers langgraph huggingface\_hub

```



\---



\# Module 1 – Data Pipeline



\## Objective



The data pipeline collects book catalog information from the public scraping-practice website:



```text

https://books.toscrape.com/

```



The pipeline performs:



```text

Web Scraping

&#x20;    ↓

Raw CSV

&#x20;    ↓

Data Cleaning

&#x20;    ↓

Price Conversion

&#x20;    ↓

SQLite Database

&#x20;    ↓

SQL Queries

&#x20;    ↓

Pandas Analysis

```



\## Scraping



The scraper uses:



\* `requests`

\* `BeautifulSoup`



The dataset contains \*\*104 books\*\* across four categories:



\* Travel

\* Mystery

\* Science

\* Fantasy



The required fields include:



\* title

\* price in GBP

\* star rating text

\* availability text

\* category



The raw dataset is saved as:



```text

data\_pipeline/data/books\_raw.csv

```



\## Cleaning



The cleaned dataset is saved as:



```text

data\_pipeline/data/books\_cleaned.csv

```



The cleaned data contains:



\* `price\_gbp` as `float`

\* `rating` as `int`

\* `in\_stock` as `bool`

\* `category` as string

\* `price\_inr` as `float`



The final cleaned dataset contains:



```text

104 rows

4 categories

```



The fixed baseline currency conversion required by the project is:



```text

1 GBP = ₹105.50

```



Therefore:



```text

price\_inr = price\_gbp × 105.50

```



The parsing and cleaning logic is implemented in:



```text

data\_pipeline/clean\_data.py

```



\## Database



The cleaned data is stored in SQLite:



```text

data\_pipeline/data/books.db

```



The database uses a normalized relational structure with separate book and category information connected using a primary key and foreign key.



Database creation is implemented in:



```text

data\_pipeline/database.py

```



\## SQL Analysis



The SQL queries are implemented in:



```text

data\_pipeline/queries.py

```



The project contains queries demonstrating:



\* `SELECT`

\* `WHERE`

\* `ORDER BY`

\* `LIMIT`

\* `DISTINCT`

\* `IN`

\* `BETWEEN`

\* `JOIN`



Six SQL queries were executed and their outputs were saved in:



```text

data\_pipeline/data/query\_outputs.txt

```



\## Pandas Reproduction



SQL results were also read using:



```python

pd.read\_sql()

```



The relational JOIN was reproduced using:



```python

pd.merge()

```



The verification result was:



```text

SQL JOIN and pd.merge() match: True

```



Pandas results are saved in:



```text

data\_pipeline/data/pandas\_outputs.txt

```



\## Running Module 1



From the project root:



```powershell

python data\_pipeline\\scraper.py

```



```powershell

python data\_pipeline\\clean\_data.py

```



```powershell

python data\_pipeline\\database.py

```



```powershell

python data\_pipeline\\queries.py

```



```powershell

python data\_pipeline\\pandas\_analysis.py

```



\---



\# Module 2 – Analytics \& Machine Learning



\## Dataset



Module 2 uses the Titanic dataset.



The dataset is loaded using:



```python

sns.load\_dataset("titanic")

```



It is immediately saved as:



```text

analytics/titanic.csv

```



The original dataset contains:



```text

891 rows

15 columns

```



All subsequent analysis uses the saved dataset.



\---



\## Part A – Exploratory Data Analysis



The analysis includes:



\* dataset shape

\* dataset information

\* descriptive statistics

\* missing-value analysis

\* outlier detection

\* age analysis

\* fare analysis

\* survival analysis

\* correlation analysis

\* multivariate visualizations

\* EDA-only standardization



\### Missing Values



The main missing




## Git Workflow

The project was developed using a feature branch and merged into the main branch after validation.
