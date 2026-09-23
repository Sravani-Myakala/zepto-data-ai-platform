from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# --------------------------------------------------
# Create ChromaDB client
# --------------------------------------------------

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

collection = client.get_or_create_collection(
    name="zepto_policy",
    metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# Read policy documents
# --------------------------------------------------

documents = []
ids = []
metadatas = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):

    text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        continue

    documents.append(text)
    ids.append(file_path.stem)
    metadatas.append({
        "source": file_path.name
    })


print(f"Found {len(documents)} policy documents.")


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

print("Creating embeddings...")

embeddings = model.encode(
    documents,
    convert_to_numpy=True
).tolist()

print("Embeddings created.")


# --------------------------------------------------
# Store in ChromaDB
# --------------------------------------------------

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Documents stored in ChromaDB.")


# --------------------------------------------------
# Verification
# --------------------------------------------------

print("\nVerification")
print("-" * 40)

print("Documents in collection:", collection.count())

test_query = "How long does Zepto delivery take?"

query_embedding = model.encode(
    [test_query],
    convert_to_numpy=True
).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print("\nTest Query:")
print(test_query)

print("\nRetrieved Sources:")

for source in results["metadatas"][0]:
    print("-", source["source"])

print("\nTop Retrieved Text:")
print(results["documents"][0][0][:300])

print("\nStep 4 completed successfully!")