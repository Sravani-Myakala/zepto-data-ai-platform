from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = BASE_DIR / "documents" / "zepto_policy.txt"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"


# ============================================================
# 1. LOAD POLICY DOCUMENT
# ============================================================

def load_policy_document():
    """Load the Zepto policy document."""

    if not POLICY_FILE.exists():
        raise FileNotFoundError(
            f"Policy document not found: {POLICY_FILE}"
        )

    text = POLICY_FILE.read_text(encoding="utf-8")

    return Document(
        page_content=text,
        metadata={"source": "zepto_policy.txt"}
    )


# ============================================================
# 2. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

def split_document(document):
    """Split the policy into smaller chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents([document])

    return chunks


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

def create_embeddings():
    """Create sentence-transformer embeddings."""

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# 4. CREATE / LOAD CHROMADB
# ============================================================

def create_vector_store(chunks, embeddings):
    """Store document chunks in ChromaDB."""

    vector_store = Chroma(
        collection_name="zepto_policy",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Add documents only when the collection is empty.
    existing = vector_store.get()

    if not existing["ids"]:
        vector_store.add_documents(chunks)
        print(f"Added {len(chunks)} policy chunks to ChromaDB.")
    else:
        print(
            f"Loaded existing ChromaDB with "
            f"{len(existing['ids'])} stored chunks."
        )

    return vector_store


# ============================================================
# 5. CREATE HUGGING FACE LLM
# ============================================================

def create_llm():
    """Create a free Hugging Face chat client."""
    return InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct"
    )

# ============================================================
# 6. RAG PROMPT
# ============================================================

PROMPT = ChatPromptTemplate.from_template(
    """
You are a Zepto customer support assistant.

Answer the customer's question ONLY using the policy context
provided below.

If the answer is not available in the context, say:

"I couldn't find that information in the available Zepto policy."

Do not invent policies, refund amounts, delivery times, guarantees,
or other information that is not present in the context.

POLICY CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

ANSWER:
"""
)


# ============================================================
# 7. RAG QUESTION ANSWERING
# ============================================================

def answer_question(question, vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(question)

    if not documents:
        return "I couldn't find that information in the available Zepto policy."

    context = "\n\n".join(document.page_content for document in documents)

    prompt = PROMPT.format(
        context=context,
        question=question
    )

    response = llm.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=256,
        temperature=0.1
    )

    return response.choices[0].message.content


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("ZEPTO RAG SUPPORT ASSISTANT")
    print("=" * 70)

    print("\nLoading policy document...")
    document = load_policy_document()
    print("Policy document loaded successfully.")

    print("\nSplitting document into chunks...")
    chunks = split_document(document)
    print(f"Created {len(chunks)} document chunks.")

    print("\nCreating Hugging Face embeddings...")
    embeddings = create_embeddings()
    print("Embeddings created successfully.")

    print("\nCreating/loading ChromaDB...")
    vector_store = create_vector_store(
        chunks,
        embeddings
    )
    print("ChromaDB ready.")

    print("\nConnecting to Hugging Face LLM...")
    llm = create_llm()
    print("LLM connection configured.")

    print("\n" + "=" * 70)
    print("CHATBOT READY")
    print("Type 'exit' to stop.")
    print("=" * 70)

    while True:

        question = input("\nCustomer: ").strip()

        if question.lower() == "exit":
            print("\nThank you for using the Zepto Support Assistant.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = answer_question(
                question,
                vector_store,
                llm
            )

            print("\nAssistant:")
            print(answer)

        except Exception as error:
            print("\nError while generating the answer:")
            print(error)


if __name__ == "__main__":
    main()