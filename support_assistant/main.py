import os
from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "zepto_policy_rubric"


# ============================================================
# MOCK LLM TOGGLE
# ============================================================

MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# STRUCTURED PROMPT
# role -> context -> task -> format -> length
# ============================================================

STRUCTURED_PROMPT = """
ROLE:
You are a Zepto customer-support policy assistant.

CONTEXT:
Use only the Zepto policy documents retrieved for this question.

TASK:
Answer the customer's question using only the retrieved context.

FORMAT:
Return a clear customer-support answer. Do not invent policy details.

LENGTH:
Keep the answer concise and normally within 2 to 4 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent prices, refund percentages, delivery guarantees, or policy rules.

FEW-SHOT EXAMPLE:
Customer Question: Can I cancel my order before it is packed?
Context: Orders can be cancelled free of cost before the order status changes to Packed.
Answer: Yes. The policy says orders can be cancelled free of cost before the order status changes to Packed.

Customer Question:
{question}

Retrieved Context:
{context}

Answer:
"""


# ============================================================
# PYDANTIC RESPONSE SCHEMA
# ============================================================

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AskRequest(BaseModel):
    query: str


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_documents: list[dict]
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD + CHUNK 8 POLICY DOCUMENTS
# ============================================================

def load_policy_documents():
    files = sorted(DOCUMENTS_DIR.glob("doc_*.txt"))

    if len(files) != 8:
        raise RuntimeError(
            f"Expected exactly 8 policy documents, found {len(files)}."
        )

    documents = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8").strip()

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": file_path.stem,
                    "source": file_path.name
                }
            )
        )

    return documents


def split_policy_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks = []

    for document in documents:
        split_chunks = splitter.split_documents([document])

        for index, chunk in enumerate(split_chunks):
            chunk.metadata["chunk_id"] = (
                f"{document.metadata['doc_id']}_chunk_{index + 1}"
            )
            chunks.append(chunk)

    return chunks


# ============================================================
# CHROMADB
# ============================================================

def create_vector_store():
    documents = load_policy_documents()
    chunks = split_policy_documents(documents)
    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )

    existing = vector_store.get()

    if existing.get("ids"):
        vector_store.delete(ids=existing["ids"])

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    stored = vector_store.get()

    if len(stored["ids"]) != 8:
        raise RuntimeError(
            f"Expected 8 stored policy chunks, found {len(stored['ids'])}."
        )

    print(f"Loaded {len(documents)} policy documents.")
    print(f"Stored {len(stored['ids'])} policy chunks in ChromaDB.")

    return vector_store


# ============================================================
# OPTIONAL REAL LLM
# ============================================================

def get_llm_client():
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is required only when MOCK_LLM=0."
        )

    model = os.getenv(
        "HF_MODEL",
        "Qwen/Qwen2.5-1.5B-Instruct"
    )

    return InferenceClient(
        model=model,
        token=token
    )


def call_real_llm(prompt):
    client = get_llm_client()

    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=256,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


# ============================================================
# LANGGRAPH NODE 1
# classify_intent
# ============================================================

def classify_intent(state: GraphState):
    query = state["query"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if MOCK_LLM:
        intent = (
            "policy_question"
            if any(keyword in query for keyword in keywords)
            else "general_question"
        )
    else:
        prompt = f"""
Classify the customer's query as exactly one of:
policy_question
general_question

Customer query:
{state["query"]}

Return only the classification.
"""
        result = call_real_llm(prompt).lower()

        intent = (
            "policy_question"
            if "policy_question" in result
            else "general_question"
        )

    return {
        "intent": intent
    }


# ============================================================
# LANGGRAPH NODE 2
# retrieve_and_answer
# ============================================================

def retrieve_and_answer(state: GraphState):
    query = state["query"]

    documents = VECTOR_STORE.similarity_search(
        query,
        k=3
    )

    retrieved_documents = []

    for document in documents:
        retrieved_documents.append(
            {
                "chunk_id": document.metadata["chunk_id"],
                "source": document.metadata["source"],
                "content": document.page_content
            }
        )

    if not retrieved_documents:
        return {
            "answer": "I could not find that information in the available Zepto policies.",
            "sources": [],
            "confidence": 1.0,
            "retrieved_documents": []
        }

    if MOCK_LLM:
        top_chunk = retrieved_documents[0]

        snippet = top_chunk["content"][:200].strip()

        answer = (
            f"Based on the retrieved context: {snippet}"
        )

    else:
        context = "\n\n".join(
            item["content"]
            for item in retrieved_documents
        )

        prompt = STRUCTURED_PROMPT.format(
            question=query,
            context=context
        )

        answer = call_real_llm(prompt)

    return {
        "retrieved_documents": retrieved_documents,
        "answer": answer,
        "sources": [
            item["chunk_id"]
            for item in retrieved_documents
        ],
        "confidence": 1.0
    }


# ============================================================
# LANGGRAPH NODE 3
# direct_answer
# ============================================================

def direct_answer(state: GraphState):
    if MOCK_LLM:
        answer = (
            "I can only answer questions about the available Zepto "
            "policies right now."
        )
    else:
        prompt = f"""
You are a Zepto support assistant.

Customer question:
{state["query"]}

Answer briefly. Do not invent policy details.
"""
        answer = call_real_llm(prompt)

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_classification(state: GraphState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node(
        "classify_intent",
        classify_intent
    )

    builder.add_node(
        "retrieve_and_answer",
        retrieve_and_answer
    )

    builder.add_node(
        "direct_answer",
        direct_answer
    )

    builder.add_edge(
        START,
        "classify_intent"
    )

    builder.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer"
        }
    )

    builder.add_edge(
        "retrieve_and_answer",
        END
    )

    builder.add_edge(
        "direct_answer",
        END
    )

    return builder.compile()


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

print("=" * 70)
print("ZEPTO SUPPORT ASSISTANT")
print("=" * 70)
print(f"MOCK_LLM mode: {MOCK_LLM}")

VECTOR_STORE = create_vector_store()
GRAPH = build_graph()

print("LangGraph ready.")
print("FastAPI application ready.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Zepto RAG Support Assistant",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "service": "Zepto RAG Support Assistant",
        "mock_llm": MOCK_LLM,
        "endpoint": "POST /ask"
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):
    result = GRAPH.invoke(
        {
            "query": request.query
        }
    )

    return AnswerResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )
