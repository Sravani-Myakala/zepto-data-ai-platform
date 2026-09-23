from pathlib import Path
from typing import TypedDict
import os


import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"


# ============================================================
# CONFIGURATION
# ============================================================

# Official grading requirement:
# MOCK_LLM unset or MOCK_LLM=1 -> Mock mode
# MOCK_LLM=0 -> Optional real Hugging Face mode

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = "openai/gpt-oss-120b"

hf_client = None

if HF_TOKEN:
    hf_client = InferenceClient(
        api_key=HF_TOKEN
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="zepto_policy",
    metadata={
        "hnsw:space": "cosine"
    }
)

print(
    "ChromaDB collection loaded. "
    f"Documents/chunks: {collection.count()}"
)


# ============================================================
# PYDANTIC RESPONSE SCHEMA
# ============================================================

class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# LANGGRAPH STATE
# ============================================================

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# STRUCTURED PROMPT TEMPLATE
# ============================================================

POLICY_PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the retrieved Zepto policy documents provided below.

TASK:
Answer the customer's question using the retrieved policy context.

FORMAT:
Give a clear and concise customer-support answer.

LENGTH:
Keep the answer between 2 and 4 sentences.

NEGATIVE CONSTRAINT:
Do not invent, assume, or add policy information that is not present
in the retrieved context.

FEW-SHOT EXAMPLE:
Customer Question:
How long does delivery take?

Retrieved Context:
Delivery typically takes 10-30 minutes depending on location,
availability, and delivery conditions.

Answer:
Zepto delivery typically takes around 10-30 minutes, depending on
your location and current delivery conditions.

Customer Question:
{query}

Retrieved Zepto Policy Context:
{context}

Answer:
"""


# ============================================================
# NODE 1: CLASSIFY INTENT
# ============================================================

def classify_intent(
    state: AssistantState
) -> AssistantState:

    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "cancellation",
        "gift card",
        "support hours"
    ]

    if any(
        keyword in query
        for keyword in policy_keywords
    ):
        intent = "policy_question"

    else:
        intent = "general_question"

    print(
        f"Intent classified as: {intent}"
    )

    return {
        "intent": intent
    }


# ============================================================
# NODE 2: RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: AssistantState
) -> AssistantState:

    query = state["query"]

    print(
        "Retrieving policy documents..."
    )

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).tolist()

    # --------------------------------------------------------
    # Search ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_documents = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    if not retrieved_documents:

        response = AssistantResponse(
            answer=(
                "I could not find relevant information "
                "in the available Zepto policy documents."
            ),
            sources=[],
            confidence=0.0
        )

        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }

    # --------------------------------------------------------
    # Get source filenames
    # --------------------------------------------------------

    sources = [
        metadata["source"]
        for metadata in retrieved_metadata
    ]

    # --------------------------------------------------------
    # Combine retrieved documents
    # --------------------------------------------------------

    context = "\n\n".join(
        retrieved_documents
    )

    # --------------------------------------------------------
    # Create structured prompt
    # --------------------------------------------------------

    prompt = POLICY_PROMPT_TEMPLATE.format(
        query=query,
        context=context
    )

    # ========================================================
    # MOCK LLM MODE
    # ========================================================

    if MOCK_LLM:

        # Official requirement:
        # Use the top retrieved chunk and keep it short.

        top_chunk = retrieved_documents[0][:200]

        answer = (
            "Based on the retrieved context: "
            + top_chunk
        )

        # ----------------------------------------------------
        # Deterministic Pydantic validation
        # ----------------------------------------------------

        response = AssistantResponse(
            answer=answer,
            sources=sources,
            confidence=1.0
        )

        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }

    # ========================================================
    # REAL HUGGING FACE LLM MODE
    # ========================================================

    if hf_client is None:

        response = AssistantResponse(
            answer=(
                "The Hugging Face API is not configured. "
                "Please use MOCK_LLM=1 or configure HF_TOKEN."
            ),
            sources=sources,
            confidence=0.0
        )

        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }

    # --------------------------------------------------------
    # Real LLM with validation/retry
    # --------------------------------------------------------

    last_error = None

    for attempt in range(3):

        try:

            # First attempt uses the normal structured prompt.
            # Later attempts include a corrective instruction.

            if attempt == 0:

                user_prompt = prompt

            else:

                user_prompt = (
                    prompt
                    + "\n\n"
                    + "CORRECTIVE INSTRUCTION:\n"
                    + "Return only a concise answer based strictly "
                    + "on the retrieved context. Do not invent facts."
                )

            llm_response = hf_client.chat_completion(
                model=HF_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Zepto customer support assistant. "
                            "Use only the provided policy context."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                max_tokens=400,
                temperature=0.2
            )

            generated_answer = (
                llm_response
                .choices[0]
                .message
                .content
            )

            # Validate generated answer using Pydantic.

            response = AssistantResponse(
                answer=generated_answer,
                sources=sources,
                confidence=0.95
            )

            return {
                "answer": response.answer,
                "sources": response.sources,
                "confidence": response.confidence
            }

        except Exception as error:

            last_error = error

            print(
                f"LLM/validation attempt "
                f"{attempt + 1}/3 failed: {error}"
            )

    # --------------------------------------------------------
    # Final error response after 3 total attempts
    # --------------------------------------------------------

    response = AssistantResponse(
        answer=(
            "I was unable to generate a validated answer "
            "from the available Zepto policy information."
        ),
        sources=sources,
        confidence=0.0
    )

    print(
        f"Final LLM error: {last_error}"
    )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


# ============================================================
# NODE 3: DIRECT ANSWER
# ============================================================

def direct_answer(
    state: AssistantState
) -> AssistantState:

    answer = (
        "I can only answer questions about Zepto policies right now."
    )

    response = AssistantResponse(
        answer=answer,
        sources=[],
        confidence=1.0
    )

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_by_intent(
    state: AssistantState
) -> str:

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    AssistantState
)

# Add required nodes

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

# Start -> classify

graph_builder.add_edge(
    START,
    "classify_intent"
)

# Conditional edge

graph_builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

# End edges

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)


# Compile graph

assistant_graph = graph_builder.compile()


# ============================================================
# PUBLIC ASSISTANT FUNCTION
# ============================================================

def ask_assistant(
    query: str
) -> AssistantResponse:

    if not query or not query.strip():

        return AssistantResponse(
            answer=(
                "Please enter a question."
            ),
            sources=[],
            confidence=0.0
        )

    initial_state: AssistantState = {
        "query": query.strip()
    }

    result = assistant_graph.invoke(
        initial_state
    )

    return AssistantResponse(
        answer=result.get(
            "answer",
            "Unable to generate an answer."
        ),
        sources=result.get(
            "sources",
            []
        ),
        confidence=result.get(
            "confidence",
            0.0
        )
    )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("Zepto Support Assistant")
    print("=" * 60)

    print(
        "MOCK_LLM:",
        MOCK_LLM
    )

    print(
        "ChromaDB documents/chunks:",
        collection.count()
    )

    print("\nPolicy test:")

    policy_result = ask_assistant(
        "What is the refund policy?"
    )

    print(
        "\nAnswer:",
        policy_result.answer
    )

    print(
        "Sources:",
        policy_result.sources
    )

    print(
        "Confidence:",
        policy_result.confidence
    )

    print("\nGeneral question test:")

    general_result = ask_assistant(
        "Tell me a joke"
    )

    print(
        "\nAnswer:",
        general_result.answer
    )

    print(
        "Sources:",
        general_result.sources
    )

    print(
        "Confidence:",
        general_result.confidence
    )

    print("\n" + "=" * 60)