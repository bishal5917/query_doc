import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid

from sentence_transformers import SentenceTransformer

from data_loader import load_and_chunk_pdf, embed_texts
from llm import generate_with_ollama, check_ollama_connection
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult
from eval import evaluate_answer

load_dotenv()

# setup
# uv init .

# install dependencies on a venv
# uv add <deps>,<deps>

# uv run uvicorn main:app
# npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

# Run qdrant vector database on docker locally
# docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# Run ollama on docker locally
# docker run -d --name ollama --gpus all -p 11434:11434 -v "$(pwd)/ollama:/ollama" ollama/ollama

# Run streamlit application
# uv run streamlit run .\streamlit_app.py

# Loading a free embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"

inngest_client = inngest.Inngest(
    app_id = "doc_query",
    logger = logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id = "RAG: Ingest DOC",
    trigger = inngest.TriggerEvent(event="rag/ingest_doc")
)

async def ingest_doc(ctx:inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(ctx: inngest.Context) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks, embed_model)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query DOC",
    trigger=inngest.TriggerEvent(event="rag/query_doc")
)
async def query_doc(ctx: inngest.Context):

    def _search(question: str, top_k: int):
        query_vec = embed_texts([question], embed_model)[0]
        store = QdrantStorage()
        return store.search(query_vec, question, top_k)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k)
    )

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])

    prompt = (
        "You are a helpful assistant. Answer the question strictly using the context provided.\n"
        "If the answer is not contained in the context, respond with 'I don't know' and do not make up any information.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    answer = await ctx.step.run(
        "ollama-generate",
        lambda: generate_with_ollama(prompt, OLLAMA_HOST, OLLAMA_MODEL)
    )

    resp =  {
        "answer": answer,
        "sources": found["sources"],
        "num_contexts": len(found["contexts"])
    }

    evaluate_answer(answer=answer, sources=found["sources"], embed_model=embed_model)

    return resp

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [ingest_doc, query_doc])

if __name__ == "__main__":
    print(check_ollama_connection(OLLAMA_HOST))
    print(generate_with_ollama("Say hello in one sentence", OLLAMA_HOST, OLLAMA_MODEL))