import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest import Inngest
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

load_dotenv()

# setup
# uv init .

# install dependencies on a venv
# uv add <deps>,<deps>

# uv run uvicorn main:app
# npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery

# Run qdrant vector database on docker locally
# docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant


# Load model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",        # Automatically put layers on CPU/GPU
    torch_dtype=torch.float16 # Saves memory, works best on GPU
)
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024, temperature=0.2)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,   # ONLY controls generated output
    temperature=0.2,
    do_sample=False
)

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
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id = "RAG: Query DOC",
    trigger = inngest.TriggerEvent(event="rag/query_doc")
)

async def query_doc(ctx:inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, question, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 3))

    found = await ctx.step.run("embed-and-search", lambda:_search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
    )

    # ---- LLaMA generation ----
    response = generator(truncate_prompt(user_content))
    answer = response[0]["generated_text"].strip()

    print("Answer: ", answer)
    print("Response: ", response)

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


def truncate_prompt(prompt: str) -> str:
    MAX_INPUT_TOKENS = 1800  # leave room for answer

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > MAX_INPUT_TOKENS:
        tokens = tokens[-MAX_INPUT_TOKENS:]  # keep last part (question + recent context)
    return tokenizer.decode(tokens)

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [ingest_doc, query_doc])
