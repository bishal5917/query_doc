
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer

load_dotenv()

splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# genai.configure(api_key=GOOGLE_API_KEY)
# EMBED_MODEL = "models/embedding-001"   # Google embedding model

# Loading a free embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


# def embed_texts(texts: list[str]) -> list[list[float]]:
#     embeddings = []
#     for t in texts:
#         r = genai.embed_content(
#             model=EMBED_MODEL,
#             content=t,
#             task_type="retrieval_document"
#         )
#         embeddings.append(r["embedding"])
#     return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    # Generate embeddings for all chunks
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()  # returns list of lists
