
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str], embed_model) -> list[list[float]]:
    # Generate embeddings for all chunks
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()  # returns list of lists
