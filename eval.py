import re
from typing import List, Dict

import numpy as np

def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    deno = np.linalg.norm(a) * np.linalg.norm(b)
    if deno == 0:
        return 0.0
    return np.dot(a, b) / deno


def evaluate_answer(answer: str, sources: List[object], embed_model):
    """
    - Splits the answer into sentences.
    - Computes cosine similarity between each sentence and all source chunks.
    """

    sentences = split_sentences(answer)

    # Embed answer sentences and source chunks
    sentence_vectors = [embed_model.encode(s) for s in sentences]
    source_texts = [getattr(doc, "page_content", str(doc)) for doc in sources]
    source_vectors = [embed_model.encode(t) for t in source_texts]

    # Max similarity across sources for all sentences
    per_sentence_max = []
    for sv in sentence_vectors:
        sim = [cosine_similarity(np.array(sv), np.array(tv)) for tv in source_vectors]
        per_sentence_max.append(max(sim) if sim else 0.0)

    coverage = float(np.mean(per_sentence_max)) if per_sentence_max else 0.0
    max_sim = float(np.max(per_sentence_max)) if per_sentence_max else 0.0

    metrics = {
        "coverage_score": round(coverage, 3),
        "max_similarity": round(max_sim, 3),
        "details": per_sentence_max,
    }

    print("Evaluation: \n")
    print(metrics)

    return metrics
