
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url = "http://localhost:6333", collection = "docs", dim = 384):
        self.client = QdrantClient(url = url, timeout = 60)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config = VectorParams(size = dim, distance = Distance.COSINE)
            )
        self.dim = dim

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id = ids[i], vector = vectors[i], payload = payloads[i]) for i in range(len(ids))
        ]
        self.client.upsert(
            collection_name = self.collection,
            points = points,
        )

    def search(self, query_vector, query_text, top_k: int = 5):

        # results = self.client.query(
        #     collection_name=self.collection,
        #     query_text = query_text,
        #     limit=top_k
        # )

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_vectors=True,
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set()

        print(type(results))

        for result in results.points:
            payload = getattr(result, "payload", {}) or {}
            text = payload.get("text")
            source = payload.get("source")
            if text:
                contexts.append(text)
            if source:
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}

