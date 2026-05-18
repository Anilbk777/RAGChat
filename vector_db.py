
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(
        self,
        url="http://localhost:6333",
        collection="docs",
        dim=768
    ):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim

    def create_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

    def delete_collection(self):
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

    def recreate_collection(self):
        self.delete_collection()
        self.create_collection()

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

if __name__ == "__main__":
    db = QdrantStorage(dim=768)
    db.recreate_collection()