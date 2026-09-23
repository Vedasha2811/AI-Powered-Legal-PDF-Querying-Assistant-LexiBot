import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGSystem:

    def __init__(self):

        # Load embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.index = None
        self.chunks = []


    def build_index(self, chunks):

        self.chunks = chunks

        # Create embeddings
        embeddings = self.model.encode(chunks)

        embeddings = np.array(
            embeddings
        ).astype("float32")

        # Create FAISS index
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        # Add embeddings
        self.index.add(embeddings)


    def search(self, question, k=3):

        # Convert question into embedding
        question_embedding = self.model.encode(
            [question]
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")

        # Search FAISS
        distances, indices = self.index.search(
            question_embedding,
            k
        )

        # Store relevant results
        results = []

        for i in range(k):

            chunk_index = indices[0][i]

            # Ignore invalid FAISS results
            if chunk_index == -1:
                continue

            results.append(
                {
                    "chunk_id": int(chunk_index + 1),
                    "text": self.chunks[chunk_index],
                    "distance": float(distances[0][i])
                }
            )

        return results