import faiss
import numpy as np

class FAISSHandler:
    def __init__(self, embedding_dim: int):
        """
        Initialize the FAISS index for similarity search.
        :param embedding_dim: The dimensionality of the embeddings.
        """
        self.index = faiss.IndexFlatL2(embedding_dim)  # Using L2 distance for similarity search
    
    def add_embeddings(self, embeddings: list[np.ndarray]):
        """
        Add embeddings to the FAISS index.
        :param embeddings: A list of embeddings to add.
        """
        embeddings_np = np.array(embeddings).astype('float32')  # Convert to NumPy array of float32
        self.index.add(embeddings_np)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Perform a similarity search for the query embedding.
        :param query_embedding: The embedding to search for.
        :param k: The number of nearest neighbors to return (default 5).
        :return: Distances and indices of the nearest neighbors.
        """
        query_embedding_np = np.array([query_embedding]).astype('float32')  # Convert to NumPy array of float32
        distances, indices = self.index.search(query_embedding_np, k)
        return distances, indices

# Example usage
if __name__ == "__main__":
    handler = FAISSHandler(embedding_dim=4096)
    
    example_embeddings = [np.random.rand(4096) for _ in range(10)]
    
    # Adding embeddings to the index
    handler.add_embeddings(example_embeddings)
    
    # Search for a query embedding
    query_embedding = np.random.rand(4096)
    distances, indices = handler.search(query_embedding)
    
    print("Nearest neighbors indices:", indices)
    print("Distances:", distances)