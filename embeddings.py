from vllm import LLM
import numpy as np


class EmbeddingHandler:
    def __init__(self, model_name: str = "intfloat/e5-mistral-7b-instruct"):
        """
        Initialize the embedding handler with vLLM optimized inference.
        :param model_name: Name of the model to load for embeddings, defaulting to e5-mistral-7b.
        """
        self.model = LLM(model_name)

    def create_embeddings(self, text_chunks: list[str]) -> np.ndarray:
        """
        Generate embeddings from a batch of text chunks using vLLM.
        :param text_chunks: A list of text chunks.
        :return: A list of embeddings, each as a list of floats.
        """
        outputs = self.model.generate(text_chunks)
        
        # Return the embeddings
        return np.array(outputs.embeddings).astype('float32')
    

# Example usage
if __name__ == "__main__":
    handler = EmbeddingHandler()
    
    text_chunks = [
        "This is the first chunk of text.",
        "Here is another chunk for embeddings."
    ]
    
    embeddings = handler.create_embeddings(text_chunks)
    
    print(embeddings)