from vllm import LLM

class RAGQueryHandler:
    def __init__(self, model_name: str = "mistral-7b-instruct"):
        """
        Initialize the query handler with vLLM for generating the final response.
        :param model_name: The name of the LLM for generating responses (default: 'mistral-7b-instruct').
        """
        self.model = LLM(model_name)

    def generate_response(self, query: str, retrieved_texts: list[str]) -> str:
        """
        Generate a response to a query based on the retrieved documents using the LLM.
        :param query: The input query.
        :param retrieved_texts: The list of retrieved texts from the FAISS index.
        :return: The generated response string.
        """
        # Format the prompt for the LLM
        prompt = f"Answer the following question based on the provided documents:\n\n"
        prompt += "Documents:\n" + "\n\n".join(retrieved_texts) + "\n\n"
        prompt += f"Question: {query}\nAnswer:"

        # Use vLLM to generate the output
        outputs = self.model.generate([prompt])

        # Return the first generated response (vLLM returns a list of responses)
        return outputs[0]