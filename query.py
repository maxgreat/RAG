from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGQueryHandler:
    def __init__(self, model_name: str = "mistral-7b-instruct"):
        """
        Initialize the query handler with a language model for generating the final response.
        :param model_name: The name of the LLM for generating responses (default: 'mistral-7b-instruct').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
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
        
        # Tokenize and generate response
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=512)
        
        # Decode the generated response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
