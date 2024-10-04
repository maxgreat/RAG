import argparse
from flask import Flask, request, jsonify
from embeddings import EmbeddingHandler
from search import FAISSHandler
from documentsHandler import start_watching, add_document_to_index
from query import RAGQueryHandler
from typing import Optional


app = Flask(__name__)

embedding_handler : Optional[EmbeddingHandler] = None
faiss_handler : Optional[FAISSHandler]= None

@app.route('/add_document', methods=['POST'])
def add_document():
    content = request.json
    document = content.get('document', None)
    chunk_size = content.get('chunk_size', 300)
    overlap = content.get('overlap', 50)

    if document:
        chunks = add_document_to_index(embedding_handler, faiss_handler, document, chunk_size, overlap)
        return jsonify({"message": "Document added successfully", "chunks": chunks}), 200
    else:
        return jsonify({"error": "No document provided"}), 400

@app.route('/query', methods=['POST'])
def query():
    content = request.json
    query_text = content.get('query', None)
    k = content.get('k', 3)

    if query_text:
        # Initialize RAG query handler
        query_handler = RAGQueryHandler(model_name="mistral-7b-instruct")
        
        # Generate query embedding and perform FAISS search
        if embedding_handler is not None:
            query_embedding = embedding_handler.create_embeddings([query_text])
        else:
            raise ValueError("embedding_handler is not initialized")
        if faiss_handler is not None:
            distances, indices = faiss_handler.search(query_embedding, k=k)
        else:
            raise ValueError("faiss_handler is not initialized")
        
        
        # Assuming document chunks are retrieved by index, this is a simple example
        retrieved_chunks = [content.get('document').split(".")[i] for i in indices]

        # Generate response
        response = query_handler.generate_response(query_text, retrieved_chunks)
        return jsonify({"query": query_text, "response": response}), 200
    else:
        return jsonify({"error": "No query provided"}), 400

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='RAG Service with Automatic Document Scanning and Querying')
    
    parser.add_argument('--model_name', type=str, default="intfloat/e5-large-v2",
                        help="Model name for the embedding handler (default: 'intfloat/e5-large-v2')")
    parser.add_argument('--embeddings_dim', type=int, default=4096,
                        help="Embeddings dimension (default: 4096)")
    
    parser.add_argument('--mode', type=str, choices=['cli', 'server'], required=True,
                        help="Mode to run the service: 'cli' for command-line interaction, 'server' for API mode")

    args = parser.parse_args()

    # Initialize the embedding handler and FAISS handler
    global embedding_handler
    embedding_handler = EmbeddingHandler(model_name=args.model_name)
    global faiss_handler
    faiss_handler = FAISSHandler(embedding_dim=args.embedding_dim)
    
    start_watching("documents/", embedding_handler, faiss_handler)
    if args.mode == 'server':
        app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
