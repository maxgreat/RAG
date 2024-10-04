# RAG Service with Document Indexing and Querying

This repository implements a Retrieval-Augmented Generation (RAG) service that allows users to:
- **Index documents** (e.g., PDFs, DOCX, Markdown, TXT).
- **Query the indexed documents** using FAISS for similarity search.
- Automatically **watch a directory** for new documents and index them.

The service supports both **command-line** and **HTTP API** interactions using `Flask`.

## Features
- Add documents in various formats (PDF, DOCX, MD, TXT).
- Query documents using similarity search and get a response from an LLM.
- Watch a directory for new documents.

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Service
You can run the service in two modes:
- **CLI Mode**: Watch a directory for new documents and index them.
- **API Mode**: Start an HTTP API to interact with the service.

#### CLI Mode
```bash
python main.py --model_name "intfloat/e5-large-v2" --watch_directory "/path/to/documents" --mode cli
```

#### API Mode
```bash
python main.py --mode server
```

## API Endpoints

### Add Document
```bash
POST /add_document
Content-Type: application/json

{
    "document": "This is the content of the document to be indexed."
}
```

### Query
```bash
POST /query
Content-Type: application/json

{
    "query": "What is the content of the document?"
}
```

## Customization

- You can modify the default models for embeddings and the RAG query handler.
- Supported document formats include `.pdf`, `.md`, and `.txt`.

## License
This project is licensed under the MIT License.
