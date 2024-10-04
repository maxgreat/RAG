import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain.document_loaders import TextLoader, PDFPlumberLoader, MarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentWatcher(FileSystemEventHandler):
    def __init__(self, watch_directory, embedding_handler, faiss_handler):
        self.watch_directory = watch_directory
        self.embedding_handler = embedding_handler
        self.faiss_handler = faiss_handler

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        print(f"New document detected: {file_path}")
        
        # Process the document based on file extension
        if file_path.endswith(".pdf"):
            loader = PDFPlumberLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".md"):
            loader = MarkdownLoader(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return
        
        # Load the document and split into chunks
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(document[0].page_content)

        # Generate embeddings for the chunks and add them to the FAISS index
        embeddings = self.embedding_handler.create_embeddings(chunks)
        self.faiss_handler.add_embeddings(embeddings)
        print(f"Document {os.path.basename(file_path)} added to index successfully.")

def start_watching(directory, embedding_handler, faiss_handler):
    event_handler = DocumentWatcher(directory, embedding_handler, faiss_handler)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    print(f"Watching directory: {directory}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
