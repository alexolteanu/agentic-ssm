from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

def print_chroma_stats(persist_directory: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if not os.path.exists(persist_directory):
        print(f"Directory '{persist_directory}' does not exist.")
        return

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Load Chroma DB
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    
    collection = vectordb._collection  # access internal collection
    count = collection.count()

    print(f"Chroma DB Summary")
    print(f"────────────────────────────────────────────────────")
    print(f"Path: {persist_directory}")
    print(f"Collection Name: {collection.name}")
    print(f"Number of Documents: {count}")

    if count > 0:
        metadata = collection.get(include=["metadatas", "documents", "embeddings"], limit=5)
        if len(metadata['embeddings']) > 0:
            embedding_dim = len(metadata['embeddings'][0])
        else:
            embedding_dim = "Unknown"

        print(f"Embedding Dimension: {embedding_dim}")
        print(f"Metadata fields example: {list(metadata['metadatas'][0].keys()) if metadata['metadatas'] else 'None'}")
        print(f"Example document snippet: {metadata['documents'][0][:250]}...")
    else:
        print("No documents found in the collection.")

    print(f"────────────────────────────────────────────────────")

print_chroma_stats("ssm_chroma_db_minilm")