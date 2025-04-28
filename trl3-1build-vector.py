from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# STEP 1: Load your PDFs
pdf_path = "legea-319-2006.pdf" 
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# STEP 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# STEP 3: Embed and Save
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="ssm_chroma_db"  
)

print("Chroma DB rebuilt and persisted!")