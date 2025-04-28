from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Reuse the same embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = "ssm_chroma_db"

# Load existing Chroma
vectordb = Chroma(
    persist_directory=persist_dir,
    embedding_function=embedding_model,
    collection_name="ssm_docs"
)

# Example usage: retrieve
retriever = vectordb.as_retriever()

# 1. Create your system prompt
system_message = SystemMessagePromptTemplate.from_template(
    """Esti un asistent AI care ajută utilizatorul să răspundă la întrebări bazate pe un document PDF. 
    Raspunde intr-un format structurat JSON 
    Nivelul de incredere poate fi de la 0 la 1. Dacă nu găsești informația în document, răspunde cu un nivel scazut de incredere. """
)

# 2. Create your user prompt
human_message = HumanMessagePromptTemplate.from_template(
    "Answer the following question based on the context: {context}\n\nQuestion: {question}"
)

# 3. Combine into a ChatPrompt
prompt = ChatPromptTemplate.from_messages([system_message, human_message])


# Example: ask something
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",   # or "map_reduce" or "refine"
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
def batch_query_rag(rag_chain, input_file="query.in", output_file="query.out"):
    # Read all queries
    with open(input_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    results = []

    for query in queries:
        print(f"Asking: {query}")
        response = rag_chain.invoke({"query": query})
        answer = response["result"]
    
        
        results.append({
            "query": query,
            "answer": answer,
        })

    # Write results to output file
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(f"{item['answer']}\n")
            f.write("\n" + "-"*50 + "\n\n")

    print(f"\n✅ Done! Results saved to '{output_file}'.")

batch_query_rag(rag_chain, input_file="query.in", output_file="query.out")