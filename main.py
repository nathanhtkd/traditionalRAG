import os
import shutil
import hashlib
import time
import random
import pandas as pd
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from groq import Groq
from dotenv import load_dotenv

# Constants
OUTPUT_FILE = 'ingestion_benchmarks.csv'
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

def split_text(documents):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def create_db(chunks, embeddings):
    """Create a new Chroma database with unique path and handle permissions."""
    unique_chroma_path = os.path.join(tempfile.gettempdir(), f"chroma_{uuid.uuid4()}")
    os.makedirs(unique_chroma_path, mode=0o700)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=unique_chroma_path,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    for root, dirs, files in os.walk(unique_chroma_path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o700)
        for f in files:
            os.chmod(os.path.join(root, f), 0o600)
            
    return db

def benchmark_ingestion(pdf_dir, num_docs):
    """Benchmark ingestion time for specified number of documents."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    selected_files = random.sample(pdf_files, num_docs)
    
    # Print selected files
    print("\nProcessing files:")
    for i, file in enumerate(selected_files, 1):
        print(f"{i}. {file}")
    
    # Create a temporary directory for selected files
    temp_dir = tempfile.mkdtemp()
    try:
        # Copy selected files to temporary directory
        for file in selected_files:
            shutil.copy2(os.path.join(pdf_dir, file), temp_dir)
        
        # Load only the selected PDF files
        loader = PyPDFDirectoryLoader(temp_dir)
        documents = loader.load()
        chunks = split_text(documents)
        
        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
        
        # Benchmark the database creation
        start_time = time.time()
        db = create_db(chunks, embeddings)
        ingestion_time = time.time() - start_time
        del db  # Remove reference to close any open resources
        
        return {
            'num_docs': num_docs,
            'time_seconds': ingestion_time,
            'time_per_doc': ingestion_time / num_docs
        }
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def run_benchmark_trials(pdf_dir, num_docs, num_trials=5):
    """Run multiple trials and aggregate results."""
    results = []
    print(f"\nRunning {num_trials} trials for {num_docs} document{'s' if num_docs > 1 else ''}...")
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}")
        result = benchmark_ingestion(pdf_dir, num_docs)
        if result:
            results.append(result)
            print(f"Time: {result['time_seconds']:.2f} seconds")
    
    if results:
        df = pd.DataFrame(results)
        if os.path.exists(OUTPUT_FILE):
            df_existing = pd.read_csv(OUTPUT_FILE)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Results saved to {OUTPUT_FILE}")

def preview_document_selection(pdf_dir, num_docs):
    """Preview which documents would be selected for a given number of documents."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    selected_files = random.sample(pdf_files, num_docs)
    
    print(f"\nDocuments selected for {num_docs} document{'s' if num_docs > 1 else ''}:")
    for i, file in enumerate(selected_files, 1):
        print(f"{i}. {file}")

def query_document(db, query_text):
    """Query the vector database and return response."""
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find relevant information."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Measure response generation time
    start_time = time.time()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
    )
    response_time = time.time() - start_time
    print(f"Response generated in {response_time:.2f} seconds.")
    
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return f"Response: {chat_completion.choices[0].message.content}\nSources: {sources}"
    # return f"Response: {response}\nSources: {sources}"

def main():
    """Main function to handle both ingestion and querying."""
    load_dotenv()
    
    # Initialize global variables
    global client, embeddings
    
    # Initialize embeddings model once
    embeddings = SentenceTransformerEmbeddings(
        SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    )
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    while True:
        print("\nOptions:")
        print("1. Ingest documents")
        print("2. Ask a question")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            pdf_dir = input("Enter the PDF directory path: ")
            if not os.path.exists(pdf_dir):
                print("Directory does not exist!")
                continue
                
            num_docs = int(input("Enter number of documents to process: "))
            num_trials = int(input("Enter number of trials (default 5): ") or "5")
            
            # Run the benchmark trials
            run_benchmark_trials(pdf_dir, num_docs, num_trials)
            
            # Load documents for querying
            loader = PyPDFDirectoryLoader(pdf_dir)
            documents = loader.load()
            chunks = split_text(documents)
            db = create_db(chunks, embeddings)
            
        elif choice == "2":
            if not 'db' in locals():
                print("Please ingest documents first (Option 1)")
                continue
                
            query = input("\nEnter your question: ")
            if query.lower() in ['quit', 'exit']:
                break
                
            print("\nSearching for answer...")
            response = query_document(db, query)
            print("\nAnswer:", response)
            
        elif choice == "3":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
