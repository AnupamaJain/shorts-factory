import os
import pickle
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

class ShortsRAG:
    def __init__(self, persist_path="agents/bm25_retriever.pkl", data_dir="inputs/rag_data"):
        self.persist_path = persist_path
        self.data_dir = data_dir
        self.retriever = None

    def load_and_ingest(self):
        """Loads text files from the data directory and creates a BM25 Retriever."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory {self.data_dir}. Please add your .txt files here.")
            return

        loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()

        if not docs:
            print("No documents found in the data directory.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Create BM25 retriever
        self.retriever = BM25Retriever.from_documents(splits)
        self.retriever.k = 3
        
        # Save to disk
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(self.retriever, f)
            
        print(f"Ingested {len(splits)} chunks and saved BM25 retriever to {self.persist_path}.")

    def get_retriever(self, k=3):
        """Returns the BM25 retriever interface."""
        if not self.retriever:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "rb") as f:
                    self.retriever = pickle.load(f)
            else:
                raise ValueError("Retriever not initialized. Run load_and_ingest() first.")
        
        self.retriever.k = k
        return self.retriever

if __name__ == "__main__":
    # Test initialization and ingestion
    rag = ShortsRAG()
    rag.load_and_ingest()
