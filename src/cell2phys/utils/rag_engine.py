import os
import glob
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from functools import lru_cache
from src.cell2phys.config import Config

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RAGController:
    """
    Manages knowledge retrieval using FAISS vector search.
    """

    def __init__(self):
        print("   Initializing RAG Knowledge Engine...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = Config
        
        # 1. Load Embeddings
        print(f"   Using Device: {self.device}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        
        # 2. Load or Create Index
        self.db = self._load_or_create_index()
        
        self._cache = {}

    def _load_or_create_index(self):
        """Loads FAISS index from disk or creates an empty one."""
        index_path = self.config.VECTOR_STORE_PATH
        
        if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
            print(f"   Loading existing knowledge base from {index_path}...")
            try:
                # Required for loading local FAISS index
                return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True) 
            except Exception as e:
                print(f"⚠️ Index load failed ({e}), creating new one.")
                return self._create_empty_index()
        else:
            print("   No existing index found. Creating empty knowledge base.")
            return self._create_empty_index()

    def _create_empty_index(self):
        """Creates a minimal index with one initialization document to initialize FAISS."""
        # FAISS requires at least one document to init
        init_doc = Document(page_content="System Initialized.", metadata={"source": "system"})
        return FAISS.from_documents([init_doc], self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Retrieves top-k context strings.
        """
        # Cache check
        if query in self._cache:
            return self._cache[query]
            
        try:
            docs = self.db.similarity_search(query, k=k)
            context = "\n".join([f"- {d.page_content}" for d in docs])
            
            # Update cache
            self._cache[query] = context
            return context
        except Exception as e:
            print(f"⚠️ Retrieval failed: {e}")
            return "No context available."

    def add_document(self, text: str, source: str = "manual"):
        """Adds a single document to the index and saves immediately."""
        doc = Document(page_content=text, metadata={"source": source})
        self.db.add_documents([doc])
        self.save_index()

    def save_index(self):
        """Persists the index to disk."""
        if not os.path.exists(self.config.VECTOR_STORE_PATH):
            os.makedirs(self.config.VECTOR_STORE_PATH)
        self.db.save_local(self.config.VECTOR_STORE_PATH)


if __name__ == "__main__":
    rag = RAGController()
    rag.add_document("Metformin activates AMPK to reduce hepatic glucose production.")
    print(rag.retrieve("How does metformin work?"))
