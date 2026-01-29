import numpy as np
import os
import json
from typing import Optional, Dict, List, Any

# Try importing FAISS and HuggingFaceEmbeddings
try:
    import faiss
    from langchain_huggingface import HuggingFaceEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ Required packages not found. ASC will run in exact match mode.")

from src.cell2phys.config import Config

class AdaptiveSemanticCache:
    """
    Adaptive Semantic Caching (ASC) Engine.
    Maps biological state vectors to kinetic parameters using vector search (FAISS).
    Persists mappings to disk for acceleration.
    """
    
    def __init__(self, dimension: int = 384, threshold: float = 0.1):
        """
        Args:
            dimension: Dimension of the projected state embedding.
            threshold: Similarity threshold (L2 distance or Cosine) for cache hits.
        """
        self.cache_dir = os.path.join(Config.PROJECT_ROOT, "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.index_path = os.path.join(self.cache_dir, "asc_index.faiss")
        self.metadata_path = os.path.join(self.cache_dir, "asc_metadata.json")
        
        self.dimension = dimension
        self.threshold = threshold
        
        self.metadata: List[Dict[str, Any]] = [] # Stores actual parameter payloads
        
        if FAISS_AVAILABLE:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': device}
            )
            
        self.index = self._load_or_create_index()
        
    def _load_or_create_index(self):
        if FAISS_AVAILABLE and os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                print(f"🧠 [ASC] Loading Semantic Cache from {self.index_path}...")
                index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                return index
            except Exception as e:
                print(f"⚠️ [ASC] Index load failed: {e}. Resetting.")
                return self._create_new_index()
        else:
            return self._create_new_index()

    def _create_new_index(self):
        if FAISS_AVAILABLE:
            # L2 Distance Index for dense floats
            return faiss.IndexFlatL2(self.dimension)
        else:
            return None

    def _embed_state(self, state_context: str) -> np.ndarray:
        """
        Projects a string or state dict into the semantic embedding space.
        Uses a frozen encoder (e.g., all-MiniLM-L6-v2) to capture biological context.
        """
        if FAISS_AVAILABLE:
            embedding = np.array(self.encoder.embed_query(state_context), dtype='float32')
        else:
            # Fallback if encoder is unavailable
            embedding = np.zeros(self.dimension, dtype='float32')
            
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return (embedding / norm).reshape(1, -1)
        return embedding.reshape(1, -1)

    def retrieve(self, input_context: str) -> Optional[float]:
        """
        Query the cache for a semantic match.
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return None

        query_vec = self._embed_state(input_context)
        
        # Search
        distances, indices = self.index.search(query_vec, 1)
        
        min_dist = distances[0][0]
        idx = indices[0][0]
        
        if min_dist < self.threshold and idx != -1:
            # Cache Hit
            return self.metadata[idx]['value']
        else:
            # Cache Miss
            return None

    def store(self, input_context: str, value: float):
        """
        Commit a new mapping to memory.
        """
        if not FAISS_AVAILABLE or self.index is None:
            return
            
        vec = self._embed_state(input_context)
        
        # Add to index
        self.index.add(vec)
        self.metadata.append({'context_hash': hash(input_context), 'value': value})

    def save(self):
        """Persist cache metadata to disk securely."""
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=4)
            print(f"💾 [ASC] Cache saved ({self.index.ntotal} items).")

# Global Instance
asc_engine = AdaptiveSemanticCache()

