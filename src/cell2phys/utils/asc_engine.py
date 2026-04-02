import numpy as np
import os
import json
from typing import Optional, Dict, List, Any
from src.cell2phys.config import Config

# Optional FAISS + embeddings — system runs without them (exact-match fallback)
try:
    import faiss
    from langchain_huggingface import HuggingFaceEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class AdaptiveSemanticCache:
    """
    Adaptive Semantic Caching (ASC) engine.
    Semantic mode: FAISS index with sentence embeddings.
    Fallback mode: exact-match dictionary (no external deps).
    """

    def __init__(self):
        self.dimension = Config.ASC_DIMENSION
        self.threshold = Config.ASC_THRESHOLD
        self.cache_dir = os.path.join(Config.PROJECT_ROOT, "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.index_path = os.path.join(self.cache_dir, "asc_index.faiss")
        self.metadata_path = os.path.join(self.cache_dir, "asc_metadata.json")
        self.metadata: List[Dict[str, Any]] = []

        # Exact-match fallback dict (always available)
        self._exact: Dict[str, float] = {}

        if FAISS_AVAILABLE:
            self.encoder = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
            self.index = self._load_or_create_index()
            print(f"   [ASC] Semantic cache ready (threshold={self.threshold})")
        else:
            self.encoder = None
            self.index = None
            print("   [ASC] Running in exact-match fallback mode")

    # ----- index management -----
    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                idx = faiss.read_index(self.index_path)
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                print(f"   [ASC] Loaded {idx.ntotal} cached items")
                return idx
            except Exception:
                pass
        return faiss.IndexFlatL2(self.dimension)

    def _embed(self, text: str) -> np.ndarray:
        vec = np.array(self.encoder.embed_query(text), dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.reshape(1, -1)

    # ----- public API -----
    def retrieve(self, key: str) -> Optional[float]:
        # exact match first
        if key in self._exact:
            return self._exact[key]
        # semantic search
        if self.index is not None and self.index.ntotal > 0:
            qvec = self._embed(key)
            dists, idxs = self.index.search(qvec, 1)
            if dists[0][0] < self.threshold and idxs[0][0] != -1:
                return self.metadata[idxs[0][0]]["value"]
        return None

    def store(self, key: str, value: float):
        self._exact[key] = value
        if self.index is not None:
            vec = self._embed(key)
            self.index.add(vec)
            self.metadata.append({"key": key[:80], "value": value})

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w") as f:
                json.dump(self.metadata, f)
            print(f"   [ASC] Saved {self.index.ntotal} items to disk")

    @property
    def size(self) -> int:
        if self.index is not None:
            return self.index.ntotal
        return len(self._exact)


# Lazy singleton
_instance: Optional[AdaptiveSemanticCache] = None


def get_asc_engine() -> AdaptiveSemanticCache:
    global _instance
    if _instance is None:
        _instance = AdaptiveSemanticCache()
    return _instance


# Backwards-compatible global reference (created lazily on first import)
asc_engine = get_asc_engine()
