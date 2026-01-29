import os
from enum import Enum

class ExecutionMode(Enum):
    REAL = "REAL"   # Uses real LLM server (OpenAI-compatible)

class SensitivityMode(Enum):
    LOW = "LOW"     # Standard variance
    HIGH = "HIGH"   # High variance

class Config:
    # --- Global Settings ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # --- Execution Control ---
    MODE = ExecutionMode.REAL
    
    # --- LLM Settings ---
    # Point this to your vLLM / Ollama / TGI server
    LLM_API_BASE = "http://localhost:8000/v1" 
    LLM_API_KEY = "EMPTY"  # Local servers usually don't check keys
    LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct" # Adjust based on your server
    
    # --- RAG Settings ---
    VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_store")
    KNOWLEDGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "raw_papers")
    
    # --- Simulation Params ---
    DT = 1.0
    TOTAL_TIME = 200
    
    @classmethod
    def get_llm_mode(cls):
        return cls.MODE

print(f"   [Config] Cell2Phys Config Loaded. Mode: {Config.MODE.value}")
