import os


class Config:
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    # LLM Settings
    LLM_API_BASE = os.environ.get("CELL2PHYS_LLM_API", "http://localhost:8000/v1")
    LLM_API_KEY = os.environ.get("CELL2PHYS_LLM_KEY", "EMPTY")
    LLM_MODEL_NAME = os.environ.get(
        "CELL2PHYS_LLM_MODEL", "meta-llama/Meta-Llama-3-70B-Instruct"
    )

    # RAG Settings
    VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_store")
    KNOWLEDGE_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "raw_papers")

    # Simulation
    DT = 1.0
    TOTAL_TIME = 200
    N_PATIENTS = 20

    # ASC
    ASC_THRESHOLD = 0.5
    ASC_DIMENSION = 384

    # Physics constraints (physiological bounds)
    GLUCOSE_MIN = 0.0
    GLUCOSE_MAX = 600.0
    INSULIN_MIN = 0.0
    INSULIN_MAX = 500.0
    BETA_MASS_MIN = 0.01
    BETA_MASS_MAX = 3.0

    # LLM output bounds
    REGULATION_FACTOR_MIN = 0.1
    REGULATION_FACTOR_MAX = 5.0
    SENSITIVITY_FACTOR_MIN = 0.1
    SENSITIVITY_FACTOR_MAX = 3.0
