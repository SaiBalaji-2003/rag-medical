import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, UPLOAD_DIR, VECTORDB_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Chunking configuration
CHUNK_CONFIG = {
    "min_tokens": 200,
    "max_tokens": 500,
    "overlap_tokens": 50,
    "chunk_strategy": "sentence_aware"  # Options: 'fixed', 'sentence_aware', 'paragraph_aware'
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "batch_size": 32,
    "normalize_embeddings": True
}

# Vector store configuration
VECTORDB_CONFIG = {
    "collection_name": "icd10_documents",
    "distance_metric": "cosine",
    "top_k_default": 5,
    "max_results": 20
}

# LLM configuration
LLM_CONFIG = {
    "model_name": "microsoft/DialoGPT-medium",
    "fallback_model": "distilgpt2",
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "use_cache": True
}

# UI configuration
UI_CONFIG = {
    "page_title": "RAG Medical Q&A System",
    "page_icon": "🏥",
    "layout": "wide",
    "sidebar_width": 300,
    "max_file_size": 10  # MB
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "rag_system.log"
}