"""
Configuration for versioned embedding models and indices.
"""

import pathlib
from typing import Dict, Optional

# Root directories
ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'

# Versioned model configurations
EMBEDDING_VERSIONS = {
    "v1_miniLM": {
        "model_name": "all-MiniLM-L6-v2",
        "directory": "v1_miniLM",
        "description": "Initial MiniLM-L6-v2 embeddings",
        "embedding_dim": 384,
        "files": {
            "embeddings": "embeddings.npy",
            "faiss_index": "faiss_transcript_index.index",
            "model_cache": "model/all-MiniLM-L6-v2"
        }
    },
    # Template for future versions
    "v2_bge": {
        "model_name": "BAAI/bge-large-en-v1.5", 
        "directory": "v2_bge",
        "description": "BGE Large English v1.5 embeddings",
        "embedding_dim": 1024,
        "files": {
            "embeddings": "embeddings.npy",
            "faiss_index": "faiss_transcript_index.index",
            "model_cache": "model"
        }
    }
}

# Default version to use
DEFAULT_VERSION = "v1_miniLM"

def get_version_config(version: str = None) -> Dict:
    """Get configuration for a specific embedding version."""
    if version is None:
        version = DEFAULT_VERSION
    
    if version not in EMBEDDING_VERSIONS:
        raise ValueError(f"Unknown version '{version}'. Available: {list(EMBEDDING_VERSIONS.keys())}")
    
    return EMBEDDING_VERSIONS[version]

def get_version_directory(version: str = None) -> pathlib.Path:
    """Get the data directory for a specific version."""
    config = get_version_config(version)
    return DATA_DIR / config["directory"]

def get_version_file_path(version: str, file_type: str) -> pathlib.Path:
    """Get the path to a specific file for a version."""
    config = get_version_config(version)
    version_dir = get_version_directory(version)
    
    if file_type not in config["files"]:
        raise ValueError(f"Unknown file type '{file_type}'. Available: {list(config['files'].keys())}")
    
    return version_dir / config["files"][file_type]

def list_available_versions() -> list:
    """List all available embedding versions."""
    return list(EMBEDDING_VERSIONS.keys())
