# Load/save json, jsonl, metadata, and index files
import json
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

import faiss
import numpy as np

import pathlib
from .embedding_config import get_version_file_path, get_version_directory

root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / 'data'

def load_chunks(file_path):
    chunks = []
    with open(str(data_dir / file_path), 'r', encoding='utf-8') as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks

def save_index(index, file_path, version=None):
    """
    Save the FAISS index to a file.
    
    Args:
        index (faiss.Index): The FAISS index to save.
        file_path (str): The path to save the index file (can be relative to data_dir or versioned).
        version (str, optional): Version identifier for versioned storage.
    """
    if version:
        # Use versioned path
        save_path = get_version_file_path(version, "faiss_index")
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Use legacy path
        save_path = data_dir / file_path
    
    faiss.write_index(index, str(save_path))
    print(f"Saved FAISS index to {save_path}")

def load_index(file_path, version=None):
    """
    Load a FAISS index from a file.
    
    Args:
        file_path (str): The path to the index file to load (can be relative to data_dir or versioned).
        version (str, optional): Version identifier for versioned loading.
        
    Returns:
        faiss.Index: The loaded FAISS index.
        
    Raises:
        FileNotFoundError: If the index file doesn't exist.
        RuntimeError: If the index file is corrupted or can't be loaded.
    """
    if version:
        # Use versioned path
        index_file_path = get_version_file_path(version, "faiss_index")
    else:
        # Use legacy path
        index_file_path = data_dir / file_path
    
    if not index_file_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_file_path}")
    
    try:
        return faiss.read_index(str(index_file_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index from {index_file_path}: {e}")

def save_metadata(data, file_path):
    """
    Save the metadata to a JSON file.

    Args:
        data (list): The metadata to save.
        file_path (str): The path to save the metadata file.
    """
    with open(str(data_dir / file_path), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_from_metadata(file_path, indices, key="text"):
    """
    Load the text content from the metadata JSON file.

    Args:
        file_path (str): The path to the metadata file.
        indices (list): List of indices to retrieve.
        key (str): The key to extract from metadata (default: "text").

    Returns:
        list: A list of text content from the metadata.
    """
    with open(str(data_dir / file_path), 'r', encoding='utf-8') as file:
        metadata = json.load(file)
    
    results = []
    for i in indices:
        if i < len(metadata):
            results.append(metadata[i][key])
        else:
            raise IndexError(f"Index {i} out of bounds for metadata with length {len(metadata)}")
    return results

def load_system_prompt(filename='system_prompt.txt'):
    """
    Load the system prompt from the filename in the root_directory.

    Returns:
        str: The content of the system prompt.
    """
    with open(str(data_dir / filename), 'r', encoding='utf-8') as file:
        return file.read()
    
def load_gemini_api_key():
    """
    Load the Gemini API key from environment variables or Streamlit secrets.
    
    Returns:
        str: The Gemini API key.
    """
    if 'GEMINI_API_KEY' in os.environ:
        return os.environ['GEMINI_API_KEY']
    else:
        return st.secrets["GEMINI_API_KEY"]
