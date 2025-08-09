# Load/save json, jsonl, metadata, and index files
import json
import os

import faiss
import numpy as np

import pathlib
root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / 'data'

def load_chunks(file_path):
    chunks = []
    with open(str(data_dir / file_path), 'r', encoding='utf-8') as file:
        for line in file:
            chunks.append(json.loads(line))
    return chunks

def save_index(index, file_path):
    """
    Save the FAISS index to a file.
    
    Args:
        index (faiss.Index): The FAISS index to save.
        file_path (str): The path to save the index file.
    """
    faiss.write_index(index, str(data_dir / file_path))

def load_index(file_path):
    """
    Load a FAISS index from a file.
    
    Args:
        file_path (str): The path to the index file to load.
        
    Returns:
        faiss.Index: The loaded FAISS index.
        
    Raises:
        FileNotFoundError: If the index file doesn't exist.
        RuntimeError: If the index file is corrupted or can't be loaded.
    """
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

def load_system_prompt(file_path='system_prompt.txt'):
    """
    Load the system prompt from a text file.

    Args:
        file_path (str): The path to the system prompt file.

    Returns:
        str: The content of the system prompt.
    """
    with open(str(data_dir / file_path), 'r', encoding='utf-8') as file:
        return file.read()