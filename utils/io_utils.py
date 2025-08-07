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

def save_metadata(data, file_path):
    """
    Save the metadata to a JSON file.

    Args:
        data (list): The metadata to save.
        file_path (str): The path to save the metadata file.
    """
    with open(str(data_dir / file_path), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
