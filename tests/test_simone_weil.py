#!/usr/bin/env python3
"""
Quick test to see the actual content for Simone Weil query results.
"""

import sys
import pathlib
import json

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.retriever import Retriever
from utils.io_utils import load_chunks

def show_results_content():
    print("Testing 'Simone Weil taught geometry' query...")
    
    # Initialize retriever
    retriever = Retriever()
    
    # Load the chunks data
    chunks_data = load_chunks("transcript_chunks.jsonl")
    print(f"Loaded {len(chunks_data)} chunks")
    
    # Perform search
    query = "Simone Weil taught geometry"
    results = retriever.retrieve(query, k=3)
    
    print(f"\nQuery: '{query}'")
    print("="*50)
    
    for i, (chunk_idx, similarity_score) in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Chunk Index: {chunk_idx}")
        print(f"Similarity Score: {similarity_score:.4f}")
        
        if chunk_idx < len(chunks_data):
            chunk = chunks_data[chunk_idx]
            print(f"Episode: {chunk.get('episode_number', 'N/A')}")
            print(f"Title: {chunk.get('title', 'N/A')}")
            print(f"Chunk ID: {chunk.get('chunk_id', 'N/A')}")
            print(f"Content: {chunk.get('text', '')}")
            print("-" * 50)

if __name__ == "__main__":
    show_results_content()
