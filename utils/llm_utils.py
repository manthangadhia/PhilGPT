# Load/save json, jsonl, metadata, and index files
import json
import os
from dotenv import load_dotenv
load_dotenv()

import pathlib

root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / 'data'

def load_system_prompt(filename='system_prompt.txt'):
    """Load the system prompt from the data directory."""
    with open(str(data_dir / filename), 'r', encoding='utf-8') as file:
        return file.read()


def load_gemini_api_key():
    """Load Gemini API key from env vars first, then Streamlit secrets."""
    if 'GEMINI_API_KEY' in os.environ:
        return os.environ['GEMINI_API_KEY']
    return os.getenv("GEMINI_API_KEY")