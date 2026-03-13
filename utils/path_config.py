"""
Configuration for paths.
"""

import pathlib
from typing import Dict, Optional

# Root directories
ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'data'
CHROMA_PERSIST_DIR = DATA_DIR / 'chroma'