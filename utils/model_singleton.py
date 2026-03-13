import os
import pathlib
import threading
import importlib
from typing import Any, Dict, Optional

root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / "data"

_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _load_sentence_transformer(model_name: str):
    try:
        sentence_transformers = importlib.import_module("sentence_transformers")
        SentenceTransformer = sentence_transformers.SentenceTransformer
    except Exception as e:
        print(f"Error importing sentence_transformers: {e}")
        raise ImportError(
            "sentence_transformers is required for embeddings. "
            "Install with `pip install sentence-transformers`."
        ) from e

    model_path = data_dir / model_name
    data_dir.mkdir(exist_ok=True)

    if model_path.exists() and os.listdir(model_path):
        try:
            return SentenceTransformer(str(model_path))
        except Exception as e:
            print(f"Failed to load cached model from {model_path}: {e}")

    try:
        print(f"Downloading model '{model_name}' from Hugging Face...")
        model = SentenceTransformer(model_name)
        print(f"Caching model to {model_path}")
        model.save(str(model_path))
        return model
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def get_cached_model(model_name: str = "all-MiniLM-L6-v2"):
    """Return a cached embedding model instance for a given model name."""
    model = _MODEL_CACHE.get(model_name)
    if model is not None:
        return model

    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(model_name)
        if model is None:
            model = _load_sentence_transformer(model_name)
            _MODEL_CACHE[model_name] = model

    return model


def clear_cached_models(model_name: Optional[str] = None) -> None:
    """Clear one cached model or all cached models."""
    with _CACHE_LOCK:
        if model_name is None:
            _MODEL_CACHE.clear()
            return
        _MODEL_CACHE.pop(model_name, None)


class ModelSingleton:
    """Backward-compatible wrapper over the keyed model cache."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name

    def get_model(self):
        return get_cached_model(self._model_name)

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2"):
        return cls(model_name)