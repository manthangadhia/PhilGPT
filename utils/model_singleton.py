from sentence_transformers import SentenceTransformer
import pathlib
import os

root_dir = pathlib.Path(__file__).parent.parent
data_dir = root_dir / 'data'

class ModelSingleton:
    _instance = None
    _model = None
    _model_name = None

    def __new__(cls, model_name='all-MiniLM-L6-v2'):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Only initialize if we don't have a model or if the model name changed
        if self._model is None or self._model_name != model_name:
            self._model_name = model_name
            self._model = self._load_model(model_name)

    def _load_model(self, model_name):
        """
        Load the SentenceTransformer model from cache or download if not available.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            SentenceTransformer: The loaded model.
        """
        model_path = data_dir / model_name
        
        # Ensure data directory exists
        data_dir.mkdir(exist_ok=True)
        
        # Check if model is cached locally
        if model_path.exists() and os.listdir(model_path):
            try:
                # print(f"Loading cached model from {model_path}")
                return SentenceTransformer(str(model_path))
            except Exception as e:
                print(f"No cached model found: {e}")
                # print("Downloading from Hugging Face...")
        
        # Download model from Hugging Face and cache it
        try:
            print(f"Downloading model '{model_name}' from Hugging Face...")
            model = SentenceTransformer(model_name)
            
            # Save to cache
            print(f"Caching model to {model_path}")
            model.save(str(model_path))
            
            return model
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def get_model(self):
        """
        Get the singleton model instance.

        Returns:
            SentenceTransformer: The loaded model.
        """
        return self._model

    @classmethod
    def get_instance(cls, model_name='all-MiniLM-L6-v2'):
        """
        Get the singleton instance with the specified model.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            ModelSingleton: The singleton instance.
        """
        return cls(model_name)