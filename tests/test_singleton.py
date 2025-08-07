#!/usr/bin/env python3
"""
Test script for ModelSingleton to verify it works correctly.
"""

import sys
import pathlib

# Add the project root to Python path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.model_singleton import ModelSingleton

def test_singleton():
    print("Testing ModelSingleton...")
    
    # Test 1: Create first instance
    print("\n1. Creating first instance...")
    model1 = ModelSingleton()
    print(f"Model 1 ID: {id(model1)}")
    print(f"Model 1 model name: {model1._model_name}")
    
    # Test 2: Create second instance (should be same object)
    print("\n2. Creating second instance...")
    model2 = ModelSingleton()
    print(f"Model 2 ID: {id(model2)}")
    print(f"Same instance? {model1 is model2}")
    
    # Test 3: Get model and test basic functionality
    print("\n3. Testing model functionality...")
    sentence_model = model1.get_model()
    test_sentences = ["This is a test sentence.", "This is another test."]
    embeddings = sentence_model.encode(test_sentences)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test 4: Using class method
    print("\n4. Testing class method...")
    model3 = ModelSingleton.get_instance()
    print(f"Model 3 ID: {id(model3)}")
    print(f"Same instance via class method? {model1 is model3}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_singleton()
