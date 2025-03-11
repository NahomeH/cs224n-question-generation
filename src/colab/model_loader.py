"""
Model loading module for medical question generation in Google Colab.

This module provides functionality to load the MedicalQuestionGenerator
model in a Colab environment.
"""

import sys
import json
from pathlib import Path
import torch
import os

# Make sure the current directory is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import
from src.models.model import MedicalQuestionGenerator

def load_model(
    model_name="BioMistral/BioMistral-7B",
    load_in_4bit=False, 
    load_in_8bit=True,
    device=None,
    use_cache=True,
    cache_dir="./model_cache"
):
    """
    Load the MedicalQuestionGenerator model.
    
    Args:
        model_name: HuggingFace model identifier
        load_in_4bit: Whether to load model in 4-bit precision (saves the most memory)
        load_in_8bit: Whether to load model in 8-bit precision (better quality than 4-bit)
        device: Device to load the model on (default: auto-detect)
        use_cache: Whether to use model caching
        cache_dir: Directory to store cached models (default: ./model_cache)
        
    Returns:
        MedicalQuestionGenerator instance
    """
    print(f"Loading model {model_name}...")
    generator = MedicalQuestionGenerator(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device=device,
        use_cache=use_cache,
        cache_dir=cache_dir
    )
    
    print("Model loaded successfully!")
    return generator

def get_quest_generator(generator):
    """
    Get the QuestAiGenerator from a loaded model.
    
    Args:
        generator: MedicalQuestionGenerator instance
        
    Returns:
        QuestAiGenerator instance
    """
    return generator.get_quest_generator()

def load_example_question(file_path):
    """
    Load an example question from a JSON file.
    
    Args:
        file_path: Path to the example question JSON file
        
    Returns:
        Dictionary containing the example question
    """
    print(f"Loading example question from {file_path}")
    with open(file_path, 'r') as f:
        example_question = json.load(f)
    return example_question

if __name__ == "__main__":
    # Example usage
    model = load_model()
    quest_generator = get_quest_generator(model)
    
    # Get the path to the example question
    input_file = str(Path(__file__).parent.parent.parent / "example_question.json")
    example_question = load_example_question(input_file)
    
    print("Model and generator loaded successfully!")