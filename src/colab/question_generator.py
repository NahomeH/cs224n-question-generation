"""
Question generator module for medical questions in Google Colab.

This module provides functionality to generate new USMLE-style
medical questions based on examples and explanations.
"""

import sys
import json
from pathlib import Path
import torch

# Make sure the current directory is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import local modules
from src.utils.prompt_templates import format_quest_chain_prompt
from src.colab.model_loader import load_model, get_quest_generator, load_example_question
from src.colab.explanation_generator import generate_explanation

def generate_question_from_explanation(
    quest_generator,
    example_question,
    explanation,
    temperature=0.7,
    max_new_tokens=1024
):
    """
    Generate a new USMLE-style question using the QUEST-AI prompt chaining approach.
    
    Args:
        quest_generator: QuestAiGenerator instance
        example_question: Dictionary containing the example question
        explanation: Generated explanation from the explanation generator
        temperature: Controls randomness for final generation
        max_new_tokens: Maximum new tokens to generate
        
    Returns:
        Generated question text and parsed question object
    """
    # Extract example question components
    question_text = example_question['question']
    correct_letter = example_question['answer']
    options = example_question['options']
    step_level = example_question.get('meta_info', 'Step 2 CK')
    
    # Generate question prompt
    print("Generating question prompt using explanation...")
    generation_prompt = format_quest_chain_prompt(
        question_text=question_text,
        options=options,
        correct_letter=correct_letter,
        explanation=explanation,
        step_level=step_level
    )
    
    # Generate new question
    print("Generating new question...")
    new_question_text = quest_generator._generate_text(
        prompt=generation_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )
    
    print("-" * 80)
    print("Generated Question:")
    print("-" * 80)
    print(new_question_text)
    print("-" * 80)
    
    # Return the raw text for now (parsing could be added in a separate function)
    return new_question_text

def save_results(explanation, new_question, output_file="generated_output.json"):
    """
    Save generated explanation and question to a JSON file.
    
    Args:
        explanation: Generated explanation text
        new_question: Generated question text
        output_file: File path to save the results
    """
    results = {
        "explanation": explanation,
        "generated_question": new_question
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    model = load_model()
    quest_generator = get_quest_generator(model)
    
    # Get the path to the example question
    input_file = str(Path(__file__).parent.parent.parent / "example_question.json")
    example_question = load_example_question(input_file)
    
    # Generate explanation
    explanation = generate_explanation(quest_generator, example_question)
    
    # Generate new question
    new_question = generate_question_from_explanation(
        quest_generator, 
        example_question, 
        explanation
    )
    
    # Save results
    save_results(explanation, new_question)