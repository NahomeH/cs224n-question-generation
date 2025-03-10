#!/usr/bin/env python3
"""
Simple script to run a QuestAiGenerator on an example question.
"""

import json
import sys
import os
from pathlib import Path

# Make sure the current directory is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import
from src.models.model import MedicalQuestionGenerator

def main():
    # Load the example question
    input_file = str(Path(__file__).parent.parent / "example_question.json")
    print(f"Loading example question from {input_file}")
    with open(input_file, 'r') as f:
        example_question = json.load(f)
    
    # Initialize the model (BioMistral by default)
    print("Loading model...")
    generator = MedicalQuestionGenerator(
        use_cache=True,
        cache_dir="./model_cache"
    )
    
    # Get the QuestAiGenerator
    quest_generator = generator.get_quest_generator()
    
    # Generate a new question
    print("Generating new question...")
    new_question = quest_generator.generate_question(
        example_question=example_question,
        temperature=0.7,
        explanation_temp=0.6,
        max_new_tokens=1024,
        step_level="Step 2 CK"
    )
    
    # Print the result
    print("\nGenerated Question:")
    print("-" * 80)
    print(f"Question: {new_question['question']}")
    print("\nOptions:")
    for letter, option in new_question['options'].items():
        print(f"{letter}. {option}")
    
    if new_question.get('answer'):
        print(f"\nAnswer: {new_question['answer']}")
    
    if new_question.get('explanation'):
        print(f"\nExplanation: {new_question['explanation']}")
    
    # Save the generated question
    output_file = str(Path(__file__).parent.parent / "generated_question.json")
    with open(output_file, 'w') as f:
        json.dump(new_question, f, indent=2)
    
    print(f"\nSaved generated question to {output_file}")

if __name__ == "__main__":
    main() 