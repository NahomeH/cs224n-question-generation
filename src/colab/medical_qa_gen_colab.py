"""
Medical Question Generation for Google Colab.

This module provides a combined workflow for generating USMLE-style
medical questions in a Colab environment.
"""

import sys
import json
import os
from pathlib import Path
import torch
import argparse

# Make sure the current directory is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the components
from src.colab.model_loader import load_model, get_quest_generator, load_example_question
from src.colab.explanation_generator import generate_explanation
from src.colab.question_generator import generate_question_from_explanation, save_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate medical questions in Colab")
    
    parser.add_argument(
        "--model_name", 
        default="BioMistral/BioMistral-7B",
        help="HuggingFace model identifier"
    )
    
    parser.add_argument(
        "--input_file", 
        default=None,
        help="Path to the example question JSON file"
    )
    
    parser.add_argument(
        "--output_file", 
        default="generated_output.json",
        help="Path to save the generated output"
    )
    
    parser.add_argument(
        "--explanation_temp", 
        type=float,
        default=0.6,
        help="Temperature for explanation generation"
    )
    
    parser.add_argument(
        "--question_temp", 
        type=float,
        default=0.7,
        help="Temperature for question generation"
    )
    
    parser.add_argument(
        "--max_tokens", 
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--cache_dir", 
        default="./model_cache",
        help="Directory to store cached models"
    )
    
    return parser.parse_args()

def main():
    """Main function for the medical question generation workflow."""
    args = parse_arguments()
    
    # Determine the input file path
    if args.input_file is None:
        input_file = str(Path(__file__).parent.parent.parent / "example_question.json")
    else:
        input_file = args.input_file
    
    print(f"Using model: {args.model_name}")
    print(f"Input file: {input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Explanation temperature: {args.explanation_temp}")
    print(f"Question temperature: {args.question_temp}")
    print(f"Max tokens: {args.max_tokens}")
    
    # Step 1: Load the model
    print("\n--- Step 1: Loading model ---")
    model = load_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir
    )
    quest_generator = get_quest_generator(model)
    
    # Step 2: Load the example question
    print("\n--- Step 2: Loading example question ---")
    example_question = load_example_question(input_file)
    
    # Step 3: Generate explanation
    print("\n--- Step 3: Generating explanation ---")
    explanation = generate_explanation(
        quest_generator, 
        example_question, 
        temperature=args.explanation_temp,
        max_new_tokens=args.max_tokens
    )
    
    # Step 4: Generate new question
    print("\n--- Step 4: Generating new question ---")
    new_question = generate_question_from_explanation(
        quest_generator, 
        example_question, 
        explanation,
        temperature=args.question_temp,
        max_new_tokens=args.max_tokens
    )
    
    # Step 5: Save results
    print("\n--- Step 5: Saving results ---")
    save_results(explanation, new_question, args.output_file)
    
    print("\nDone! Your medical question has been generated.")

if __name__ == "__main__":
    main()