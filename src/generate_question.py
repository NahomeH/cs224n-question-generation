#!/usr/bin/env python3
import bitsandbytes as bnb
"""
Command-line tool to generate USMLE-style medical questions using BioMistral.

Example usage:
    python -m src.generate_question --text "A 45-year-old female presents with chest pain..."
    python -m src.generate_question --file medical_case.txt --quantize 8bit
"""

import argparse
import sys
from pathlib import Path
from src.models.model import MedicalQuestionGenerator


def main():
    """Command-line interface for generating USMLE-style medical questions."""
    parser = argparse.ArgumentParser(
        description="Generate USMLE-style medical questions using BioMistral",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add input arguments - either text or file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Medical text to generate a question for")
    input_group.add_argument("--file", type=str, help="File containing medical text")
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str,
        default="BioMistral/BioMistral-7B",
        help="HuggingFace model identifier"
    )
    
    # Quantization options
    parser.add_argument(
        "--quantize", 
        type=str, 
        choices=["4bit", "8bit", "none"], 
        default="none",
        help="Quantization level for model"
    )
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens to generate")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Get the input text
    if args.file:
        try:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"Error: File {args.file} not found", file=sys.stderr)
                sys.exit(1)
            text = file_path.read_text().strip()
        except Exception as e:
            print(f"Error reading file {args.file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text
    
    # Configure quantization
    load_in_4bit = args.quantize == "4bit"
    load_in_8bit = args.quantize == "8bit"
    
    if load_in_4bit or load_in_8bit:
        try:
            print(f"Using {args.quantize} quantization...")
        except ImportError:
            print("Warning: bitsandbytes not installed. Falling back to full precision.")
            load_in_4bit = False
            load_in_8bit = False
    
    # Initialize generator
    print(f"Using model: {args.model}")
    generator = MedicalQuestionGenerator(
        model_name=args.model,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit
    )
    
    # Configure generation parameters
    generation_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    # Print input text
    print("\nInput Text:")
    print("-" * 80)
    print(text)
    print("-" * 80)
    
    # Generate and print question
    print("\nGenerating question...")
    question = generator.generate_question(
        text, 
        max_new_tokens=args.max_tokens,
        generation_params=generation_params
    )
    
    print("\nGenerated Question:")
    print("-" * 80)
    print(question)
    print("-" * 80)


if __name__ == "__main__":
    main() 