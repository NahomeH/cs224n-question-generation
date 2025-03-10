"""
Prompt templates for medical question generation.

This module provides standardized prompt templates for fine-tuning
and inference with the BioMistral model.
"""

from typing import Dict, Any, Optional
from string import Template


# Base template for USMLE question generation with BioMistral
BIOMISTRAL_GENERATION_TEMPLATE = """<s>[INST] Generate a detailed USMLE-style medical question based on this clinical case. 
The question should test understanding of key clinical findings and diagnosis.

Clinical Case:
$clinical_case

Create a single, well-formulated USMLE-style question with a clear focus on the key medical concepts. [/INST]"""

# Template for fine-tuning with explicit instructions
BIOMISTRAL_TRAINING_TEMPLATE = """<s>[INST] Generate a detailed USMLE-style medical question based on this clinical case. 
The question should test understanding of key clinical findings and diagnosis.

Clinical Case:
$clinical_case

Create a single, well-formulated USMLE-style question with a clear focus on the key medical concepts. [/INST]

$question"""

# Template for generating multiple-choice questions
BIOMISTRAL_MCQ_TEMPLATE = """<s>[INST] Generate a detailed USMLE-style multiple-choice question based on this clinical case. 
The question should test understanding of key clinical findings and diagnosis.

Clinical Case:
$clinical_case

Create a well-formulated USMLE-style multiple-choice question with 5 answer options (A-E),
clearly marking the correct answer, and providing a brief explanation for the correct answer. [/INST]"""

# Template for generating questions with specific difficulty
BIOMISTRAL_DIFFICULTY_TEMPLATE = """<s>[INST] Generate a $difficulty USMLE-style medical question based on this clinical case. 
The question should test understanding of key clinical findings and diagnosis.

Clinical Case:
$clinical_case

Create a single, well-formulated USMLE-style question with a clear focus on the key medical concepts.
Make sure the question is $difficulty level, suitable for $audience. [/INST]"""


def format_generation_prompt(clinical_case: str) -> str:
    """
    Format the standard generation prompt.
    
    Args:
        clinical_case: Clinical case description
        
    Returns:
        Formatted prompt
    """
    return Template(BIOMISTRAL_GENERATION_TEMPLATE).substitute(
        clinical_case=clinical_case
    )


def format_training_prompt(clinical_case: str, question: str) -> str:
    """
    Format the training prompt with example completion.
    
    Args:
        clinical_case: Clinical case description
        question: Target question to generate
        
    Returns:
        Formatted prompt with completion
    """
    return Template(BIOMISTRAL_TRAINING_TEMPLATE).substitute(
        clinical_case=clinical_case,
        question=question
    )


def format_mcq_prompt(clinical_case: str) -> str:
    """
    Format prompt for multiple-choice question generation.
    
    Args:
        clinical_case: Clinical case description
        
    Returns:
        Formatted MCQ prompt
    """
    return Template(BIOMISTRAL_MCQ_TEMPLATE).substitute(
        clinical_case=clinical_case
    )


def format_difficulty_prompt(
    clinical_case: str, 
    difficulty: str = "medium",
    audience: str = "medical students in clinical rotations"
) -> str:
    """
    Format prompt for generating questions with specific difficulty.
    
    Args:
        clinical_case: Clinical case description
        difficulty: Difficulty level ("easy", "medium", "hard")
        audience: Target audience for the question
        
    Returns:
        Formatted prompt with difficulty specification
    """
    return Template(BIOMISTRAL_DIFFICULTY_TEMPLATE).substitute(
        clinical_case=clinical_case,
        difficulty=difficulty,
        audience=audience
    )


# Example usage
if __name__ == "__main__":
    case = """A 45-year-old female presents with sudden onset chest pain radiating to the left arm. 
    She reports shortness of breath and nausea. Patient has a history of hypertension 
    and Type 2 diabetes. Vital signs show BP 160/95, HR 98, RR 22."""
    
    print("Generation Prompt Example:")
    print("-" * 80)
    print(format_generation_prompt(case))
    print("-" * 80) 