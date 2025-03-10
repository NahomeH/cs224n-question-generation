"""
Prompt templates for medical question generation.

This module provides standardized prompt templates
to generate USMLE-style questions.
"""

from typing import Dict, Any, Optional
from string import Template

# Template for explanation generation step in QUEST-AI approach
BIOMISTRAL_EXPLANATION_TEMPLATE = """<s>[INST] 
Here is an example question from the USMLE $step_level:

$question_text

Answer Choices:
$options_text

Why is ($correct_letter) "$correct_option" the correct answer and why are the other answers incorrect? Provide a detailed explanation. [/INST]"""

# Template for question generation using example and explanation
BIOMISTRAL_QUEST_CHAIN_TEMPLATE = """<s>[INST] 
Here is an example question, answer, and explanation from the USMLE $step_level:

Sample Question:
$question_text

Answer Choices:
$options_text

The correct answer is $correct_letter.

Answer & Explanation:
$explanation

Generate another question for the USMLE $step_level using a similar format. 
The new question should:
1. Include a detailed clinical vignette
2. Test understanding of key clinical findings and appropriate management
3. Include 5-6 answer choices labeled A through F
4. Clearly indicate which answer is correct
[/INST]"""


def format_explanation_prompt(question_text: str, options: dict, correct_letter: str, step_level: str = "Step 2 CK") -> str:
    """
    Format prompt for the explanation generation step.
    
    Args:
        question_text: Text of the example question
        options: Dictionary of options (letter -> text)
        correct_letter: The letter of the correct answer
        step_level: USMLE step level (e.g., "Step 1", "Step 2 CK", "Step 3")
        
    Returns:
        Formatted explanation prompt
    """
    options_text = "\n".join([f"{letter}. {text}" for letter, text in options.items()])
    correct_option = options[correct_letter]
    
    return Template(BIOMISTRAL_EXPLANATION_TEMPLATE).substitute(
        question_text=question_text,
        options_text=options_text,
        correct_letter=correct_letter,
        correct_option=correct_option,
        step_level=step_level
    )


def format_quest_chain_prompt(question_text: str, options: dict, correct_letter: str, explanation: str, step_level: str = "Step 2 CK") -> str:
    """
    Format prompt for the question generation step using explanation.
    
    Args:
        question_text: Text of the example question
        options: Dictionary of options (letter -> text)
        correct_letter: The letter of the correct answer
        explanation: Generated explanation of the correct answer
        step_level: USMLE step level (e.g., "Step 1", "Step 2 CK", "Step 3")
        
    Returns:
        Formatted prompt for generating a new question
    """
    options_text = "\n".join([f"{letter}. {text}" for letter, text in options.items()])
    
    return Template(BIOMISTRAL_QUEST_CHAIN_TEMPLATE).substitute(
        question_text=question_text,
        options_text=options_text,
        correct_letter=correct_letter,
        explanation=explanation,
        step_level=step_level
    )


# NOTE: Other templates and formatting functions are to be implemented in the future
# as additional question generation approaches are needed.