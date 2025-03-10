"""
QUEST-AI implementation for USMLE question generation.

This module implements the QUEST-AI approach for generating high-quality
USMLE-style questions from example questions using a prompt chaining technique.
"""

import re
import torch
from typing import Dict, Any, Optional, List, Union
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestAiGenerator:
    """
    Implementation of the QUEST-AI approach for generating USMLE-style questions.
    Uses a prompt chaining method with explanation generation and refinement.
    """
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize the QUEST-AI generator.
        
        Args:
            model: Loaded language model
            tokenizer: Tokenizer for the model
            device: Device the model is loaded on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_question(
        self,
        example_question: dict,
        temperature: float = 0.7,
        explanation_temp: float = 0.6,
        max_new_tokens: int = 1024,
        step_level: str = "Step 2 CK"
    ) -> dict:
        """
        Generate a USMLE-style question using the QUEST-AI prompt chaining approach.
        
        Args:
            example_question: Dict containing sample question with 'question', 'answer', and 'options'
            temperature: Controls randomness for final generation
            explanation_temp: Temperature for the explanation generation step
            max_new_tokens: Maximum new tokens to generate
            step_level: USMLE step level (e.g., "Step 1", "Step 2 CK", "Step 3")
            
        Returns:
            Dict containing the generated question, options, and answer
        """
        from src.utils.prompt_templates import format_explanation_prompt, format_quest_chain_prompt
        
        # Extract example question components
        question_text = example_question['question']
        correct_letter = example_question['answer']
        options = example_question['options']
        step_level = example_question['meta_info']
        
        # STEP 1: Generate explanation of why the answer is correct
        logger.info("Generating explanation for the example question")
        explanation_prompt = format_explanation_prompt(
            question_text=question_text, 
            options=options, 
            correct_letter=correct_letter,
            step_level=step_level
        )
        
        explanation = self._generate_text(
            prompt=explanation_prompt,
            temperature=explanation_temp,
            max_new_tokens=max_new_tokens
        )
        
        # STEP 2: Generate new question using the explanation
        logger.info("Generating new question based on example and explanation")
        generation_prompt = format_quest_chain_prompt(
            question_text=question_text,
            options=options,
            correct_letter=correct_letter,
            explanation=explanation,
            step_level=step_level
        )
        
        new_question_text = self._generate_text(
            prompt=generation_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        # Parse the generated question into structured format
        parsed_question = self._parse_generated_question(new_question_text)
        
        return parsed_question

    def _generate_text(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        """Helper method to generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the content after [/INST]
        return generated_text.split("[/INST]")[-1].strip()

    def _parse_generated_question(self, text: str) -> dict:
        """
        Parse the generated question text into a structured format.
        
        Args:
            text: Raw generated text from model
            
        Returns:
            Dict containing question, options, and answer
        """
        result = {
            "question": "",
            "options": {},
            "answer": "",
            "explanation": ""
        }
        
        # Find where the options start and the question ends
        option_start_match = re.search(r'(?:\n|\s)([A-F])[\.\)]\s', text)
        if option_start_match:
            # Extract the question text (everything before the options)
            result["question"] = text[:option_start_match.start()].strip()
            options_section = text[option_start_match.start():].strip()
            
            # Extract the options
            option_pattern = re.compile(r'(?:^|\n|\s)([A-F])[\.\)]\s+(.*?)(?=(?:\n|\s)[A-F][\.\)]|$)', re.DOTALL)
            for match in option_pattern.finditer(options_section):
                letter, option_text = match.groups()
                result["options"][letter] = option_text.strip()
            
            # Find the correct answer if specified
            answer_pattern = re.compile(r'(?:correct answer is|the answer is)[:\s]*([A-F])', re.IGNORECASE)
            answer_match = answer_pattern.search(text)
            if answer_match:
                result["answer"] = answer_match.group(1)
                
            # Extract explanation if present
            explanation_pattern = re.compile(r'(?:explanation:|discussion:|the correct answer is.*?because)', re.IGNORECASE)
            explanation_match = explanation_pattern.search(text)
            if explanation_match:
                result["explanation"] = text[explanation_match.start():].strip()
        
        return result