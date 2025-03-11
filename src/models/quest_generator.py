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
            Dict containing the raw generated explanation and question
        """
        from src.utils.prompt_templates import format_explanation_prompt, format_quest_chain_prompt
        
        # Extract example question components
        question_text = example_question['question']
        correct_letter = example_question['answer']
        options = example_question['options']
        step_level = example_question['meta_info']
        print(f"Example question: {example_question}")
        
        # STEP 1: Generate explanation of why the answer is correct
        logger.info("Generating explanation for the example question")
        explanation_prompt = format_explanation_prompt(
            question_text=question_text, 
            options=options, 
            correct_letter=correct_letter,
            step_level=step_level
        )
        logger.info(f"Explanation prompt: {explanation_prompt}")
        
        explanation = self._generate_text(
            prompt=explanation_prompt,
            temperature=explanation_temp,
            max_new_tokens=max_new_tokens
        )
        
        # Log the explanation
        logger.info(f"Generated explanation: {explanation}")
        
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
        
        # Log the new question
        logger.info(f"Generated new question (first 200 chars): {new_question_text[:200]}...")
        
        # Return raw outputs without parsing
        return {
            "raw_explanation": explanation,
            "raw_generated_question": new_question_text,
            "question": "See raw_generated_question for complete output",
            "options": {},
            "answer": "",
            "explanation": "See raw_explanation for complete output"
        }

    def _generate_text(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        """Helper method to generate text from a prompt."""
        logger.info(f"Tokenizing prompt of length {len(prompt)}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        logger.info(f"Starting generation with temperature={temperature}, max_new_tokens={max_new_tokens}")
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                logger.info(f"Generation completed. Output shape: {outputs.shape}")
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Decoded full response length: {len(generated_text)}")
                
                # Extract the content after [/INST]
                if "[/INST]" in generated_text:
                    result = generated_text.split("[/INST]")[-1].strip()
                    logger.info(f"Extracted text after [/INST], length: {len(result)}")
                    return result
                else:
                    logger.warning("No [/INST] tag found in response")
                    return generated_text
                
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                return f"Generation error: {str(e)}"