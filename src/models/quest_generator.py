"""
Simple USMLE question generator.
"""

import torch

class QuestionGenerator:
    """
    A simplified generator for USMLE-style medical questions.
    """
    
    def __init__(self, model, tokenizer, device):
        """
        Initialize the question generator.
        
        Args:
            model: Loaded language model
            tokenizer: Tokenizer for the model
            device: Device for model inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate_question(self, temperature=0.7, max_new_tokens=1024):
        """
        Generate a USMLE-style question with a simple prompt.
        
        Args:
            temperature: Controls randomness of generation
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            The generated question as a string
        """
        from src.utils.prompt_templates import get_simple_prompt
        
        prompt = get_simple_prompt()
        return self._generate_text(prompt, temperature, max_new_tokens)

    def _generate_text(self, prompt, temperature, max_new_tokens):
        """Generate text from a prompt using the model."""
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
            
            # Extract content after instruction tag if present
            if "[/INST]" in generated_text:
                return generated_text.split("[/INST]")[-1].strip()
            else:
                return generated_text