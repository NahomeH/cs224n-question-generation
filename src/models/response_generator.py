"""
General-purpose response generator for the language model.
"""

import torch
from src.models.model import ModelLoader

class ResponseGenerator:
    """
    A class for generating various types of responses using a loaded language model.
    This class separates the response generation logic from model loading.
    """
    
    def __init__(self, model_loader=None, model_id="biomistral/BioMistral-7B"):
        """
        Initialize the response generator.
        
        Args:
            model_loader: An existing ModelLoader instance or None to create a new one
            model_id: HuggingFace model ID to use if creating a new ModelLoader
        """
        if model_loader is None:
            self.model_loader = ModelLoader(model_id=model_id)
            self.model_loader.load_model()
        else:
            self.model_loader = model_loader
    
    def generate_response(self, prompt, max_new_tokens=100, temperature=0.7):
        """
        Generate a basic response for any prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Randomness control parameter
            
        Returns:
            Generated text response
        """
        if self.model_loader.model is None or self.model_loader.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded before generating responses")
            
        inputs = self.model_loader.tokenizer(prompt, return_tensors="pt").to(self.model_loader.device)
        
        with torch.no_grad():
            outputs = self.model_loader.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )

        generated_text = self.model_loader.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part, removing the original prompt
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        else:
            return generated_text.strip()


# Example usage
if __name__ == "__main__":
    # Create a response generator
    generator = ResponseGenerator()
    
    # Generate a simple response
    prompt = """Generate a brief description of heart failure symptoms."""
    result = generator.generate_response(prompt, max_new_tokens=250)
    print("Response:\n", result) 