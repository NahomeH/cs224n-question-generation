"""
General-purpose response generator for the language model.
"""

import torch
import logging
from src.models.model import ModelLoader

# Configure logging
logger = logging.getLogger(__name__)

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
    
    def generate_response(self, prompt, max_new_tokens=100, temperature=0.7, 
                          top_p=0.9, top_k=50, repetition_penalty=1.0):
        """
        Generate a basic response for any prompt with advanced parameters.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Randomness control (higher = more random, lower = more deterministic)
            top_p: Nucleus sampling parameter (keep tokens with cumulative probability < top_p)
            top_k: Keep only top k tokens with highest probability
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If model or tokenizer not loaded or generation fails
        """
        try:
            if self.model_loader.model is None or self.model_loader.tokenizer is None:
                raise RuntimeError("Model and tokenizer must be loaded before generating responses")
                
            inputs = self.model_loader.tokenizer(prompt, return_tensors="pt").to(self.model_loader.device)
            
            with torch.no_grad():
                outputs = self.model_loader.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.model_loader.tokenizer.pad_token_id,
                    eos_token_id=self.model_loader.tokenizer.eos_token_id
                )

            generated_text = self.model_loader.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the response part, removing the original prompt
            if generated_text.startswith(prompt):
                return generated_text[len(prompt):].strip()
            else:
                return generated_text.strip()
                
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during text generation")
            raise RuntimeError("GPU memory exceeded during generation. Try reducing max_new_tokens.")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory during text generation")
                raise RuntimeError("GPU memory exceeded during generation. Try reducing max_new_tokens.")
            else:
                logger.error(f"Error during text generation: {str(e)}")
                raise RuntimeError(f"Error during text generation: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during text generation: {str(e)}")
            raise RuntimeError(f"Unexpected error during text generation: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Create a response generator
    generator = ResponseGenerator()
    
    # Generate a simple response
    prompt = """Generate a brief description of heart failure symptoms."""
    result = generator.generate_response(prompt, max_new_tokens=250)
    print("Response:\n", result) 