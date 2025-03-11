"""
Example script demonstrating how to use the ModelLoader and ResponseGenerator.
"""

from src.models.model import ModelLoader
from src.models.response_generator import ResponseGenerator

def main():
    """Generate responses using the model."""
    
    # Method 1: Create a response generator which creates its own model loader
    print("Creating response generator with default model...")
    generator = ResponseGenerator()
    
    # Generate a simple response
    print("\n=== Basic Response ===")
    prompt = "Generate a brief description of heart failure symptoms."
    result = generator.generate_response(prompt, max_new_tokens=250)
    print(result)
    
    # Method 2: Load model separately and pass to response generator
    print("\n=== Using custom model loader ===")
    # This approach allows reusing the same model for multiple generators
    # or configuring the model with specific parameters
    loader = ModelLoader(model_id="biomistral/BioMistral-7B", use_gpu=True)
    loader.load_model()
    
    custom_generator = ResponseGenerator(model_loader=loader)
    
    # Generate another response with custom parameters
    print("\n=== Custom Response with Different Parameters ===")
    prompt = "Explain what USMLE is and why it's important for medical students."
    response = custom_generator.generate_response(prompt, max_new_tokens=300, temperature=0.8)
    print(response)


if __name__ == "__main__":
    main() 