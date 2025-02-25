import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.model import MedicalQuestionGenerator

def test_model():
    # Initialize the model
    generator = MedicalQuestionGenerator()
    
    # Sample medical text to test with
    medical_text = """
    A 45-year-old female presents with sudden onset chest pain radiating to the left arm. 
    She reports shortness of breath and nausea. Patient has a history of hypertension 
    and Type 2 diabetes. Vital signs show BP 160/95, HR 98, RR 22.
    """
    
    # Generate a question
    print("\nInput Text:")
    print(medical_text)
    
    print("\nGenerating question...")
    question = generator.generate_question(medical_text)
    
    print("\nGenerated Question:")
    print(question)

if __name__ == "__main__":
    test_model()