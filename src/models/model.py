import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class MedicalQuestionGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
    def generate_question(self, text: str) -> str:
        # Create a more specific prompt
        prompt = f"""Generate a detailed medical question based on this case:
        
        Context: {text}
        Task: Create a single USMLE-style question that tests understanding of the key clinical findings and diagnosis.
        Question:"""
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.device)
        
        # Generate with adjusted parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5,          # Increased for better search
                temperature=0.8,       # Slightly increased for more creativity
                do_sample=True,
                top_p=0.9,            # Added nucleus sampling
                repetition_penalty=1.2 # Discourage repetition
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question

def test_generation():
    generator = MedicalQuestionGenerator()
    
    text = """A 45-year-old female presents with sudden onset chest pain radiating to the left arm. 
    She reports shortness of breath and nausea. Patient has a history of hypertension 
    and Type 2 diabetes. Vital signs show BP 160/95, HR 98, RR 22."""
    
    print("\nInput Text:")
    print(text)
    
    print("\nGenerating question...")
    question = generator.generate_question(text)
    
    print("\nGenerated Question:")
    print(question)

if __name__ == "__main__":
    test_generation()