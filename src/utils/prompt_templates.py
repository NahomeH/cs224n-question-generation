"""
Simplified prompt templates for medical question generation.
"""

# Simple template for USMLE question generation
SIMPLE_USMLE_PROMPT = """<s>[INST] 
Generate a high-quality USMLE-style question that follows this format:

1. Start with a clinical vignette describing a patient's presentation
2. Include relevant history, physical exam findings, and lab/imaging results
3. Ask a clear clinical question
4. Provide 5 answer choices labeled A through E
5. Indicate which answer is correct
6. Provide a brief explanation of why the chosen answer is correct

The question should test medical knowledge and clinical reasoning at the USMLE Step 2 CK level.
[/INST]"""

def get_simple_prompt():
    """
    Return a simple prompt for USMLE question generation.
    
    Returns:
        String containing the prompt template
    """
    return SIMPLE_USMLE_PROMPT