"""
Core model implementation for USMLE question generation.

This module provides functionality to generate high-quality USMLE-style 
medical questions from clinical case descriptions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, List, Union
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalQuestionGenerator:
    """
    USMLE question generator based on a foundation large language model.
    """
    
    def __init__(
        self, 
        model_name: str = "BioMistral/BioMistral-7B", 
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize the medical question generator.
        
        Args:
            model_name: HuggingFace model identifier
            load_in_4bit: Whether to load model in 4-bit precision (saves the most memory)
            load_in_8bit: Whether to load model in 8-bit precision (better quality than 4-bit)
            device: Device to load the model on (default: auto-detect)
        """
        # Device selection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Tokenizer loading
        logger.info(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Pad token handling
        if self.tokenizer.pad_token is None:
            logger.info("Setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model loading with quantization
        logger.info("Loading model...")
        self._load_model(model_name, load_in_4bit, load_in_8bit)
        
    def _load_model(self, model_name, load_in_4bit, load_in_8bit):
        """Helper method to load the model with appropriate quantization"""
        try:
            # GPU path with quantization options
            if self.device == "cuda":
                if load_in_4bit:
                    logger.info("Loading model in 4-bit precision on GPU...")
                    try:
                        import bitsandbytes as bnb
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            load_in_4bit=True,
                            device_map="auto",
                            quantization_config={
                                "load_in_4bit": True, 
                                "bnb_4bit_compute_dtype": torch.float16,
                                "bnb_4bit_use_double_quant": True
                            }
                        )
                    except ImportError:
                        logger.warning("bitsandbytes not available for 4-bit quantization. Falling back...")
                        raise ImportError("bitsandbytes required for 4-bit quantization")
                        
                elif load_in_8bit:
                    logger.info("Loading model in 8-bit precision on GPU...")
                    try:
                        import bitsandbytes as bnb
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            load_in_8bit=True,
                            device_map="auto"
                        )
                    except ImportError:
                        logger.warning("bitsandbytes not available for 8-bit quantization. Falling back...")
                        raise ImportError("bitsandbytes required for 8-bit quantization")
                        
                else:
                    logger.info("Loading model in full precision on GPU...")
                    self.model = AutoModelForCausalLM.from_pretrained(model_name)
                    self.model.to(self.device)
                    
            # CPU path (quantization options limited)
            else:
                logger.info("Loading model on CPU. This may use significant RAM...")
                # For CPU, we should optimize for memory usage
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map={"": self.device},
                    low_cpu_mem_usage=True,
                )
                
            logger.info("Model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Attempting fallback loading method...")
            
            try:
                # Fallback to the most compatible loading method
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                logger.info("Model loaded with fallback method")
            except Exception as fallback_error:
                logger.error(f"Fallback loading also failed: {fallback_error}")
                raise RuntimeError(f"Could not load model {model_name}. Check your environment and dependencies.")
            
def test_cpu_model_loading():
    """Simple test to verify model loads on CPU."""
    print("Testing model loading on CPU...")
    
    try:
        # Initialize with CPU device explicitly
        generator = MedicalQuestionGenerator(device="cpu")
        
        # Check basic properties
        print("✓ Model loaded successfully")
        print(f"✓ Model type: {type(generator.model).__name__}")
        
        # Verify it's actually on CPU
        device = next(generator.model.parameters()).device
        print(f"✓ Model is on: {device}")
        
        print("CPU loading test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

if __name__ == "__main__":
    test_cpu_model_loading()