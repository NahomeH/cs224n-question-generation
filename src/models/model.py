"""
Core model implementation for USMLE question generation.

This module provides functionality to generate high-quality USMLE-style 
medical questions from clinical case descriptions.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, List, Union
import logging
import os
import hashlib
from pathlib import Path

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
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,
        device: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the medical question generator.
        
        Args:
            model_name: HuggingFace model identifier
            load_in_4bit: Whether to load model in 4-bit precision (saves the most memory)
            load_in_8bit: Whether to load model in 8-bit precision (better quality than 4-bit)
            device: Device to load the model on (default: auto-detect)
            use_cache: Whether to use model caching
            cache_dir: Directory to store cached models (default: ./model_cache)
        """
        # Device selection
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Caching settings
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "model_cache")
        
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
        
    def _get_cache_path(self, model_name, load_in_4bit, load_in_8bit):
        """Generate a unique cache path based on model configuration"""
        # Create a unique identifier based on model settings
        config_string = f"{model_name}_{load_in_4bit}_{load_in_8bit}_{self.device}"
        config_hash = hashlib.md5(config_string.encode()).hexdigest()[:10]
        
        # Create a safe filename
        safe_model_name = model_name.replace('/', '_')
        cache_path = os.path.join(self.cache_dir, f"{safe_model_name}_{config_hash}.pt")
        
        return cache_path
    
    def _load_model(self, model_name, load_in_4bit, load_in_8bit):
        """Helper method to load the model with appropriate quantization"""
        # Check for cached model if caching is enabled
        cache_path = None
        if self.use_cache:
            cache_path = self._get_cache_path(model_name, load_in_4bit, load_in_8bit)
            if os.path.exists(cache_path):
                try:
                    logger.info(f"Loading model from cache: {cache_path}")
                    self.model = torch.load(cache_path, map_location=self.device)
                    logger.info("Model loaded from cache successfully")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}. Will load from scratch.")
        
        # Create offload folder for disk offloading if needed
        offload_folder = os.path.join(self.cache_dir, "offload")
        os.makedirs(offload_folder, exist_ok=True)
        
        try:
            # GPU path (CUDA)
            if self.device == "cuda":
                logger.info("Loading model on GPU...")
                
                # T4 GPU optimized loading
                if load_in_8bit:
                    try:
                        logger.info("Attempting 8-bit quantization (good for T4 GPUs)...")
                        import bitsandbytes as bnb
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            load_in_8bit=True,
                            device_map="auto",
                            offload_folder=offload_folder,
                            torch_dtype=torch.float16
                        )
                        logger.info("Model loaded with 8-bit quantization successfully")
                    except (ImportError, ValueError) as e:
                        logger.warning(f"8-bit quantization failed: {e}")
                        load_in_8bit = False
                        # Fall through to standard loading
                    except Exception as e:
                        logger.warning(f"8-bit loading failed with unexpected error: {e}")
                        load_in_8bit = False
                        # Fall through to standard loading
                
                # 4-bit quantization attempt
                if load_in_4bit and not load_in_8bit:
                    try:
                        logger.info("Attempting 4-bit quantization...")
                        import bitsandbytes as bnb
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            load_in_4bit=True,
                            device_map="auto",
                            offload_folder=offload_folder,
                            torch_dtype=torch.float16
                        )
                        logger.info("Model loaded with 4-bit quantization successfully")
                    except (ImportError, ValueError) as e:
                        logger.warning(f"4-bit quantization failed: {e}")
                        # Fall through to standard loading
                    except Exception as e:
                        logger.warning(f"4-bit loading failed with unexpected error: {e}")
                        # Fall through to standard loading
                
                # Standard loading (fallback that always works)
                if not hasattr(self, 'model') or self.model is None:
                    logger.info("Using standard float16 loading (most compatible)...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto",
                        offload_folder=offload_folder,
                        torch_dtype=torch.float16
                    )
                    logger.info("Model loaded with standard precision")
                    
            # CPU path
            else:
                logger.info("Loading model on CPU. This may use significant RAM...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map={"": self.device},
                    low_cpu_mem_usage=True
                )
                logger.info("Model loaded on CPU successfully")
            
            # Cache the loaded model if caching is enabled
            if self.use_cache and cache_path:
                try:
                    logger.info(f"Caching model to: {cache_path}")
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    torch.save(self.model, cache_path)
                    logger.info("Model cached successfully")
                except Exception as e:
                    logger.warning(f"Failed to cache model: {e}")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load model {model_name}. See error above.")

    def get_quest_generator(self):
        """
        Create and return a QuestGenerator instance using this model.
        
        Returns:
            QuestGenerator instance configured with this model
        """
        from src.models.quest_generator import QuestionGenerator
        return QuestionGenerator(self.model, self.tokenizer, self.device)
            
def test_cpu_model_loading():
    """Simple test to verify model loads on CPU."""
    print("Testing model loading on CPU...")
    
    try:
        # Using a smaller model for testing
        small_model = "google/flan-t5-small"  # Much smaller than BioMistral

        # First run - will create cache
        print("First run - should load model from scratch and cache it")
        generator1 = MedicalQuestionGenerator(
            model_name=small_model,
            device="cpu", 
            use_cache=True,
            cache_dir="./test_model_cache",
            load_in_8bit=False,  # Can't use 8-bit on CPU
            load_in_4bit=False   # Can't use 4-bit on CPU
        )
        
        # Check basic properties
        print("✓ Model loaded successfully")
        print(f"✓ Model type: {type(generator1.model).__name__}")
        
        # Verify it's actually on CPU
        device = next(generator1.model.parameters()).device
        print(f"✓ Model is on: {device}")
        
        # Second run - should use cache
        print("\nSecond run - should load model from cache")
        generator2 = MedicalQuestionGenerator(
            model_name=small_model,
            device="cpu", 
            use_cache=True,
            cache_dir="./test_model_cache",
            load_in_8bit=False,
            load_in_4bit=False
        )
        
        print("✓ Second model loaded")
        
        # Test disabling cache
        print("\nThird run - with caching disabled")
        generator3 = MedicalQuestionGenerator(
            model_name=small_model,
            device="cpu", 
            use_cache=False,
            load_in_8bit=False,
            load_in_4bit=False
        )
        
        print("✓ Third model loaded without cache")
        print("CPU loading test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

if __name__ == "__main__":
    # Uncomment to test model loading and caching
    test_cpu_model_loading()