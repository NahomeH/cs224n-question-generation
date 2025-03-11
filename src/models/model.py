import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Class for loading and managing an AI language model.
    """
    
    def __init__(self, model_id="biomistral/BioMistral-7B", use_gpu=True):
        """
        Initialize the model loader.
        
        Args:
            model_id (str): HuggingFace model identifier
            use_gpu (bool): Whether to use GPU if available
        """
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        
        # Simple device management
        self.use_gpu = use_gpu
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            if self.use_gpu and not torch.cuda.is_available():
                print("GPU requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
            print("Using CPU")
    
    def load_model(self):
        """
        Load the model and tokenizer with error handling.
        
        Returns:
            self: For method chaining
            
        Raises:
            RuntimeError: For model loading failures
            ValueError: For invalid configuration
            ConnectionError: For network issues
        """
        try:
            print(f"Loading model: {self.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with error handling
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"Model loaded successfully")
            return self
            
        except OSError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(f"Not enough GPU memory to load {self.model_id}. Try with a smaller model or use CPU.") from e
            elif "No such file or directory" in str(e) or "404" in str(e):
                raise ValueError(f"Model {self.model_id} not found. Check the model ID.") from e
            else:
                raise ConnectionError(f"Failed to download model: {e}") from e
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError(f"GPU out of memory when loading {self.model_id}. Try with a smaller model or use CPU.") from e
            else:
                raise RuntimeError(f"Error loading model: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading model: {str(e)}") from e