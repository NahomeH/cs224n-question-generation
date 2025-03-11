#!/usr/bin/env python3
"""
Fine-tuning script for USMLE question generation using QLoRA and contrastive semantic loss.
Uses preprocessed USMLE dataset to learn question semantics and structure.
"""

import argparse
import logging
import os
import sys
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Tuple, Union

# Default dataset path
DEFAULT_DATASET_PATH = "src/data/processed_data/preprocessed_usmle_dataset.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class USMLEQuestionTrainer:
    """
    Trainer class for fine-tuning models to generate USMLE-style questions.
    Uses QLoRA and contrastive semantic loss to learn question structure and semantics.
    """
    
    def __init__(
        self,
        base_model_name: str = "BioMistral/BioMistral-7B",
        semantic_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        output_dir: str = "./outputs",
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
        contrastive_loss_weight: float = 0.1,
        max_length: int = 1024,  # Increased for longer question formats
        add_eos: bool = True  # Whether to add EOS token to input prompts
    ):
        """
        Initialize the trainer with model configurations and hyperparameters.
        
        Args:
            base_model_name: Name of the base LLM model
            semantic_model_name: Name of the medical semantic model (PubMedBERT)
            output_dir: Directory to save outputs
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            lora_target_modules: List of modules to apply LoRA to
            contrastive_loss_weight: Weight of contrastive loss in total loss
            max_length: Maximum sequence length for inputs
            add_eos: Whether to add EOS token to input prompts
        """
        self.base_model_name = base_model_name
        self.semantic_model_name = semantic_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.add_eos = add_eos
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.contrastive_loss_weight = contrastive_loss_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models and tokenizers
        self._init_models()
    
    def _init_models(self):
        """Initialize and prepare all models and tokenizers."""
        logger.info("Initializing models and tokenizers...")
        
        # Setup 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model with quantization
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Set pad token if not set
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
        
        # Prepare model for QLoRA
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.base_model = get_peft_model(self.base_model, self.lora_config)
        
        # Load PubMedBERT for semantic embeddings
        self.semantic_model = AutoModel.from_pretrained(self.semantic_model_name)
        self.semantic_tokenizer = AutoTokenizer.from_pretrained(self.semantic_model_name)
        self.semantic_model.to(self.device)
        self.semantic_model.eval()  # Freeze semantic model
        
        logger.info("Models initialized successfully")
    
    def _prepare_input(self, input_text: str) -> str:
        """Prepare the input prompt."""
        # The input_text is already a complete prompt, just ensure it ends with a space
        # This helps create a clear separation between prompt and generated text
        text = input_text.rstrip() + " "
        if self.add_eos:
            # Add EOS token if specified (helps model understand prompt boundary)
            text = text + self.base_tokenizer.eos_token
        return text
    
    def _get_semantic_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get semantic embeddings using PubMedBERT."""
        with torch.no_grad():
            inputs = self.semantic_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.semantic_model(**inputs)
            # Use [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :]
            return F.normalize(embeddings, p=2, dim=1)
    
    def _compute_contrastive_loss(
        self,
        generated_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute contrastive loss between generated and target question embeddings.
        Uses InfoNCE loss formulation to align the semantic space.
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(generated_embeddings, target_embeddings.T) / temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(sim_matrix.size(0)).to(self.device)
        
        # Compute loss in both directions (symmetric)
        loss_g2t = F.cross_entropy(sim_matrix, labels)
        loss_t2g = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_g2t + loss_t2g) / 2
    
    def _load_csv_dataset(self, dataset_path: str) -> Dataset:
        """
        Load dataset from CSV file.
        
        Args:
            dataset_path: Path to the CSV dataset file
            
        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading dataset from CSV: {dataset_path}")
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Load CSV with pandas
        df = pd.read_csv(dataset_path)
        
        # Verify required columns exist
        required_columns = ['input_text', 'output_text']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)
        
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        return dataset
    
    def train(
        self,
        dataset_path: str,
        eval_dataset: Optional[Union[str, Dataset]] = None,
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 0.3,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100,
    ):
        """
        Train the model using QLoRA and contrastive semantic loss.
        
        The training process takes complete input prompts (input_text) and teaches
        the model to generate responses matching the structure and content of output_text.
        
        Example from the USMLE dataset:
        
        input_text: "Write an official question for USMLE preparation. Give the solution."
        
        output_text: "Question: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination...
        Answer: Nitrofurantoin"
        
        Args:
            dataset_path: Path to the dataset CSV file
            eval_dataset: Optional evaluation dataset
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            gradient_accumulation_steps: Number of steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            logging_steps: Steps between logging
            save_steps: Steps between saving checkpoints
            eval_steps: Steps between evaluations
        """
        # Initialize wandb
        wandb.init(
            project="usmle-question-gen",
            config={
                "base_model": self.base_model_name,
                "semantic_model": self.semantic_model_name,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "lora_config": self.lora_config.__dict__,
                "contrastive_loss_weight": self.contrastive_loss_weight,
            }
        )
        
        # Load dataset
        train_dataset = self._load_csv_dataset(dataset_path)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="wandb",
        )
        
        # Training loop
        self.base_model.train()
        optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Prepare inputs with template
                input_texts = [self._prepare_input(text) for text in batch["input_text"]]
                
                # Tokenize inputs and targets
                inputs = self.base_tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                targets = self.base_tokenizer(
                    batch["output_text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Forward pass
                outputs = self.base_model(**inputs, labels=targets["input_ids"])
                lm_loss = outputs.loss
                
                # Get semantic embeddings for generated and target questions
                generated_embeddings = self._get_semantic_embeddings(input_texts)
                target_embeddings = self._get_semantic_embeddings(batch["output_text"])
                
                # Compute contrastive loss
                contrastive_loss = self._compute_contrastive_loss(
                    generated_embeddings, target_embeddings
                )
                
                # Combined loss
                loss = lm_loss + self.contrastive_loss_weight * contrastive_loss
                
                # Backward pass
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.base_model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Log metrics
                if (batch_idx + 1) % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    wandb.log({
                        "loss": avg_loss,
                        "lm_loss": lm_loss.item(),
                        "contrastive_loss": contrastive_loss.item(),
                    })
                    total_loss = 0
                
                # Save checkpoint
                if (batch_idx + 1) % save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{epoch}-{batch_idx}")
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
        
        # Save final model
        self.save_checkpoint("final")
        wandb.finish()
    
    def save_checkpoint(self, name: str):
        """Save a model checkpoint."""
        save_dir = self.output_dir / name
        self.base_model.save_pretrained(save_dir)
        self.base_tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved checkpoint to {save_dir}")

def main():
    """Main function to run fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune model for USMLE question generation with semantic structure learning"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="BioMistral/BioMistral-7B",
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--semantic_model",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Model for semantic embeddings"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the CSV dataset file (must have input_text and output_text columns)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs"
    )
    
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--add_eos", type=bool, default=True,
                      help="Whether to add EOS token to input prompts")
    
    args = parser.parse_args()
    
    # Initialize and run trainer
    trainer = USMLEQuestionTrainer(
        base_model_name=args.base_model,
        semantic_model_name=args.semantic_model,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        contrastive_loss_weight=args.contrastive_loss_weight,
        max_length=args.max_length,
        add_eos=args.add_eos,
    )
    
    trainer.train(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )

if __name__ == "__main__":
    main()