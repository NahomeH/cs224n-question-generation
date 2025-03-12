#!/usr/bin/env python3
"""
Fine-tuning script for USMLE question generation using QLoRA and contrastive semantic loss.
Uses preprocessed USMLE dataset to learn question semantics and structure.
Includes evaluation and early stopping functionality.
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
    default_data_collator,
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
from typing import Dict, List, Optional, Tuple, Union, Any

# Default dataset path
DEFAULT_DATASET_PATH = "src/data/processed_data/preprocessed_usmle_dataset.csv"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class USMLEModelFineTuner:
    """
    Trainer class for fine-tuning models to generate USMLE-style questions.
    Uses QLoRA and contrastive semantic loss to learn question structure and semantics.
    Includes evaluation and early stopping to prevent overfitting.
    """
    
    def __init__(
        self,
        base_model_name: str = "biomistral/BioMistral-7B",
        semantic_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        output_dir: str = "./finetuning",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
        contrastive_loss_weight: float = 0.1,
        max_length: int = 1024,
        add_eos: bool = True,
        random_seed: int = 42
    ):
        """
        Initialize the trainer with model configurations and hyperparameters.
        
        Args:
            base_model_name: Base LLM model to fine-tune
            semantic_model_name: Medical semantic model for contrastive learning
            output_dir: Directory to save model checkpoints and outputs
            lora_r: LoRA rank - controls adapter complexity (higher = more capacity)
            lora_alpha: LoRA alpha - controls adaptation strength
            lora_dropout: LoRA dropout rate for regularization
            lora_target_modules: Model layers to apply LoRA adapters to
            contrastive_loss_weight: Weight of contrastive loss in combined loss
            max_length: Maximum sequence length for tokenization
            add_eos: Whether to add EOS token to input prompts
            random_seed: Seed for reproducibility
        """
        # Store configuration parameters
        self.base_model_name = base_model_name
        self.semantic_model_name = semantic_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.add_eos = add_eos
        self.contrastive_loss_weight = contrastive_loss_weight
        self.random_seed = random_seed
        
        # Set seed for reproducibility
        torch.manual_seed(self.random_seed)
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Configure LoRA for parameter-efficient fine-tuning
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Initialize models and tokenizers to None initially
        self.base_model = None
        self.base_tokenizer = None
        self.semantic_model = None
        self.semantic_tokenizer = None
    
    def load_models(self):
        """
        Load and prepare all required models and tokenizers:
        1. Base LLM with 4-bit quantization
        2. Semantic model for contrastive learning
        """
        logger.info("Loading models and tokenizers...")
        
        # === 1. Load Base Model with Quantization ===
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Configure 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer first (it's lightweight)
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        # Set pad token if not set
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # Load base model with quantization
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Ensure pad token ID is set
        self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
        
        # Prepare model for LoRA fine-tuning
        self.prepare_for_qlora()
        
        # === 2. Load Semantic Model for Contrastive Learning ===
        logger.info(f"Loading semantic model: {self.semantic_model_name}")
        
        self.semantic_model = AutoModel.from_pretrained(self.semantic_model_name)
        self.semantic_tokenizer = AutoTokenizer.from_pretrained(self.semantic_model_name)
        
        # Move to appropriate device and set to eval mode (we don't train this model)
        self.semantic_model.to(self.device)
        self.semantic_model.eval()
        
        logger.info("All models loaded successfully")
    
    def prepare_for_qlora(self):
        """
        Prepare the base model for QLoRA fine-tuning by:
        1. Preparing for k-bit training
        2. Adding LoRA adapters
        3. Logging trainable parameters
        """
        logger.info("Preparing model for QLoRA fine-tuning")
        
        # Prepare model for k-bit training (required for quantized models)
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Add LoRA adapters to specified layers
        self.base_model = get_peft_model(self.base_model, self.lora_config)
        
        # Log parameter efficiency information
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.base_model.parameters())
        percentage = 100 * trainable_params / total_params
        
        logger.info(f"Trainable params: {trainable_params:,} ({percentage:.2f}% of {total_params:,} total)")
    
    def prepare_input_prompt(self, input_text: str) -> str:
        """
        Prepare the input prompt for the model.
        
        Args:
            input_text: Raw input prompt
            
        Returns:
            Formatted input prompt
        """
        # Ensure the prompt ends with a space for better generation
        text = input_text.rstrip() + " "
        
        # Optionally add EOS token to help the model understand prompt boundaries
        if self.add_eos:
            text = text + self.base_tokenizer.eos_token
            
        return text
    
    def get_semantic_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Extract semantic embeddings from text using the semantic model.
        
        Args:
            texts: List of text strings
            
        Returns:
            Normalized semantic embeddings
        """
        with torch.no_grad():
            inputs = self.semantic_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.semantic_model(**inputs)
            
            # Use [CLS] token embeddings as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # L2 normalize for cosine similarity computation
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return normalized_embeddings
    
    def compute_contrastive_loss(
        self,
        generated_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute contrastive loss between generated and target embeddings.
        Uses InfoNCE loss formulation to align semantic spaces.
        
        Args:
            generated_embeddings: Embeddings of generated/input text
            target_embeddings: Embeddings of target/output text
            temperature: Temperature parameter for scaling similarity
            
        Returns:
            Contrastive loss tensor
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(generated_embeddings, target_embeddings.T) / temperature
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(sim_matrix.size(0)).to(self.device)
        
        # Compute loss in both directions (symmetric)
        loss_g2t = F.cross_entropy(sim_matrix, labels)
        loss_t2g = F.cross_entropy(sim_matrix.T, labels)
        
        # Average both directions for stability
        return (loss_g2t + loss_t2g) / 2
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """
        Load dataset from CSV file.
        
        Args:
            dataset_path: Path to the CSV dataset file
            
        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading dataset from: {dataset_path}")
        
        # Verify the file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Check if it's a CSV file
        if dataset_path.endswith('.csv'):
            # Load CSV with pandas
            df = pd.read_csv(dataset_path)
            
            # Verify required columns exist
            required_columns = ['input_text', 'output_text']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Dataset missing required columns: {missing_columns}")
            
            # Convert to HuggingFace Dataset
            dataset = Dataset.from_pandas(df)
            
        else:
            # Try loading as a HuggingFace dataset
            try:
                dataset = load_dataset(dataset_path, split='train')
                
                # Verify required columns
                required_columns = ['input_text', 'output_text']
                missing_columns = [col for col in required_columns if col not in dataset.column_names]
                
                if missing_columns:
                    raise ValueError(f"Dataset missing required columns: {missing_columns}")
                    
            except Exception as e:
                raise ValueError(f"Could not load dataset: {str(e)}")
        
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
        return dataset
    
    def split_dataset(self, dataset: Dataset, eval_split: float = 0.1) -> Dict[str, Dataset]:
        """
        Split dataset into training and evaluation sets.
        
        Args:
            dataset: Full dataset
            eval_split: Fraction to use for evaluation
            
        Returns:
            Dictionary with 'train' and 'test' splits
        """
        logger.info(f"Splitting dataset with {eval_split:.1%} for evaluation")
        
        # Split dataset
        splits = dataset.train_test_split(test_size=eval_split, shuffle=True, seed=self.random_seed)
        
        logger.info(f"Training set: {len(splits['train'])} examples")
        logger.info(f"Evaluation set: {len(splits['test'])} examples")
        
        return splits
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate model performance on a validation dataset.
        
        Args:
            eval_dataset: Dataset for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting evaluation")
        
        # Create dataloader for evaluation
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=8,  # Can use larger batch size for evaluation
            shuffle=False,
            collate_fn=default_data_collator
        )
        
        # Set model to evaluation mode
        self.base_model.eval()
        
        # Initialize metrics
        total_lm_loss = 0
        total_contrastive_loss = 0
        total_samples = 0
        
        # Evaluation loop
        with torch.no_grad():  # Disable gradient calculation for efficiency
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.base_model(**batch)
                lm_loss = outputs.loss
                
                # Update LM metrics
                batch_size = batch["input_ids"].size(0)
                total_lm_loss += lm_loss.item() * batch_size
                total_samples += batch_size
                
                # Compute contrastive loss
                # Extract input and output texts
                input_texts = self.base_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                output_texts = self.base_tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                
                # Get semantic embeddings
                input_embeddings = self.get_semantic_embeddings(input_texts)
                output_embeddings = self.get_semantic_embeddings(output_texts)
                
                # Compute contrastive loss
                contrastive_loss = self.compute_contrastive_loss(input_embeddings, output_embeddings)
                total_contrastive_loss += contrastive_loss.item() * batch_size
        
        # Calculate average losses
        avg_lm_loss = total_lm_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples
        combined_loss = avg_lm_loss + self.contrastive_loss_weight * avg_contrastive_loss
        perplexity = torch.exp(torch.tensor(avg_lm_loss)).item()
        
        # Set model back to training mode
        self.base_model.train()
        
        # Prepare metrics dictionary
        metrics = {
            "lm_loss": avg_lm_loss,
            "contrastive_loss": avg_contrastive_loss,
            "combined_loss": combined_loss,
            "perplexity": perplexity
        }
        
        logger.info(f"Evaluation results: combined_loss = {combined_loss:.4f}, "
                   f"perplexity = {perplexity:.4f}")
        
        return metrics
    
    def save_checkpoint(self, name: str):
        """
        Save a model checkpoint.
        
        Args:
            name: Name for the checkpoint directory
        """
        save_dir = self.output_dir / name
        self.base_model.save_pretrained(save_dir)
        self.base_tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved checkpoint to {save_dir}")
    
    def train(
        self,
        dataset_path: str,
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 0.3,
        eval_split: float = 0.1,
        eval_steps: int = 500,
        save_steps: int = 100,
        logging_steps: int = 10,
        patience: int = 3,
        use_wandb: bool = True
    ) -> Tuple[Any, int]:
        """
        Train the model using QLoRA and contrastive semantic loss with evaluation.
        
        Args:
            dataset_path: Path to the dataset file
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            gradient_accumulation_steps: Steps for gradient accumulation
            max_grad_norm: Maximum gradient norm for clipping
            eval_split: Fraction of data to use for evaluation
            eval_steps: Steps between evaluations
            save_steps: Steps between saving checkpoints
            logging_steps: Steps between logging metrics
            patience: Number of evaluations without improvement before early stopping
            use_wandb: Whether to use Weights & Biases for tracking
            
        Returns:
            Tuple of (trained model, best model step)
        """
        # Make sure models are loaded
        if self.base_model is None or self.semantic_model is None:
            self.load_models()
        
        # Initialize Weights & Biases for experiment tracking
        if use_wandb:
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
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                }
            )
        
        # Load and split dataset
        full_dataset = self.load_dataset(dataset_path)
        datasets = self.split_dataset(full_dataset, eval_split=eval_split)
        train_dataset, eval_dataset = datasets["train"], datasets["test"]
        
        # Create training dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.base_model.parameters(), lr=learning_rate)
        
        # Training tracking variables
        global_step = 0
        total_loss = 0
        total_lm_loss = 0
        total_contrastive_loss = 0
        
        # For early stopping
        best_eval_loss = float('inf')
        best_model_step = 0
        no_improvement_count = 0
        
        logger.info(f"Starting training with contrastive loss (weight={self.contrastive_loss_weight})")
        
        # Initial evaluation
        logger.info("Performing initial evaluation")
        eval_metrics = self.evaluate(eval_dataset)
        best_eval_loss = eval_metrics["combined_loss"]
        
        # Training loop
        for epoch in range(num_epochs):
            # Set model to training mode at the start of each epoch
            self.base_model.train()
            
            epoch_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(epoch_progress):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass for language modeling
                outputs = self.base_model(**batch)
                lm_loss = outputs.loss
                
                # Extract input and output texts for contrastive learning
                input_texts = self.base_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                output_texts = self.base_tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                
                # Get semantic embeddings
                input_embeddings = self.get_semantic_embeddings(input_texts)
                output_embeddings = self.get_semantic_embeddings(output_texts)
                
                # Compute contrastive loss
                contrastive_loss = self.compute_contrastive_loss(input_embeddings, output_embeddings)
                
                # Combined loss
                loss = lm_loss + self.contrastive_loss_weight * contrastive_loss
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update loss tracking
                total_loss += loss.item()
                total_lm_loss += lm_loss.item() / gradient_accumulation_steps
                total_contrastive_loss += contrastive_loss.item() / gradient_accumulation_steps
                
                # Update parameters every gradient_accumulation_steps
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_grad_norm)
                    
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update global step
                    global_step += 1
                    
                    # Log metrics periodically
                    if global_step % logging_steps == 0:
                        avg_loss = total_loss * gradient_accumulation_steps / logging_steps
                        avg_lm_loss = total_lm_loss * gradient_accumulation_steps / logging_steps
                        avg_contrastive_loss = total_contrastive_loss * gradient_accumulation_steps / logging_steps
                        
                        logger.info(f"Step {global_step}: "
                                   f"loss = {avg_loss:.4f}, "
                                   f"lm_loss = {avg_lm_loss:.4f}, "
                                   f"contrastive_loss = {avg_contrastive_loss:.4f}")
                        
                        if use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/lm_loss": avg_lm_loss,
                                "train/contrastive_loss": avg_contrastive_loss,
                                "train/global_step": global_step,
                            })
                        
                        # Reset trackers
                        total_loss = 0
                        total_lm_loss = 0
                        total_contrastive_loss = 0
                    
                    # Evaluate periodically
                    if global_step % eval_steps == 0:
                        eval_metrics = self.evaluate(eval_dataset)
                        
                        if use_wandb:
                            wandb.log({
                                "eval/lm_loss": eval_metrics["lm_loss"],
                                "eval/contrastive_loss": eval_metrics["contrastive_loss"],
                                "eval/combined_loss": eval_metrics["combined_loss"],
                                "eval/perplexity": eval_metrics["perplexity"],
                                "eval/global_step": global_step,
                            })
                        
                        # Check for improvement
                        eval_loss = eval_metrics["combined_loss"]
                        
                        if eval_loss < best_eval_loss:
                            logger.info(f"New best model! Loss improved from {best_eval_loss:.4f} to {eval_loss:.4f}")
                            best_eval_loss = eval_loss
                            best_model_step = global_step
                            no_improvement_count = 0
                            
                            # Save best model
                            self.save_checkpoint("best_model")
                        else:
                            no_improvement_count += 1
                            logger.info(f"No improvement for {no_improvement_count} evaluations")
                            
                            # Early stopping
                            if patience > 0 and no_improvement_count >= patience:
                                logger.info(f"Early stopping after {no_improvement_count} evaluations without improvement")
                                if use_wandb:
                                    wandb.finish()
                                return self.base_model, best_model_step
                    
                    # Save regular checkpoint
                    if global_step % save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{global_step}")
                
                # Update progress bar
                epoch_progress.set_postfix({
                    "loss": loss.item() * gradient_accumulation_steps,
                    "lm_loss": lm_loss.item(),
                    "contr_loss": contrastive_loss.item()
                })
        
        # Save final model
        self.save_checkpoint("final_model")
        logger.info(f"Training complete. Final model saved.")
        
        if use_wandb:
            wandb.finish()
        
        return self.base_model, best_model_step


def main():
    """Main function to run fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune model for USMLE question generation with semantic structure learning"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="biomistral/BioMistral-7B",
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
        help="Path to the dataset file (must have input_text and output_text columns)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs"
    )
    
    # QLoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Contrastive learning parameter
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                      help="Weight of contrastive loss in total loss")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                       help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, 
                       help="Maximum gradient norm")
    parser.add_argument("--max_length", type=int, default=1024, 
                       help="Maximum sequence length")
    parser.add_argument("--add_eos", type=bool, default=True,
                      help="Whether to add EOS token to input prompts")
    
    # Evaluation parameters
    parser.add_argument("--eval_split", type=float, default=0.1, 
                       help="Fraction of data to use for evaluation")
    parser.add_argument("--eval_steps", type=int, default=500, 
                       help="Steps between evaluations")
    parser.add_argument("--save_steps", type=int, default=100, 
                       help="Steps between saving checkpoints")
    parser.add_argument("--logging_steps", type=int, default=10, 
                       help="Steps between logging")
    parser.add_argument("--patience", type=int, default=3, 
                       help="Early stopping patience (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--no_wandb", action="store_true", 
                       help="Disable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Initialize trainer with configuration
    trainer = USMLEModelFineTuner(
        base_model_name=args.base_model,
        semantic_model_name=args.semantic_model,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        contrastive_loss_weight=args.contrastive_loss_weight,
        max_length=args.max_length,
        add_eos=args.add_eos,
        random_seed=args.seed
    )
    
    # Load models
    trainer.load_models()
    
    # Run training
    _, best_step = trainer.train(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        eval_split=args.eval_split,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        patience=args.patience,
        use_wandb=not args.no_wandb
    )
    
    logger.info(f"Training complete. Best model found at step {best_step}")


if __name__ == "__main__":
    main()