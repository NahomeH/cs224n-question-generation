#!/usr/bin/env python3
"""
Main script for fine-tuning BioMistral models for USMLE question generation.

Example usage:
    python -m src.fine_tune --dataset medqa --output_dir ./outputs/fine_tuned_model
    python -m src.fine_tune --dataset ./data/my_dataset.json --epochs 5 --batch_size 2
"""

import argparse
import logging
import sys
from pathlib import Path
from src.training.trainer import MedicalQuestionTrainer
from src.data.data_loader import MedicalDataLoader
from src.evaluation.evaluator import MedicalQuestionEvaluator
import torch
import os


def setup_logging(log_level="INFO"):
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("fine_tuning.log")
        ]
    )


def main():
    """Main fine-tuning function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune BioMistral model for USMLE question generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Dataset path or name (local path or HuggingFace dataset)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="BioMistral/BioMistral-7B",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit", "none"],
        default="8bit",
        help="Quantization method for training"
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4,
        help="Number of steps for gradient accumulation"
    )
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    
    # Output arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs/fine_tuned_model",
        help="Directory to save fine-tuned model and outputs"
    )
    parser.add_argument(
        "--eval_during_training", 
        action="store_true",
        help="Whether to evaluate during training"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true",
        help="Whether to use Weights & Biases for logging"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting fine-tuning process with args: {args}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Fine-tuning may be very slow!")
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize data loader and load dataset
    logger.info(f"Loading dataset from: {args.dataset}")
    data_loader = MedicalDataLoader()
    
    # TODO: Implement dataset loading logic
    # dataset = data_loader.load_dataset(args.dataset)
    # train_dataset, eval_dataset = data_loader.split_data(dataset)
    
    # For demonstration only - placeholder
    from datasets import Dataset
    import pandas as pd
    
    # Placeholder datasets (to be replaced with actual implementation)
    train_dataset = Dataset.from_pandas(pd.DataFrame({
        "text": ["This is a placeholder for training data"],
        "question": ["This is a placeholder for training question"]
    }))
    
    eval_dataset = Dataset.from_pandas(pd.DataFrame({
        "text": ["This is a placeholder for evaluation data"],
        "question": ["This is a placeholder for evaluation question"]
    }))
    
    # Initialize trainer
    logger.info(f"Initializing trainer with model: {args.model}")
    trainer = MedicalQuestionTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        quantization=args.quantize if args.quantize != "none" else None,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_wandb=args.use_wandb
    )
    
    # Setup model
    logger.info("Setting up model")
    trainer.setup_model()
    
    # Prepare training arguments
    training_args = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
    }
    
    # Train model
    logger.info("Starting training")
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_during_training else None,
        training_args=training_args
    )
    
    # Save model
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    trainer.save_model()
    
    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main() 