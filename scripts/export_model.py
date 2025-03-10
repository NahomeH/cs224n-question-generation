#!/usr/bin/env python3
"""
Script to export a fine-tuned BioMistral model for deployment.

This script handles exporting a fine-tuned model to various formats:
1. HuggingFace format (for pushing to HF Hub)
2. ONNX format (for deployment in production)
3. Merged format (combining LoRA weights with base model)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import torch
import shutil
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, HfApi


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
            logging.FileHandler("export_model.log")
        ]
    )


def merge_lora_with_base_model(
    base_model_path: str, 
    lora_model_path: str, 
    output_path: str,
    device: str = "auto"
):
    """
    Merge LoRA weights with the base model.
    
    Args:
        base_model_path: Path to the base model
        lora_model_path: Path to the LoRA adapter weights
        output_path: Path to save the merged model
        device: Device to load the models on ("auto", "cpu", "cuda")
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Merging LoRA weights from {lora_model_path} with base model {base_model_path}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load base model and tokenizer
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load LoRA model
    logger.info("Loading LoRA model...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge weights
    logger.info("Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    logger.info(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Model successfully merged and saved!")
    
    return output_path


def push_to_hub(model_path: str, hub_repo: str, token: str = None):
    """
    Push model to HuggingFace Hub.
    
    Args:
        model_path: Path to the model to push
        hub_repo: HuggingFace Hub repository name (username/repo)
        token: HuggingFace token (optional)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Pushing model to HuggingFace Hub: {hub_repo}")
    
    # Login to HuggingFace if token provided
    if token:
        login(token=token)
    
    # Create repo if it doesn't exist
    api = HfApi()
    user = hub_repo.split('/')[0]
    repo_id = hub_repo.split('/')[1] if '/' in hub_repo else hub_repo
    
    # Push model to Hub
    logger.info(f"Pushing files from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.push_to_hub(hub_repo)
    tokenizer.push_to_hub(hub_repo)
    
    logger.info(f"Model successfully pushed to {hub_repo}!")


def export_to_onnx(model_path: str, output_path: str):
    """
    Export model to ONNX format.
    
    Args:
        model_path: Path to the model to export
        output_path: Path to save the ONNX model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Exporting model to ONNX format: {output_path}")
    
    # TODO: Implement ONNX export
    # This requires specific handling depending on the model architecture
    logger.warning("ONNX export not implemented yet")


def main():
    """Main function to export model."""
    parser = argparse.ArgumentParser(
        description="Export a fine-tuned BioMistral model for deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="BioMistral/BioMistral-7B",
        help="Path or name of the base model"
    )
    
    parser.add_argument(
        "--lora_model",
        type=str,
        required=True,
        help="Path to the fine-tuned LoRA adapter weights"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exported_model",
        help="Directory to save the exported model"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["merged", "onnx", "lora_only"],
        default="merged",
        help="Format to export the model in"
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--hub_repo",
        type=str,
        help="HuggingFace Hub repository name (username/repo)"
    )
    
    parser.add_argument(
        "--hub_token",
        type=str,
        help="HuggingFace token for pushing to Hub"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to load the model on"
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
    
    logger.info(f"Starting model export with args: {args}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Handle different export formats
    if args.format == "merged":
        # Merge LoRA weights with base model
        merged_model_path = merge_lora_with_base_model(
            args.base_model,
            args.lora_model,
            str(output_dir),
            args.device
        )
        
        if args.push_to_hub:
            if not args.hub_repo:
                logger.error("--hub_repo must be specified when using --push_to_hub")
                sys.exit(1)
            
            push_to_hub(
                merged_model_path,
                args.hub_repo,
                args.hub_token
            )
    
    elif args.format == "onnx":
        # Export to ONNX format
        # First merge the models
        merged_model_path = merge_lora_with_base_model(
            args.base_model,
            args.lora_model,
            str(output_dir / "merged"),
            args.device
        )
        
        # Then export to ONNX
        export_to_onnx(
            merged_model_path,
            str(output_dir / "onnx")
        )
    
    elif args.format == "lora_only":
        # Just copy the LoRA weights
        lora_path = Path(args.lora_model)
        if lora_path.is_dir():
            logger.info(f"Copying LoRA weights from {lora_path} to {output_dir}")
            shutil.copytree(lora_path, output_dir, dirs_exist_ok=True)
        else:
            logger.error(f"LoRA path {lora_path} is not a directory")
            sys.exit(1)
        
        if args.push_to_hub:
            if not args.hub_repo:
                logger.error("--hub_repo must be specified when using --push_to_hub")
                sys.exit(1)
            
            push_to_hub(
                str(output_dir),
                args.hub_repo,
                args.hub_token
            )
    
    logger.info("Model export completed successfully!")


if __name__ == "__main__":
    main() 