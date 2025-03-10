"""
Script to download and prepare medical datasets for fine-tuning.

This script downloads and processes publicly available medical QA datasets
to prepare them for fine-tuning the USMLE question generation model.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json
import requests
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional, Any
from huggingface_hub import hf_hub_download, login

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define available datasets
AVAILABLE_DATASETS = {
    "medqa": {
        "huggingface": "GBaker/MedQA-USMLE-4-options",
        "description": "Multiple-choice medical exam questions from USMLE"
    },
    "pubmedqa": {
        "huggingface": "pubmed_qa",
        "description": "PubMedQA: A Dataset for Biomedical Research Question Answering"
    },
    "medical_meadow": {
        "huggingface": "medalpaca/medical_meadow_medical_qa",
        "description": "Medical question-answering dataset from MedAlpaca"
    },
    # Add more datasets as needed
}


def download_dataset(dataset_name: str, output_dir: Path) -> Path:
    """
    Download a dataset from HuggingFace or other sources.
    
    Args:
        dataset_name: Name of the dataset to download
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    if dataset_name not in AVAILABLE_DATASETS:
        available = ", ".join(AVAILABLE_DATASETS.keys())
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {available}")
    
    dataset_info = AVAILABLE_DATASETS[dataset_name]
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Downloading dataset: {dataset_name}")
    logger.info(f"Description: {dataset_info['description']}")
    
    try:
        # Download from HuggingFace
        if "huggingface" in dataset_info:
            dataset = load_dataset(dataset_info["huggingface"])
            
            # Save all splits
            for split_name, split_data in dataset.items():
                output_file = dataset_dir / f"{split_name}.jsonl"
                logger.info(f"Saving {split_name} split to {output_file}")
                
                with open(output_file, "w") as f:
                    for item in tqdm(split_data, desc=f"Saving {split_name}"):
                        f.write(json.dumps(item) + "\n")
            
            return dataset_dir
        else:
            # Implement other download methods if needed
            raise NotImplementedError(f"Download method for {dataset_name} not implemented")
    
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_name}: {e}")
        raise


def process_dataset(dataset_path: Path, output_path: Path, dataset_name: str) -> Path:
    """
    Process a downloaded dataset into a standardized format.
    
    Args:
        dataset_path: Path to the downloaded dataset
        output_path: Path to save the processed dataset
        dataset_name: Name of the dataset
        
    Returns:
        Path to the processed dataset
    """
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Process based on dataset type
    if dataset_name == "medqa":
        return process_medqa(dataset_path, output_path)
    elif dataset_name == "pubmedqa":
        return process_pubmedqa(dataset_path, output_path)
    elif dataset_name == "medical_meadow":
        return process_medical_meadow(dataset_path, output_path)
    else:
        raise ValueError(f"Processing method for {dataset_name} not implemented")


def process_medqa(dataset_path: Path, output_path: Path) -> Path:
    """
    Process MedQA dataset into standardized format.
    
    Args:
        dataset_path: Path to the downloaded MedQA dataset
        output_path: Path to save the processed dataset
        
    Returns:
        Path to the processed dataset
    """
    # Placeholder implementation - will need to be customized
    processed_path = output_path / "medqa_processed.json"
    
    logger.info("TODO: Implement MedQA processing")
    
    # Example placeholder data
    processed_data = {"pairs": [], "metadata": {"source": "MedQA-USMLE"}}
    
    with open(processed_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    return processed_path


def process_pubmedqa(dataset_path: Path, output_path: Path) -> Path:
    """
    Process PubMedQA dataset into standardized format.
    
    Args:
        dataset_path: Path to the downloaded PubMedQA dataset
        output_path: Path to save the processed dataset
        
    Returns:
        Path to the processed dataset
    """
    # Placeholder implementation - will need to be customized
    processed_path = output_path / "pubmedqa_processed.json"
    
    logger.info("TODO: Implement PubMedQA processing")
    
    # Example placeholder data
    processed_data = {"pairs": [], "metadata": {"source": "PubMedQA"}}
    
    with open(processed_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    return processed_path


def process_medical_meadow(dataset_path: Path, output_path: Path) -> Path:
    """
    Process Medical Meadow dataset into standardized format.
    
    Args:
        dataset_path: Path to the downloaded Medical Meadow dataset
        output_path: Path to save the processed dataset
        
    Returns:
        Path to the processed dataset
    """
    # Placeholder implementation - will need to be customized
    processed_path = output_path / "medical_meadow_processed.json"
    
    logger.info("TODO: Implement Medical Meadow processing")
    
    # Example placeholder data
    processed_data = {"pairs": [], "metadata": {"source": "Medical Meadow"}}
    
    with open(processed_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    return processed_path


def main():
    """Main function to download and process datasets."""
    parser = argparse.ArgumentParser(
        description="Download and prepare medical datasets for fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--datasets", 
        nargs="+",
        default=["medqa"],
        choices=list(AVAILABLE_DATASETS.keys()) + ["all"],
        help="Datasets to download and process"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data",
        help="Directory to save the datasets"
    )
    
    parser.add_argument(
        "--huggingface_token", 
        type=str, 
        default=None,
        help="HuggingFace token for downloading gated datasets"
    )
    
    parser.add_argument(
        "--process_only", 
        action="store_true",
        help="Only process already downloaded datasets"
    )
    
    args = parser.parse_args()
    
    # Login to HuggingFace if token provided
    if args.huggingface_token:
        logger.info("Logging in to HuggingFace Hub")
        login(token=args.huggingface_token)
    
    # Determine which datasets to download
    datasets_to_process = list(AVAILABLE_DATASETS.keys()) if "all" in args.datasets else args.datasets
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Download and process each dataset
    for dataset_name in datasets_to_process:
        logger.info(f"Working on dataset: {dataset_name}")
        
        try:
            # Download dataset if needed
            if not args.process_only:
                dataset_path = download_dataset(dataset_name, raw_dir)
            else:
                dataset_path = raw_dir / dataset_name
                if not dataset_path.exists():
                    logger.error(f"Dataset {dataset_name} not found in {raw_dir}. Skipping.")
                    continue
            
            # Process dataset
            processed_path = process_dataset(dataset_path, processed_dir, dataset_name)
            logger.info(f"Successfully processed {dataset_name} to {processed_path}")
            
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {e}")
            continue
    
    logger.info("All datasets processed successfully!")


if __name__ == "__main__":
    main() 