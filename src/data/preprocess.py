"""
Preprocess USMLE dataset for training or inference.

This script:
1. Downloads the MedQA-USMLE dataset from HuggingFace
2. Processes it into an input-output format suitable for generative models
3. Saves the processed dataset for later use
"""

import os
import random
import logging
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_dataset(dataset_name="GBaker/MedQA-USMLE-4-options", output_dir="processed_data"):
    """
    Process the USMLE dataset and save it in a format suitable for training.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Directory to save processed data
    """
    try:
        logger.info(f"Downloading dataset: {dataset_name}")
        hf_dataset = load_dataset(dataset_name)
        
        # Merge train and test sets into one dataset
        logger.info("Merging train and test sets")
        merged_dataset = concatenate_datasets([hf_dataset["train"], hf_dataset["test"]])
        
        # Convert to DataFrame
        df = pd.DataFrame(merged_dataset)
        
        # Retain only necessary columns: 'question' and 'answer'
        df = df[['question', 'answer']].dropna()
        logger.info(f"Dataset loaded with {len(df)} examples")
        
        # Define variation templates and components
        prompt_templates = [
            "{action} {description} question for USMLE preparation. {answer_action} {answer_noun}.",
            "{action} {description} USMLE-style question. {answer_action} {answer_noun}.",
            "{action} {description} board-style question for the USMLE. {answer_action} {answer_noun}.",
            "{action} {description} question following the USMLE format. {answer_action} {answer_noun}.",
            "{action} {description} medical board question for USMLE trainees. {answer_action} {answer_noun}.",
            "{action} {description} clinical reasoning question for USMLE study. {answer_action} {answer_noun}.",
            "{action} {description} USMLE practice question. {answer_action} {answer_noun}.",
            "{action} {description} high-yield USMLE exam question. {answer_action} {answer_noun}.",
            "{action} {description} board exam question for the USMLE. {answer_action} {answer_noun}.",
            "{action} {description} question for USMLE board practice. {answer_action} {answer_noun}."
        ]
        
        question_actions = ["Generate", "Create", "Develop", "Write", "Formulate", "Produce", "Make"]
        question_descriptions = ["a high-quality", "a challenging", "a realistic", "a board-style", "an official"]
        answer_actions = ["Provide", "Include", "Give", "Provide", "State"]
        answer_nouns = ["the correct answer", "the correct response", "the answer", "the solution", "the right answer", "the right response"]
        
        # Generate a randomized prompt for each row
        logger.info("Generating prompts with random templates")
        df["input_text"] = df.apply(
            lambda x: random.choice(prompt_templates).format(
                action=random.choice(question_actions),
                description=random.choice(question_descriptions),
                answer_action=random.choice(answer_actions),
                answer_noun=random.choice(answer_nouns)
            ),
            axis=1
        )
        
        # Modify the output format to include both the generated question and its correct answer
        df["output_text"] = df.apply(lambda x: f"Question: {x['question']}\nAnswer: {x['answer']}", axis=1)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the preprocessed dataset as a CSV file
        csv_path = output_path / "preprocessed_usmle_dataset.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved to {csv_path}")
        
        # Convert to Hugging Face dataset format and save as parquet
        processed_dataset = Dataset.from_pandas(df[['input_text', 'output_text']])
        parquet_path = output_path / "preprocessed_usmle_dataset"
        processed_dataset.save_to_disk(parquet_path)
        logger.info(f"Dataset also saved in HuggingFace format to {parquet_path}")
        
        # Display dataset sample
        print("\nSample of preprocessed data:")
        print(df[['input_text', 'output_text']].head(2))
        
        logger.info(f"Dataset successfully processed with {len(df)} examples")
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the script directory (src/data)
    script_dir = Path(__file__).parent
    
    # Create the output directory in the data folder
    output_dir = script_dir / "processed_data"
    
    print(f"Starting preprocessing. Output will be saved to: {output_dir}")
    preprocess_dataset(output_dir=output_dir)
