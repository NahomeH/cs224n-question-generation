#!/usr/bin/env python3
"""
Evaluation script for assessing the medical accuracy of generated USMLE questions.
This script:
1. Loads input texts from the USMLE dataset
2. Generates questions using the BioMistral model
3. Evaluates the medical accuracy of the generated questions
"""

import logging
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATASET_PATH = "src/data/processed_data/preprocessed_usmle_dataset.csv"
DEFAULT_OUTPUT_PATH = "./evaluation_results"
BIOMISTRAL_MODEL_NAME = "biomistral/BioMistral-7B"

class MedicalAccuracyEvaluator:
    """
    Evaluator for assessing the medical accuracy of generated USMLE questions.
    """
    
    def __init__(
        self,
        generator_model_name: str = BIOMISTRAL_MODEL_NAME,
        evaluation_model_name: str = "stanford-crfm/BioMedLM",  # Keep BioMedLM as evaluator
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        seed: int = 42,
        use_gpu: bool = True,
        output_dir: str = DEFAULT_OUTPUT_PATH
    ):
        """
        Initialize the medical accuracy evaluator.
        
        Args:
            generator_model_name: Model to use for generating questions (BioMistral)
            evaluation_model_name: Model to use for evaluation (ideally medically specialized)
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to evaluate (None = all)
            seed: Random seed for reproducibility
            use_gpu: Whether to use GPU for inference
            output_dir: Directory to save evaluation results
        """
        self.generator_model_name = generator_model_name
        self.evaluation_model_name = evaluation_model_name
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.seed = seed
        self.use_gpu = use_gpu
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize generator and evaluator models"""
        logger.info(f"Initializing question generator: {self.generator_model_name}")
        
        # Load BioMistral model for generation
        device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1
        self.generator = pipeline(
            "text-generation",
            model=self.generator_model_name,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32
        )
        
        logger.info(f"Initializing evaluation model: {self.evaluation_model_name}")
        # Load the evaluation model
        self.evaluator = pipeline(
            "text-generation",
            model=self.evaluation_model_name,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32
        )
    
    def _load_dataset(self, dataset_path: str):
        """Load and preprocess the dataset"""
        logger.info(f"Loading dataset from {dataset_path}")
        
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported file format for {dataset_path}")
            
        # If max_samples specified, randomly sample the dataset
        if self.max_samples and self.max_samples < len(df):
            df = df.sample(n=self.max_samples, random_state=self.seed)
            
        logger.info(f"Loaded {len(df)} samples for evaluation")
        return df
    
    def _generate_questions(self, input_texts: List[str]) -> List[str]:
        """Generate questions using the BioMistral model"""
        logger.info("Generating questions from input texts")
        generated_questions = []
        
        for input_text in tqdm(input_texts, desc="Generating questions"):
            # Use the input_text directly as the prompt
            prompt = input_text
            
            # Generate the question
            response = self.generator(
                prompt,
                max_new_tokens=250,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True
            )[0]['generated_text']
            
            # Extract just the generated question part (remove the prompt)
            # Since we're using the input_text directly, remove it from the response
            if len(response) > len(prompt) and response.startswith(prompt):
                question = response[len(prompt):].strip()
            else:
                question = response.strip()
                
            generated_questions.append(question)
            
        return generated_questions
    
    def _evaluate_medical_accuracy(self, input_texts: List[str], generated_questions: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate the medical accuracy of generated questions using an evaluation model.
        
        This uses a medical evaluation prompt that asks the evaluation model to assess:
        1. Medical factual correctness
        2. Clinical relevance
        3. Terminology accuracy
        """
        logger.info("Evaluating medical accuracy of generated questions")
        evaluation_results = []
        
        # Evaluation prompt template
        evaluation_prompt_template = """
        You are a medical expert evaluating the accuracy of USMLE-style medical questions.
        
        Generated Question:
        {generated_question}
        
        Please evaluate the medical accuracy of the generated question on a scale of 1-5 (5 being highest) across these dimensions:
        1. Factual Correctness: Are the medical facts in the question scientifically accurate according to current medical knowledge?
        2. Clinical Relevance: Is the question clinically appropriate and relevant to medical practice?
        3. Terminology Accuracy: Are medical terms used correctly and appropriately?
        
        Your response must be in JSON format exactly like this:
        {{
            "factual_correctness": [score 1-5],
            "clinical_relevance": [score 1-5],
            "terminology_accuracy": [score 1-5],
            "average_score": [average of the three scores, rounded to 2 decimal places],
            "explanation": [brief explanation of your evaluation]
        }}
        
        Respond with ONLY the JSON.
        """
        
        for input_text, generated_question in tqdm(zip(input_texts, generated_questions), 
                                                   desc="Evaluating medical accuracy", 
                                                   total=len(input_texts)):
            
            # Prepare evaluation prompt
            evaluation_prompt = evaluation_prompt_template.format(
                generated_question=generated_question
            )
            
            # Get evaluation from the model
            evaluation_response = self.evaluator(
                evaluation_prompt,
                max_new_tokens=1024,
                temperature=0.1,  # Low temperature for more deterministic evaluation
                do_sample=False
            )[0]['generated_text']
            
            # Extract the JSON response
            try:
                # Find the JSON part of the response
                start_idx = evaluation_response.find('{')
                end_idx = evaluation_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = evaluation_response[start_idx:end_idx]
                    evaluation_result = json.loads(json_str)
                else:
                    # If no JSON found, assign default failed values
                    evaluation_result = {
                        "factual_correctness": 0,
                        "clinical_relevance": 0,
                        "terminology_accuracy": 0,
                        "average_score": 0,
                        "explanation": "Failed to parse evaluation response"
                    }
                    logger.warning(f"Failed to extract JSON from evaluation response: {evaluation_response}")
            except json.JSONDecodeError:
                evaluation_result = {
                    "factual_correctness": 0,
                    "clinical_relevance": 0,
                    "terminology_accuracy": 0,
                    "average_score": 0,
                    "explanation": "Failed to parse evaluation response"
                }
                logger.warning(f"Failed to parse JSON from evaluation response: {evaluation_response}")
            
            # Add the input and output to the result
            result_entry = {
                "input_text": input_text,
                "generated_question": generated_question,
                "evaluation": evaluation_result
            }
            
            evaluation_results.append(result_entry)
            
        return evaluation_results
    
    def evaluate(self, dataset_path: str) -> Dict[str, Any]:
        """
        Main evaluation function that:
        1. Loads the dataset
        2. Generates questions
        3. Evaluates medical accuracy
        4. Computes overall statistics
        5. Saves results
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Load dataset
        df = self._load_dataset(dataset_path)
        input_texts = df['input_text'].tolist()  # Adjust column name as needed
        
        # Generate questions
        generated_questions = self._generate_questions(input_texts)
        
        # Evaluate medical accuracy
        evaluation_results = self._evaluate_medical_accuracy(input_texts, generated_questions)
        
        # Compute overall statistics
        metrics = self._compute_metrics(evaluation_results)
        
        # Save results
        self._save_results(evaluation_results, metrics)
        
        return metrics
    
    def _compute_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall metrics from evaluation results"""
        metrics = {
            "factual_correctness_avg": 0,
            "clinical_relevance_avg": 0,
            "terminology_accuracy_avg": 0,
            "overall_accuracy_avg": 0,
            "total_evaluated": len(evaluation_results),
            "success_rate": 0  # Percentage of evaluations that completed successfully
        }
        
        successful_evaluations = 0
        
        for result in evaluation_results:
            eval_data = result["evaluation"]
            if eval_data["factual_correctness"] > 0:  # Check if evaluation was successful
                metrics["factual_correctness_avg"] += eval_data["factual_correctness"]
                metrics["clinical_relevance_avg"] += eval_data["clinical_relevance"]
                metrics["terminology_accuracy_avg"] += eval_data["terminology_accuracy"]
                metrics["overall_accuracy_avg"] += eval_data["average_score"]
                successful_evaluations += 1
        
        # Compute averages
        if successful_evaluations > 0:
            metrics["factual_correctness_avg"] /= successful_evaluations
            metrics["clinical_relevance_avg"] /= successful_evaluations
            metrics["terminology_accuracy_avg"] /= successful_evaluations
            metrics["overall_accuracy_avg"] /= successful_evaluations
            
        metrics["success_rate"] = (successful_evaluations / len(evaluation_results)) * 100 if len(evaluation_results) > 0 else 0
        
        # Round all metrics to 2 decimal places
        for key in metrics:
            if isinstance(metrics[key], float):
                metrics[key] = round(metrics[key], 2)
        
        return metrics
    
    def _save_results(self, evaluation_results: List[Dict[str, Any]], metrics: Dict[str, float]):
        """Save evaluation results and metrics to files"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = self.output_dir / f"biomistral_evaluation_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Save metrics summary
        metrics_path = self.output_dir / f"biomistral_evaluation_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Create a summary report
        report_path = self.output_dir / f"biomistral_evaluation_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(f"BioMistral Medical Accuracy Evaluation Report - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=================================================================\n\n")
            f.write(f"Generator model evaluated: {self.generator_model_name}\n")
            f.write(f"Evaluation model: {self.evaluation_model_name}\n")
            f.write(f"Total samples evaluated: {metrics['total_evaluated']}\n")
            f.write(f"Successful evaluations: {metrics['success_rate']}%\n\n")
            f.write(f"Evaluation Metrics:\n")
            f.write(f"  - Average Factual Correctness: {metrics['factual_correctness_avg']}/5\n")
            f.write(f"  - Average Clinical Relevance: {metrics['clinical_relevance_avg']}/5\n")
            f.write(f"  - Average Terminology Accuracy: {metrics['terminology_accuracy_avg']}/5\n")
            f.write(f"  - Overall Medical Accuracy: {metrics['overall_accuracy_avg']}/5\n\n")
            
            # Add some example evaluations (3 good, 3 bad if available)
            f.write(f"Example Evaluations:\n")
            f.write(f"-----------------\n\n")
            
            # Sort by average score
            sorted_results = sorted(
                [r for r in evaluation_results if r["evaluation"]["average_score"] > 0],
                key=lambda x: x["evaluation"]["average_score"],
                reverse=True
            )
            
            # Write 3 best examples
            f.write(f"Top scoring examples:\n")
            for i, result in enumerate(sorted_results[:3]):
                f.write(f"Example {i+1} (Score: {result['evaluation']['average_score']}/5)\n")
                f.write(f"Generated Question: {result['generated_question'][:300]}...\n")
                f.write(f"Explanation: {result['evaluation']['explanation']}\n\n")
            
            # Write 3 worst examples
            f.write(f"Lowest scoring examples:\n")
            for i, result in enumerate(sorted_results[-3:]):
                f.write(f"Example {i+1} (Score: {result['evaluation']['average_score']}/5)\n")
                f.write(f"Generated Question: {result['generated_question'][:300]}...\n")
                f.write(f"Explanation: {result['evaluation']['explanation']}\n\n")
        
        logger.info(f"Evaluation results saved to {self.output_dir}")
        logger.info(f"Detailed results: {results_path}")
        logger.info(f"Metrics summary: {metrics_path}")
        logger.info(f"Evaluation report: {report_path}")


def main():
    """Main entry point for the evaluation script"""
    # Use default values for BioMistral evaluation
    generator_model_name = BIOMISTRAL_MODEL_NAME
    evaluation_model_name = "stanford-crfm/BioMedLM"
    dataset_path = DEFAULT_DATASET_PATH
    output_dir = DEFAULT_OUTPUT_PATH
    batch_size = 8
    max_samples = None  # Set to an integer if you want to evaluate fewer samples
    seed = 42
    use_gpu = True  # Set to False if you don't want to use GPU
    
    # Initialize the evaluator
    evaluator = MedicalAccuracyEvaluator(
        generator_model_name=generator_model_name,
        evaluation_model_name=evaluation_model_name,
        batch_size=batch_size,
        max_samples=max_samples,
        seed=seed,
        use_gpu=use_gpu,
        output_dir=output_dir
    )
    
    # Run the evaluation
    metrics = evaluator.evaluate(dataset_path)
    
    # Print summary of results
    print("\nBioMistral Evaluation Results Summary:")
    print(f"  - Generator model: {generator_model_name}")
    print(f"  - Evaluation model: {evaluation_model_name}")
    print(f"  - Total samples evaluated: {metrics['total_evaluated']}")
    print(f"  - Successful evaluations: {metrics['success_rate']}%")
    print(f"  - Overall Medical Accuracy: {metrics['overall_accuracy_avg']}/5")
    print(f"    - Factual Correctness: {metrics['factual_correctness_avg']}/5")
    print(f"    - Clinical Relevance: {metrics['clinical_relevance_avg']}/5")
    print(f"    - Terminology Accuracy: {metrics['terminology_accuracy_avg']}/5")
    print(f"\nDetailed results saved to {output_dir}")


if __name__ == "__main__":
    main() 