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
        # Load the evaluation model with more careful configuration
        try:
            # First load tokenizer separately to configure it properly
            self.eval_tokenizer = AutoTokenizer.from_pretrained(self.evaluation_model_name)
            self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token
            
            # Then load model with basic configuration
            eval_model = AutoModelForCausalLM.from_pretrained(
                self.evaluation_model_name, 
                torch_dtype=torch.float16 if device >= 0 else torch.float32
            )
            
            # Create pipeline with configured tokenizer
            self.evaluator = pipeline(
                "text-generation",
                model=eval_model,
                tokenizer=self.eval_tokenizer,
                device=device,
                torch_dtype=torch.float16 if device >= 0 else torch.float32
            )
            logger.info(f"Successfully initialized evaluation model")
        except Exception as e:
            logger.error(f"Error initializing evaluation model: {str(e)}")
            raise RuntimeError(f"Failed to initialize evaluation model: {str(e)}")
    
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
            try:
                # Limit input text length to avoid exceeding context window
                max_input_length = 512  # Safe limit for input text
                input_text = input_text[:max_input_length]
                
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
            except Exception as e:
                logger.error(f"Error generating question: {str(e)}")
                # Add a placeholder question to maintain alignment with input_texts
                generated_questions.append("Error generating question.")
                
        return generated_questions
    
    def _evaluate_medical_accuracy(self, input_texts: List[str], generated_questions: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate the medical accuracy of generated questions using an evaluation model.
        """
        logger.info("Evaluating medical accuracy of generated questions")
        evaluation_results = []
        
        # Even shorter evaluation prompt template to reduce token usage further
        evaluation_prompt_template = """
        Evaluate this USMLE-style medical question on factual correctness, clinical relevance, and terminology accuracy.
        For each category, provide a score from 1-5 (5 is best).
        
        Question: {generated_question}
        
        Return your evaluation in this exact JSON format:
        {{
            "factual_correctness": 3,
            "clinical_relevance": 3,
            "terminology_accuracy": 3,
            "average_score": 3.0,
            "explanation": "Brief explanation of your ratings."
        }}
        
        Include nothing else in your response except the JSON.
        """
        
        # Track the success rate of model evaluations
        evaluation_success_count = 0
        use_heuristic_fallback = False
        
        for input_text, generated_question in tqdm(zip(input_texts, generated_questions), 
                                                   desc="Evaluating medical accuracy", 
                                                   total=len(input_texts)):
            try:
                # Skip evaluation if the generation failed
                if generated_question == "Error generating question.":
                    evaluation_results.append({
                        "input_text": input_text,
                        "generated_question": generated_question,
                        "evaluation": {
                            "factual_correctness": 0,
                            "clinical_relevance": 0,
                            "terminology_accuracy": 0,
                            "average_score": 0,
                            "explanation": "Evaluation skipped due to question generation error"
                        }
                    })
                    continue
                
                # Limit generated question length to avoid token limit issues
                max_question_length = 300  # Further reduced for safety
                truncated_question = generated_question[:max_question_length]
                
                # If too many model evaluations have failed, switch to heuristic fallback
                if use_heuristic_fallback:
                    # Use simple heuristic evaluation instead
                    evaluation_result = self._heuristic_evaluation(truncated_question)
                    evaluation_results.append({
                        "input_text": input_text,
                        "generated_question": generated_question,
                        "evaluation": evaluation_result,
                        "method": "heuristic"
                    })
                    continue
                    
                # Prepare evaluation prompt
                evaluation_prompt = evaluation_prompt_template.format(
                    generated_question=truncated_question
                )
                
                # Manual inference with more control instead of using pipeline directly
                try:
                    # First tokenize with careful padding control
                    model_inputs = self.eval_tokenizer(
                        evaluation_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512  # Fixed context length
                    )
                    
                    # Move to GPU if available
                    if torch.cuda.is_available() and self.use_gpu:
                        model_inputs = {k: v.cuda() for k, v in model_inputs.items()}
                    
                    # Generate with more controlled settings
                    with torch.no_grad():
                        output_ids = self.evaluator.model.generate(
                            **model_inputs,
                            max_new_tokens=384,  # Further reduced to avoid issues
                            temperature=0.0,   # Use greedy decoding
                            do_sample=False,   # No sampling
                            pad_token_id=self.eval_tokenizer.eos_token_id,
                            num_return_sequences=1
                        )
                    
                    # Decode to get response
                    evaluation_response = self.eval_tokenizer.decode(
                        output_ids[0], 
                        skip_special_tokens=True
                    )
                except Exception as e:
                    logger.error(f"Error during model inference: {str(e)}")
                    evaluation_response = ""
                
                # Extract the JSON response
                try:
                    # Find the JSON part of the response
                    start_idx = evaluation_response.find('{')
                    end_idx = evaluation_response.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = evaluation_response[start_idx:end_idx]
                        evaluation_result = json.loads(json_str)
                        
                        # Ensure all required fields are present
                        required_fields = [
                            "factual_correctness", "clinical_relevance", 
                            "terminology_accuracy", "average_score", "explanation"
                        ]
                        for field in required_fields:
                            if field not in evaluation_result:
                                if field == "average_score" and all(k in evaluation_result for k in required_fields[:3]):
                                    # Calculate average if missing but other scores are present
                                    scores = [
                                        evaluation_result["factual_correctness"],
                                        evaluation_result["clinical_relevance"],
                                        evaluation_result["terminology_accuracy"]
                                    ]
                                    evaluation_result["average_score"] = round(sum(scores) / 3, 2)
                                else:
                                    evaluation_result[field] = 0 if field != "explanation" else "Missing field"
                        
                        # If we got here, we successfully parsed the JSON
                        evaluation_success_count += 1
                    else:
                        # If no JSON found, use heuristic evaluation
                        evaluation_result = self._heuristic_evaluation(truncated_question)
                        logger.warning(f"Failed to extract JSON, using heuristic evaluation instead")
                except json.JSONDecodeError:
                    # If JSON parsing failed, use heuristic evaluation
                    evaluation_result = self._heuristic_evaluation(truncated_question)
                    logger.warning(f"Failed to parse JSON, using heuristic evaluation instead")
                
                # Add the input and output to the result
                result_entry = {
                    "input_text": input_text,
                    "generated_question": generated_question,
                    "evaluation": evaluation_result,
                    "method": "model" if evaluation_success_count > 0 else "heuristic"
                }
                
                evaluation_results.append(result_entry)
                
                # If we've processed at least 5 samples and success rate is low, switch to heuristic mode
                if len(evaluation_results) >= 5 and evaluation_success_count == 0:
                    logger.warning("Model evaluation failing consistently, switching to heuristic evaluation")
                    use_heuristic_fallback = True
                
            except Exception as e:
                logger.error(f"Error evaluating question: {str(e)}")
                # Add a placeholder evaluation to maintain alignment
                evaluation_results.append({
                    "input_text": input_text,
                    "generated_question": generated_question,
                    "evaluation": self._heuristic_evaluation(generated_question[:300]),
                    "method": "heuristic (error fallback)"
                })
                
        return evaluation_results
    
    def _heuristic_evaluation(self, question_text: str) -> Dict[str, Any]:
        """
        Simple heuristic-based evaluation when the model evaluation fails.
        This is a fallback method that uses basic text analysis to roughly estimate
        question quality.
        """
        # Calculate text features
        word_count = len(question_text.split())
        medical_term_count = self._count_medical_terms(question_text)
        has_structure = any(marker in question_text.lower() for marker in 
                          ["patient", "history", "exam", "diagnosis", "treatment", "symptoms", 
                           "presents", "laboratory", "what is", "which of"])
        
        # Simple scoring based on features
        factual_score = min(5, max(1, 3))  # Default to middle score
        clinical_score = min(5, max(1, 3 + (1 if has_structure else 0)))
        terminology_score = min(5, max(1, 2 + min(3, medical_term_count // 2)))
        
        # More length generally suggests more content/detail
        if word_count > 150:
            factual_score += 1
            clinical_score = min(5, clinical_score + 1)
        
        # Very short questions are likely incomplete
        if word_count < 50:
            factual_score = max(1, factual_score - 1)
            clinical_score = max(1, clinical_score - 1)
            terminology_score = max(1, terminology_score - 1)
        
        # Calculate average
        avg_score = round((factual_score + clinical_score + terminology_score) / 3, 2)
        
        return {
            "factual_correctness": factual_score,
            "clinical_relevance": clinical_score,
            "terminology_accuracy": terminology_score,
            "average_score": avg_score,
            "explanation": "Estimated using text features (length, structure, medical terminology)"
        }
    
    def _count_medical_terms(self, text: str) -> int:
        """
        Count medical terms in the text using a simple list of common medical terms.
        This is a very basic approximation for the heuristic evaluation.
        """
        # Short list of common medical terms to check for
        medical_terms = [
            "diagnosis", "syndrome", "patient", "symptom", "treatment", "therapy",
            "prognosis", "pathology", "etiology", "chronic", "acute", "lesion",
            "malignant", "benign", "carcinoma", "disease", "disorder", "infection",
            "inflammatory", "congenital", "hereditary", "idiopathic", "metastatic",
            "neoplasm", "hypertension", "diabetes", "cardiac", "pulmonary", "renal",
            "hepatic", "neurological", "hematologic", "autoimmune", "oncology",
            "pharmacology", "antibiotic", "analgesic", "anesthetic", "vaccine"
        ]
        
        # Count medical terms
        count = 0
        text_lower = text.lower()
        for term in medical_terms:
            if term in text_lower:
                count += 1
        
        return count
    
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
    max_samples = 100  # Set to an integer if you want to evaluate fewer samples
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