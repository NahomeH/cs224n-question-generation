# USMLE Question Generation Evaluation

This repository contains tools for fine-tuning language models to generate high-quality USMLE-style questions and evaluating their medical accuracy.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. For GPU acceleration (recommended):
```bash
conda env create -f env-gpu.yml
conda activate usmle-question-gen
```

## Medical Accuracy Evaluation

The `evaluate_medical_accuracy.py` script allows you to evaluate the medical accuracy of generated USMLE questions from your fine-tuned model.

### How it works

The script:
1. Takes input texts from your dataset
2. Uses your fine-tuned model to generate USMLE-style questions
3. Evaluates the medical accuracy of the generated questions in isolation
4. Generates comprehensive evaluation reports

### Usage

Using the script is simple - just run it directly with no arguments:

```bash
python src/evaluate_medical_accuracy.py
```

### Default Configuration

The script uses the following defaults specific to our use case:

- **Fine-tuned model path**: "./outputs/final"
- **Evaluation model**: "stanford-crfm/BioMedLM"
- **Dataset path**: "src/data/processed_data/preprocessed_usmle_dataset.csv"
- **Output directory**: "./evaluation_results"
- **Batch size**: 8
- **Samples to evaluate**: All samples in dataset

If you need to modify any of these settings, you can edit the values directly in the `main()` function of the script.

### Output

The script generates three types of output files:
1. `evaluation_results_[timestamp].json`: Detailed evaluation data for each sample
2. `evaluation_metrics_[timestamp].json`: Summary metrics in JSON format
3. `evaluation_report_[timestamp].txt`: Human-readable evaluation report with examples

## Evaluation Metrics

The generated questions are evaluated for their standalone medical accuracy across three dimensions:
1. **Factual Correctness**: Accuracy of medical facts according to current medical knowledge (1-5 scale)
2. **Clinical Relevance**: Appropriateness and relevance to medical practice (1-5 scale)
3. **Terminology Accuracy**: Correct usage of medical terminology (1-5 scale)

The evaluation focuses solely on the medical correctness of the generated questions themselves, without comparing them to the input prompts.

## Medical Evaluation Model

The script uses Stanford's BioMedLM model for evaluation to ensure proper assessment of medical accuracy. This is a specialized biomedical language model developed by Stanford's Center for Research on Foundation Models (CRFM) with expertise in:

- Medical research and literature
- Clinical terminology
- Medical facts and relationships
- Medical context understanding

### Model Loading

The evaluation model is automatically downloaded from the Hugging Face Hub when you run the script. No additional code or manual downloads are required! The script uses the Hugging Face `pipeline` function which handles:

1. Downloading the model files (first time only)
2. Loading the model into memory
3. Setting up the appropriate device (GPU/CPU)
4. Configuring inference parameters

All this happens behind the scenes when you initialize the `MedicalAccuracyEvaluator` class.

## Customizing the Evaluation

If you need to modify the evaluation criteria, you can edit the evaluation prompt template in the `_evaluate_medical_accuracy` method of the `MedicalAccuracyEvaluator` class. 