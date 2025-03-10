# Medical USMLE Question Generator

A project for fine-tuning BioMistral models to generate high-quality USMLE-style medical questions from clinical cases.

## Overview

This project focuses on fine-tuning [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B), a state-of-the-art biomedical language model, to generate United States Medical Licensing Examination (USMLE) style questions from clinical case descriptions. The system is designed to help medical students and professionals practice with realistic medical scenarios.

## Features

- Generate USMLE-style questions from clinical case descriptions
- Fine-tune BioMistral models on medical question datasets
- Support for memory-efficient training with quantization (4-bit/8-bit)
- Parameter-efficient fine-tuning using LoRA
- Comprehensive evaluation metrics for generated questions

## Project Structure

```
medical-qa-gen/
├── src/
│   ├── data/                # Data loading and processing
│   │   └── data_loader.py   # Dataset handling
│   ├── models/              # Model definitions
│   │   └── model.py         # Question generation model
│   ├── training/            # Training utilities
│   │   └── trainer.py       # Fine-tuning implementation
│   ├── evaluation/          # Evaluation tools
│   │   └── evaluator.py     # Metrics and quality assessment
│   ├── fine_tune.py         # Main fine-tuning script
│   └── generate_question.py # Command-line tool for generation
├── data/                    # Dataset storage (not tracked)
├── outputs/                 # Model checkpoints and outputs
├── pyproject.toml           # Project dependencies (Poetry)
└── README.md                # Project documentation
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/NahomeH/medical-qa-gen.git
cd medical-qa-gen

# Install dependencies with Poetry
poetry install
```

Alternatively, you can install with pip:

```bash
pip install -e .
```

## Usage

### Generating Questions

To generate USMLE-style questions using the pre-trained model:

```bash
# Generate a question from text
python -m src.generate_question --text "A 45-year-old female presents with sudden onset chest pain radiating to the left arm. She reports shortness of breath and nausea. Patient has a history of hypertension and Type 2 diabetes. Vital signs show BP 160/95, HR 98, RR 22."

# Generate a question from a file
python -m src.generate_question --file clinical_case.txt

# Use 8-bit quantization to save memory
python -m src.generate_question --file clinical_case.txt --quantize 8bit
```

### Fine-tuning

To fine-tune the model on your own dataset:

```bash
# Fine-tune on a dataset
python -m src.fine_tune --dataset medqa --output_dir ./outputs/fine_tuned_model

# Fine-tune with custom parameters
python -m src.fine_tune \
    --dataset ./data/my_dataset.json \
    --epochs 5 \
    --batch_size 2 \
    --lr 1e-4 \
    --quantize 8bit \
    --eval_during_training
```

## Dataset Format

The training dataset should be in one of the following formats:

1. A HuggingFace dataset name (e.g., "medqa")
2. A local JSON file with the following structure:

```json
[
  {
    "clinical_case": "Clinical case description...",
    "question": "USMLE-style question..."
  },
  ...
]
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.38+
- 8GB+ GPU memory (with 8-bit quantization)
- 16GB+ GPU memory (with 4-bit quantization)
- 40GB+ GPU memory (without quantization)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- BioMistral model by [Mistral AI](https://mistral.ai/)
- HuggingFace for the Transformers library
- PEFT library for parameter-efficient fine-tuning
