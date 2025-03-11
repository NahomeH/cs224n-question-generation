import random
import pandas as pd
from datasets import load_dataset, Dataset, concatenate_datasets
from IPython.display import display

# Download full dataset from Hugging Face (Train + Test)
dataset_name = "GBaker/MedQA-USMLE-4-options"
hf_dataset = load_dataset(dataset_name)

# Merge train and test sets into one dataset
merged_dataset = concatenate_datasets([hf_dataset["train"], hf_dataset["test"]])

# Convert to DataFrame
df = pd.DataFrame(merged_dataset)

# Retain only necessary columns: 'question' and 'answer'
df = df[['question', 'answer']].dropna()

# Define new prompt templates requesting both a question and an answer
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

# Define variation lists
question_actions = ["Generate", "Create", "Develop", "Write", "Formulate", "Produce", "Make"]
question_descriptions = ["a high-quality", "a challenging", "a realistic", "a board-style", "an official"]
answer_actions = ["Provide", "Include", "Give", "Provide", "State"]
answer_nouns = ["the correct answer", "the correct response", "the answer", "the solution", "the right answer", "the right response"]

# Generate a randomized prompt for each row
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

# Convert to Hugging Face dataset format for fine-tuning
processed_dataset = Dataset.from_pandas(df[['input_text', 'output_text']])

# Save the preprocessed dataset as a file (optional)
df.to_csv("preprocessed_usmle_dataset.csv", index=False)

# Display processed dataset sample
display(df.head())

print("Dataset successfully downloaded, merged, and preprocessed!")
