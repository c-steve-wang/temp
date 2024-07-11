import argparse
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
from torch.utils.data import DataLoader
import wandb

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['tweet_text'], padding='max_length', truncation=True)

def compute_metrics(p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main(args):
    # Initialize Weights & Biases
    wandb.init(project="text-classification")

    # Load the data
    data = pd.read_csv(args.file_name)

    # Perform stratified split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['event_label'])

    # Convert to Huggingface Dataset
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Apply the tokenizer to the datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Set the format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'event_label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'event_label'])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(data['event_label'].unique()))

    # Calculate class weights
    labels = train_data['event_label'].factorize()[0]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Define the Trainer with wandb integration
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to="wandb",  # Report to wandb
    )

    # Define AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # Pass the optimizer
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print(results)

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classification model.")
    parser.add_argument("--file_name", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model from Huggingface.")
    args = parser.parse_args()

    main(args)
