import argparse
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

def create_sample_data():
    """Create sample NER training data for demonstration purposes"""
    # Sample data in IOB format (Inside-Outside-Beginning)
    data = {
        "tokens": [
            ["There", "is", "a", "cat", "in", "the", "picture"],
            ["I", "see", "a", "dog", "and", "a", "cow"],
            ["The", "elephant", "is", "huge"],
            ["A", "zebra", "is", "in", "the", "image"],
            ["No", "animals", "here"]
        ],
        "ner_tags": [
            [0, 0, 0, 1, 0, 0, 0],  # B-ANIMAL for "cat"
            [0, 0, 0, 1, 0, 0, 1],  # B-ANIMAL for "dog" and "cow"
            [0, 1, 0, 0],           # B-ANIMAL for "elephant"
            [0, 1, 0, 0, 0, 0],     # B-ANIMAL for "zebra"
            [0, 0, 0]               # No animals
        ]
    }
    return data

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """Tokenize and align labels for NER training"""
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config JSON file', default=None)
    parser.add_argument('--output_dir', default='models/ner', help='Output directory for the trained model')
    parser.add_argument('--model_name', default='dslim/bert-base-NER', help='Pretrained model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        args.output_dir = cfg.get('ner_output_dir', args.output_dir)
        args.model_name = cfg.get('ner_model_name', args.model_name)
        args.epochs = cfg.get('ner_epochs', args.epochs)

    print(f"Training NER model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")

    # Create sample data (in a real scenario, you would load your dataset)
    sample_data = create_sample_data()
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        "tokens": sample_data["tokens"],
        "ner_tags": sample_data["ner_tags"]
    })
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2)
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f'Saved NER model to: {args.output_dir}')

if __name__ == '__main__':
    main()