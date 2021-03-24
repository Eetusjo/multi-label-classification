import argparse
import json
import numpy as np
import os
import torch
import transformers

import utils, models, data

from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm import tqdm

def main(args):
    if not args.dev_data:
        dataset, tag2id, id2tag = data.load_and_split(args.train_data, args.dev_size)
    else:
        dataset, tag2id, id2tag = data.load_existing_split(args.train_data, args.dev_data)

    model = models.BertForMultiLabelSequenceClassification.from_pretrained(
        args.model, num_labels=len(tag2id)
    )
    model.config.id2label = id2tag
    model.config.label2id = tag2id

    tokenizer = transformers.BertTokenizer.from_pretrained(args.model)

    def preprocess_function(examples):
        result = tokenizer(examples["text"], truncation=True)
        result["labels"] = [
            utils.tags_to_onehot(tags, tag2id) for tags in examples["tags"]
        ]
        return result

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        accuracy = ((predictions > 0.5) == labels).sum()/labels.size
        return {"accuracy": accuracy}

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    #trainer.save_model("test-model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, type=str,
                        help="Path to a train data file in jsonl format")
    parser.add_argument("--dev_data", required=False, type=str, default=None,
                        help="Path to a dev data file in jsonl format")
    parser.add_argument("--dev_size", required=False, type=float, default=0.1,
                        help="Percentage of full data to be used as dev data "
                             "if no separate dev set is supplied.")
    parser.add_argument("--batch_size", required=False, default=8,
                        type=int, help="Training batch size")
    parser.add_argument("--epochs", required=False, default=10,
                        type=int, help="Number of training epochs")
    parser.add_argument("--save_dir", required=True, type=str,
                        help="Directory for saving a model.")
    parser.add_argument("--model", required=False, type=str,
                        default="TurkuNLP/bert-base-finnish-cased-v1",
                        help="Pretrained model name or a checkpoint dir")

    args = parser.parse_args()
    main(args)
