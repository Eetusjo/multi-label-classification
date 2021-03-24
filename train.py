import argparse
import json
import numpy as np
import os
import torch
import transformers

import utils, models

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from tqdm import tqdm
from utils import fincore_to_dict_upper, fincore_tags_to_onehot

def main(args):
    if not os.path.exists("fincore.train.jsonl"):
        train = fincore_to_dict_upper("../../data/fincore-train.tsv", "train")
        dev = fincore_to_dict_upper("../../data/fincore-dev.tsv", "dev")

        with open("fincore.train.jsonl", "w") as f:
            for sample in tqdm(train):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with open("fincore.dev.jsonl", "w") as f:
            for sample in tqdm(dev):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    dataset = load_dataset(
        'json', data_files={
            "train": "fincore.train.jsonl",
            "dev": "fincore.dev.jsonl"
          }
    )

    model = models.BertForMultiLabelSequenceClassification.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1",
        num_labels=len(utils.FC_CAT_UPPER_L2I)
    )
    tokenizer = transformers.BertTokenizer.from_pretrained(
        "TurkuNLP/bert-base-finnish-cased-v1"
    )

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    args = TrainingArguments(
        "model-finsen",
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
    trainer.save_model("test-model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=False, default=8,
                        type=int, help="Training batch size")
    parser.add_argument("--epochs", required=False, default=10,
                        type=int, help="Number of training epochs")
    args = parser.parse_args()
    main(args)
