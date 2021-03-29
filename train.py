import argparse
import json
import numpy as np
import os
import torch
import torch.nn.functional as F
import transformers

import utils, models, data, callbacks

from collections import defaultdict
from datasets import load_metric
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def main(args):
    if not args.dev_data:
        dataset, tag2id, id2tag = data.load_and_split(args.train_data, args.dev_size)
    else:
        dataset, tag2id, id2tag = data.load_existing_split(args.train_data, args.dev_data)

    tokenizer = transformers.BertTokenizer.from_pretrained(args.model)
    model = models.BertForMultiLabelSequenceClassification.from_pretrained(
        args.model, num_labels=len(tag2id)
    )

    model.config.id2label, model.config.label2id = id2tag, tag2id

    if args.pos_weight:
        model.set_pos_weight(args.pos_weight)
    elif args.label_weights:
        model.set_label_weights(args.label_weights)

    if args.freeze_bert:
        model.freeze_bert()

    def preprocess_simple(examples):
        result = tokenizer(examples["text"], truncation=True)
        result["labels"] = [
            utils.tags_to_onehot(tags, tag2id) for tags in examples["tags"]
        ]
        return result

    def preprocess_sents(examples):
        raise NotImplementedError("Sents-strategy not implemented.")
        result = tokenizer(examples["text"], truncation=True)
        result["labels"] = [
            utils.tags_to_onehot(tags, tag2id) for tags in examples["tags"]
        ]
        return result

    def preprocess_blocks(examples):
        result = defaultdict(list)
        for i, (id, text, tags) in enumerate(zip(examples["id"], examples["text"], examples["tags"])):
            tokenized = tokenizer(text)
            blocks = utils.get_batches(
                tokenized["input_ids"][1:-1],
                args.max_block_len - 3
            )
            for j, block in enumerate(blocks):
                # Use unused token no. 1 as end-of-block signal
                result["input_ids"].append([102] + block + [1, 103])
                result["token_type_ids"].append([0]*(len(block) + 3))
                result["attention_mask"].append([1]*(len(block) + 3))
                # Duplicate tags len(blocks) times
                result["tags"].append(tags)
                result["id"].append(f"{id}-b{j}")
                result["text"].append(tokenizer.decode(block, skip_special_tokens=True))
                # Duplicate any other values found in the examples dict
                for key, value in examples.items():
                    if key not in ["id", "text", "tags"]:
                        result[key].append(value[i])

        result["labels"] = [
            utils.tags_to_onehot(tags, tag2id) for tags in result["tags"]
        ]

        return result

    pp_fn = {
        "simple": preprocess_simple,
        "sentences": preprocess_sents,
        "blocks": preprocess_blocks
    }

    preprocess_function =  pp_fn[args.preprocess_type]
    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.shuffle()

    train_args = TrainingArguments(
        output_dir=args.save_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_model_metric,
        logging_first_step=True
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = (sigmoid(predictions) >= args.classification_threshold).astype(int)
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_micro = f1_score(labels, predictions, average="micro")
        f1_samples = f1_score(labels, predictions, average="samples")
        f1_weighted = f1_score(labels, predictions, average="weighted")
        return {
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_samples": f1_samples,
            "f1_weighted": f1_weighted
        }

    trainer = Trainer(
        model,
        train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if args.patience:
        trainer.add_callback(transformers.EarlyStoppingCallback(args.patience))

    # Remove default mlflow callback and add a custom one callback if needed
    trainer.remove_callback(transformers.integrations.MLflowCallback)
    if not args.no_mlflow:
        mlf_callback = callbacks.MLflowCustomCallback(
            experiment=args.mlflow_experiment,
            run=args.mlflow_run,
            register_best=args.register_best_model
        )
        trainer.add_callback(mlf_callback)

    trainer.train()
    trainer.save_model(os.path.join(args.save_dir, "checkpoint-best"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, type=str,
                        default="TurkuNLP/bert-base-finnish-cased-v1",
                        help="Pretrained model name or a checkpoint dir")
    parser.add_argument("--train_data", required=True, type=str,
                        help="Path to a train data file in jsonl format")
    parser.add_argument("--dev_data", required=False, type=str, default=None,
                        help="Path to a dev data file in jsonl format")
    parser.add_argument("--dev_size", required=False, type=float, default=0.1,
                        help="Percentage of full data to be used as dev data "
                             "if no separate dev set is supplied.")
    parser.add_argument("--preprocess_type", default="simple", required=False,
                        help="Choose how to handle docs longer than max len")
    parser.add_argument("--classification_threshold", required=False,
                        type=float, default=0.5,
                        help="Threshold in (0, 1) for deciding class label.")
    parser.add_argument("--max_block_len", required=False, default=512, type=int,
                        help="Maximum length of a tokenized text block if "
                             "preprocessing data using the 'blocks' strategy. "
                             "Will include special tokens.")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--label_weights", required=False,
                       type=str, default=None,
                       help="Path to json file with label weights")
    group.add_argument("--pos_weight", required=False,
                       type=float, default=None,
                       help="Use a single label weight for all pos samples")

    parser.add_argument("--learning_rate", required=False,
                        type=float, default=2e-5,
                        help="Optimizer learning rate")
    parser.add_argument("--weight_decay", required=False,
                        type=float, default=0.01,
                        help="Weight decay regularization parameter")
    parser.add_argument("--batch_size", required=False, default=8,
                        type=int, help="Training batch size")
    parser.add_argument("--epochs", required=False, default=10,
                        type=int, help="Number of training epochs")
    parser.add_argument("--max_steps", required=False, default=-1,
                        type=int, help="Number of training steps")
    parser.add_argument("--eval_steps", required=False, default=100,
                        type=int, help="Step interval for evaluation")
    parser.add_argument("--save_steps", required=False, default=100,
                        type=int, help="Step interval for model saving")
    parser.add_argument("--save_dir", required=True, type=str,
                        help="Directory for saving a model.")
    parser.add_argument("--mlflow_experiment", required=False,
                        default="default", type=str,
                        help="Experiment under which to log run")

    parser.add_argument("--mlflow_run", required=False, type=str, default=None,
                        help="Run name for mlflow")
    parser.add_argument("--best_model_metric", required=False, type=str,
                        default="f1_samples",
                        help="Evaluation metric for deciding on best model.")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="Freeze bert weights and only train classifier")
    parser.add_argument("--no_mlflow", action="store_true",
                        help="Do not log the run using mlflow")
    parser.add_argument("--patience", required=False, default=None,
                        type=int, help="Early stopping patience")
    parser.add_argument("--register_best_model", action="store_true",
                        help="Register best model in mlflow model registry")

    args = parser.parse_args()
    main(args)
