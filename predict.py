import argparse
import json
import numpy as np
import os
import tabulate
import torch
import transformers

import utils, models, data, callbacks

from mosestokenizer import MosesSentenceSplitter
from transformers import AutoModelForSequenceClassification


def get_batches(sentences, size):
    return [sentences[i: i + size] for i in range(0, len(sentences), size)]


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def main(args):
    splits = MosesSentenceSplitter('fi')

    with open(os.path.join(args.model, "config.json"), "r") as f:
        config = json.load(f)

    id2label = config["id2label"]
    label2id = config["label2id"]
    labels = [v for k, v in sorted(id2label.items(), key=lambda x: int(x[0]))]
    tokenizer = transformers.BertTokenizer.from_pretrained(args.model)
    model = models.BertForMultiLabelSequenceClassification.from_pretrained(
        args.model, num_labels=len(id2label)
    )
    model.eval()

    with open(args.data, "r") as f:
        data = [json.loads(line.strip()) for line in f]

    if args.sentences:
        for sample in data:
            preds = []
            sents = splits([sample["text"]])
            batches = get_batches(sents, 8)
            for batch in batches:
                input = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
                result = model(
                    input_ids=input["input_ids"],
                    token_type_ids=input["token_type_ids"],
                    attention_mask=input["attention_mask"]
                )
                pred = sigmoid(result.logits.detach().numpy())
                pred = (pred >= args.classification_threshold).astype(int)
                preds.append(pred)

            preds = np.concatenate(preds, axis=0)
            docpred = list(preds.sum(axis=0))
            doctrue = [0]*len(label2id)
            for tag in sample["tags"]:
                doctrue[label2id[tag]] = 1

            print("-"*50)
            print(f"DOCUMENT: {sample['id']}, #sents: {len(sents)}")
            print(tabulate.tabulate(
                [["pred"] + docpred, ["gold"] + doctrue],
                headers=[""] + labels)
            )
            print()
            if args.print_sentences:
                for sent, inds in zip(sents, preds):
                    print(sent, end="")
                    for i, ind in enumerate(inds):
                        if ind == 1:
                            print(" " + id2label[str(i)], end="")
                    print()

            print()
        else:
            texts = [sample["text"] for sample in data]
            batches = get_batches(sents, 8)
            for batch in batches:
                input = tokenizer(batch, truncation=True, padding=True, return_tensors="pt")
                result = model(
                    input_ids=input["input_ids"],
                    token_type_ids=input["token_type_ids"],
                    attention_mask=input["attention_mask"]
                )

                print(f"DOCUMENT: {sample['id']}, #chars: {len(sents)}")
                # pred = sigmoid(result.logits.detach().numpy())
                # pred = (pred >= args.classification_threshold).astype(int)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str,
                        help="Data to be predicted in jsonl format")
    parser.add_argument("--model", required=True, type=str,
                        help="Model checkpoint to use")
    parser.add_argument("--sentences", action="store_true",
                        help="Split docs to sentences and predict on sents")
    parser.add_argument("--print_sentences", action="store_true",
                        help="Print sentences with assigned labels")
    parser.add_argument("--classification_threshold", default=0.5, type=float,
                         required=False)
    args = parser.parse_args()
    main(args)
