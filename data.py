import json
import numpy as np
import utils

from datasets import load_dataset
from skmultilearn.model_selection import iterative_train_test_split


def load_existing_split(train_data, dev_data):
    dataset = load_dataset(
        'json', data_files={
            "train": train_data,
            "dev": dev_data
          }
    )
    tag2id = utils.get_tag_mappings(dataset["train"]["tags"])
    id2tag = {v: k for k, v in tag2id.items()}
    return dataset, tag2id, id2tag


def load_and_split(data, dev_size):
    """Loads a data file and uses stratified sampling to create splits."""
    samples, ids, tags = dict(), [], []
    with open(data, "r") as f:
        for line in f:
            sample = json.loads(line.strip())
            samples[sample["id"]] = sample
            ids.append(sample["id"])
            tags.append(sample["tags"])

    tag2id = utils.get_tag_mappings(tags)
    id2tag = {v: k for k, v in tag2id.items()}

    tags = np.array([utils.tags_to_onehot(tag, tag2id) for tag in tags])
    ids = np.array(ids).reshape(-1, 1)

    train_ids, _, dev_ids, _ = iterative_train_test_split(ids, tags, dev_size)


    with open("train.tmp", "w") as f:
        for id in train_ids:
            f.write(f"{json.dumps(samples[id.item()], ensure_ascii=False)}\n")

    with open("dev.tmp", "w") as f:
        for id in dev_ids:
            f.write(f"{json.dumps(samples[id.item()], ensure_ascii=False)}\n")

    dataset = load_dataset(
        'json', data_files={
            "train": "train.tmp",
            "dev": "dev.tmp"
          }
    )

    return dataset, tag2id, id2tag
