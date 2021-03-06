import numpy as np


# FinCORE labels hierarchy
FC_CAT_UPPER = {
    "NA": "narrative",
    "OP": "opinion",
    "IN": "informational descript.",
    "ID": "discussion",
    "HI": "how-to",
    "IG": "persuation",
    "LY": "lyrical",
    "SP": "spoken",
    "OS": "other"
}


def get_tag_mappings(taglist):
    """Create a mapping from tags to indices."""
    tagset = set([tag for tags in taglist for tag in tags])
    return {tag: i for i, tag in enumerate(sorted(tagset))}


def tags_to_onehot(tags, mapping):
    "Convert list of tags to a one-hot encoded vector given a mapping."
    l = [0]*len(mapping.keys())
    for tag in tags:
        l[mapping[tag]] = 1
    return l


def fincore_to_dict_upper(path, id_prefix):
    """Convert FinCORE samples to dict format with only high-level tags."""
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            try:
                tags, text = line.strip().split("\t")
            except:
                continue

            tags = [t for t in tags.split() if t in FC_CAT_UPPER.keys()]

            data.append({
                "id": f"{id_prefix}-{i}",
                "tags": tags,
                "num_tags": len(tags),
                "text": text.strip()
            })

    return data


def get_batches(data, size):
    """Batch a list of data into a list of lists, each with length size"""
    return [data[i: i + size] for i in range(0, len(data), size)]


def sigmoid(x):
    return 1/(1 + np.exp(-x))
