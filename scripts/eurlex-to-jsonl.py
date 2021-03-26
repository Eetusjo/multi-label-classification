import json
import glob
import os

from collections import Counter

def process_files(dir, concepts):
    concept_counts = Counter()
    docs = []
    for fn in glob.glob(f"{dir}/*"):
        with open(fn, "r") as f:
            doc = json.load(f)

        for cid in doc["concepts"]:
            concept_counts[cid] += 1

        docs.append({
            "text": doc["header"] + doc["recitals"] + "\n\n" + " ".join(doc["main_body"]),
            "tags": [
                concepts[str(cid)]["label_proc"] if str(cid) in concepts.keys()
                else "<UNK>"for cid in doc["concepts"]
            ]
        })

    return concept_counts, docs


def write_jsonl(docs, path):
    with open(path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def prune(concepts, concept_counts):
    concepts = {
        cid: concept for cid, concept in concepts.items()
        if concept_counts[cid] > 1
    }
    return concepts

def main():
    with open("EURLEX57K/EURLEX57K.json") as f:
        concepts = json.load(f)
    for _, concept_data in concepts.items():
        concept_data["label_proc"] = concept_data["label"].replace(" ", "_")

    concept_counts, _ = process_files("EURLEX57K/train/", concepts)
    print("Original concepts:", len(concepts))
    concepts = prune(concepts, concept_counts)
    print("Pruned concepts", len(concepts))

    concept_counts, docs = process_files("EURLEX57K/train/", concepts)
    write_jsonl(docs, "eurlex-train.jsonl")

    _, docs = process_files("EURLEX57K/dev/", concepts)
    write_jsonl(docs, "eurlex-dev.jsonl")

    _ = process_files("EURLEX57K/test/", concepts)
    write_jsonl(docs, "eurlex-test.jsonl")

    for tag, count in sorted(concept_counts.items(), key=lambda x: x[1]):
        if tag in concepts.keys():
            print(concepts[tag]["label_proc"], count)


if __name__ == "__main__":
    main()
