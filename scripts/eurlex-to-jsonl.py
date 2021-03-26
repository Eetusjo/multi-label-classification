import json
import glob
import os

from collections import Counter

def process_files(dir, concepts, outpath):
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

    with open(outpath, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    return concept_counts

def main():
    with open("EURLEX57K/EURLEX57K.json") as f:
        concepts = json.load(f)
    for _, concept_data in concepts.items():
        concept_data["label_proc"] = concept_data["label"].replace(" ", "_")

    _ = process_files("EURLEX57K/dev/", concepts, "eurlex-dev.jsonl")
    _ = process_files("EURLEX57K/test/", concepts, "eurlex-test.jsonl")
    concept_counts = process_files("EURLEX57K/train/", concepts, "eurlex-train.jsonl")

    for tag, count in sorted(concept_counts.items(), key=lambda x: x[1]):
        print(concepts[tag]["label_proc"], count)


if __name__ == "__main__":
    main()
