import argparse
import html
import json

from mosestokenizer import MosesSentenceSplitter, MosesDetokenizer


def main(args):
    splits = MosesSentenceSplitter('fi')
    detok = MosesDetokenizer("fi")
    with open(args.infile, "r") as fi, open(args.outfile, "w") as fo:
        for line in fi:
            data = json.loads(line.strip())
            text = html.unescape(detok(data["text"].split())) if args.moses_tokenized else data["text"]
            sents = splits([text])
            for i, s in enumerate(sents):
                d = data.copy()
                d["text"] = s
                if "id" in d.keys():
                    d["id"] = d["id"] + f"-s{i}"

                fo.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input jsonl file")
    parser.add_argument("outfile", type=str, help="Input jsonl file")
    parser.add_argument("--moses_tokenized", action="store_true",
                        help="Text in input jsonl was tokenized using moses")
    args = parser.parse_args()

    main(args)
