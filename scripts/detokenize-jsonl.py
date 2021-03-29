import argparse
import html
import json

from mosestokenizer import MosesDetokenizer


def main(args):
    detok = MosesDetokenizer("fi")
    with open(args.infile, "r") as fi, open(args.outfile, "w") as fo:
        for line in fi:
            data = json.loads(line.strip())
            data["text"] = html.unescape(detok(data["text"].split()))
            fo.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Input jsonl file")
    parser.add_argument("outfile", type=str, help="Input jsonl file")
    args = parser.parse_args()

    main(args)
