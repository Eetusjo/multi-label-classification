import json

from collections import Counter

def main():
    tagcounts = Counter()
    genres = dict()
    with open("MovieSummaries/movie.metadata.tsv", "r") as f:
        for line in f:
            line = line.strip().split("\t")
            id = line[0]
            tags = [tag for tag in json.loads(line[-1]).values()]

            for tag in tags:
                tagcounts[tag] += 1

            genres[id] = tags

    summaries = dict()
    with open("MovieSummaries/plot_summaries.txt", "r") as f:
        for line in f:
            id, summary = line.strip().split("\t")
            summaries[id] = summary

    with open("cmu-movies.jsonl", "w") as f:
        for id, summ in summaries.items():
            if id not in genres.keys():
                continue
            f.write(f'{json.dumps({"id": id, "text": summ, "tags": genres[id]})}\n')

    for tag, count in sorted(tagcounts.items(), key=lambda x: x[1]):
        print(tag, count)

if __name__ == "__main__":
    main()
