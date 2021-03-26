#!/bin/bash

wget http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
tar xvzf MovieSummaries.tar.gz && rm MovieSummaries.tar.gz
rm MovieSummaries/{character.metadata.tsv,name.clusters.txt,tvtropes.clusters.txt}

python cmu-movie-to-jsonl.py

rm -rf MovieSummaries
