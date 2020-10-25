import argparse
import csv
from collections import defaultdict

import pandas as pd
import re

N = 50


def score(tsv_path):
    at_least_one_pos_counts = defaultdict(lambda: {"label": 0, "synset": 0, "lemma": 0, "total": 0})
    at_least_one_label = 0
    at_least_one_synset = 0
    at_least_one_lemma = 0
    no_pos_tag = 0

    df = pd.read_csv(tsv_path, sep="\t")
    for i, row in df.iterrows():
        retrieved_labels = [getattr(row, f'label_{i+1}') for i in range(N)]
        retrieved_synsets = [getattr(row, f'synset_{i+1}') for i in range(N)]
        retrieved_lemmas = [getattr(row, f'lemma_{i+1}') for i in range(N)]

        try:
            pos = re.findall(r"\.(.*)\.", row.label)[0]
        except:
            no_pos_tag += 1
            assert row.label == "NE"
            pos = "NE"
        #print(row.sentence)
        #print(row.label, retrieved_labels)
        #print(row.synset, retrieved_synsets)
        #print(row.lemma, retrieved_lemmas)
        #print()
        at_least_one_pos_counts[pos]["total"] += 1

        if any(row.label == label for label in retrieved_labels):
            at_least_one_label += 1
            at_least_one_pos_counts[pos]['label'] += 1
        if any(row.synset == synset for synset in retrieved_synsets):
            at_least_one_synset += 1
            at_least_one_pos_counts[pos]['synset'] += 1
        if any(row.lemma == lemma for lemma in retrieved_lemmas):
            at_least_one_lemma += 1
            at_least_one_pos_counts[pos]['lemma'] += 1

    n = len(df.index)
    print(f"at least one label:  {at_least_one_label / n}")
    print(f"at least one synset: {at_least_one_synset / n}")
    print(f"at least one lemma:  {at_least_one_lemma / n}")
    print(f"no pos tag:          {no_pos_tag} ({no_pos_tag / n * 100}%)")

    for pos, dist in at_least_one_pos_counts.items():
        n = dist['total']
        print()
        print(f"pos: {pos}")
        print(f"at least one label:  {dist['label'] / n}")
        print(f"at least one synset: {dist['synset'] / n}")
        print(f"at least one lemma:  {dist['lemma'] / n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions_tsv")
    args = p.parse_args()
    score(args.predictions_tsv)


if __name__ == '__main__':
    main()