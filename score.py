import argparse
import csv
import pandas as pd

N = 5


def score(tsv_path):
    at_least_one_label = 0
    at_least_one_synset = 0

    df = pd.read_csv(tsv_path, sep="\t")
    for i, row in df.iterrows():
        retrieved_labels = [getattr(row, f'label_{i+1}') for i in range(N)]
        retrieved_synsets = [getattr(row, f'synset_{i+1}') for i in range(N)]
        retrieved_lemma = [getattr(row, f'lemma_{i+1}') for i in range(N)]

        if any(row.synset == synset for synset in retrieved_synsets):
            at_least_one_synset += 1
        if any(row.label == label for label in retrieved_labels):
            at_least_one_label += 1

    print(f"at least one synset: {at_least_one_synset / len(df.index)}")
    print(f"at least one label:  {at_least_one_label / len(df.index)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions_tsv")
    args = p.parse_args()
    score(args.predictions_tsv)


if __name__ == '__main__':
    main()