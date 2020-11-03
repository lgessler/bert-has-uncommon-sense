import argparse
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from ldg.pickle import pickle_read, pickle_write

N = 50


def precisions_at_k(df, pickle_path=None, min_train_freq=5):
    if pickle_path is not None:
        data = pickle_read(pickle_path)
        if data is not None:
            return data
    # k -> name of exact metric (incl. total) -> count
    # Dict[int, Dict[str, int]]

    # for pickles
    def f1():
        def f2():
            return 0
        return defaultdict(f2)
    correct_at_k = defaultdict(f1)
    
    for _, row in tqdm(df.iterrows()):
        label = row.label
        synset = row.synset
        lemma = row.lemma
        if row.label_freq_in_train < min_train_freq:
            continue
        
        for k in range(1, N+1):
            labels = [getattr(row, f"label_{i}") for i in range(1, k+1)]
            synsets = [getattr(row, f"synset_{i}") for i in range(1, k+1)]
            lemmas = [getattr(row, f"lemma_{i}") for i in range(1, k+1)]
            correct_at_k[k]['label'] +=  len([x for x in labels if x == label])
            correct_at_k[k]['synset'] += len([x for x in synsets if x == synset])
            correct_at_k[k]['lemma'] +=  len([x for x in lemmas if x == lemma])
            correct_at_k[k]['total'] += k

    ps_at_k = defaultdict(lambda: dict())
    for k in range(1, N+1):
        for l in ['label', 'synset', 'lemma']:
            ps_at_k[k][l] = correct_at_k[k][l] / correct_at_k[k]['total'] 

    if pickle_path is not None:
        ps_at_k = dict(ps_at_k)
        for key, value in ps_at_k.items():
            ps_at_k[key] = dict(value)
        pickle_write(ps_at_k, pickle_path)
    return ps_at_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions_tsv")
    args = p.parse_args()

    df = pd.read_csv(args.predictions_tsv, sep="\t")
    for train_freq in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        precisions_at_k(df, pickle_path=args.predictions_tsv + f'.{train_freq}.pkl', min_train_freq=train_freq)


if __name__ == '__main__':
    main()
