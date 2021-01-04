import argparse
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from ldg.pickle import pickle_read, pickle_write
import nltk
from nltk.corpus import wordnet as wn

N = 50

def get_supersense(synset):
    try:
        return wn.synset(synset).lexname()
    except nltk.corpus.reader.wordnet.WordNetError:
        print("OUCH", synset)
        return 'unk'


def pr_at_k(df, pickle_path=None, min_train_freq=5, rarity_threshold=1.0):
    #if pickle_path is not None:
    #    data = pickle_read(pickle_path)
    #    if data is not None:
    #        return data
    # k -> name of exact metric (incl. total) -> count
    # Dict[int, Dict[str, int]]

    # for pickles
    def f1():
        def f2():
            return 0
        return defaultdict(f2)
    correct_at_k = defaultdict(f1)

    pickle_path += 'rarity%d' % (100 * rarity_threshold)

    for _, row in tqdm(df.iterrows()):
        label = row.label
        synset = row.synset
        lemma = row.lemma
        if len(synset.split(".")) == 3:
            get_supersense(synset)
        rarity = LABEL_FREQS[label] / LEMMA_FREQS[lemma]
        if rarity > rarity_threshold:
            continue

        for k in range(1, N + 1):
            if row.label_freq_in_train < min_train_freq:
                continue

            correct_at_k[k]['label'] += (getattr(row, f'label_{k}') == label)
            correct_at_k[k]['synset'] += (getattr(row, f'synset_{k}') == synset)
            correct_at_k[k]['lemma'] += (getattr(row, f'lemma_{k}') == lemma)

            correct_at_k[k]['total'] += k
            correct_at_k[k]['label_total'] += (LABEL_FREQS[label])
            correct_at_k[k]['synset_total'] += (SYNSET_FREQS[synset])
            correct_at_k[k]['lemma_total'] += (LEMMA_FREQS[lemma])


    ps_at_k = defaultdict(lambda: dict())
    for k in range(1, N+1):
        for l in ['label', 'synset', 'lemma']:
            correct_at_k[k][l] += correct_at_k[k-1][l]
            ps_at_k[k][l] = correct_at_k[k][l] / correct_at_k[k]['total']

    recalls_at_k = defaultdict(lambda: dict())
    for k in range(1, N+1):
        for l in ['label', 'synset', 'lemma']:
            recalls_at_k[k][l] = correct_at_k[k][l] / correct_at_k[k][l + '_total']


    #if pickle_path is not None:
    ps_at_k = dict(ps_at_k)
    for key, value in ps_at_k.items():
        ps_at_k[key] = dict(value)
    pickle_write(ps_at_k, pickle_path + '.prec')

    recalls_at_k = dict(recalls_at_k)
    for key, value in recalls_at_k.items():
        recalls_at_k[key] = dict(value)
    pickle_write(recalls_at_k, pickle_path + '.rec')
    return recalls_at_k, recalls_at_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions_tsv")
    args = p.parse_args()

    df = pd.read_csv(args.predictions_tsv, sep="\t")
    # TODO: probably this should be synset cutoff, not label cutoff
    for train_freq in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        for rarity_threshold in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            print("Cutoff:", train_freq)
            pr_at_k(df, pickle_path=args.predictions_tsv + f'.{train_freq}.pkl',
                    min_train_freq=train_freq, rarity_threshold=rarity_threshold)


if __name__ == '__main__':
    with open('synset_freqs.tsv', 'r') as f:
        SYNSET_FREQS = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}
    with open('lemma_freqs.tsv', 'r') as f:
        LEMMA_FREQS = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}
    with open('label_freqs.tsv', 'r') as f:
        LABEL_FREQS = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}

    main()
