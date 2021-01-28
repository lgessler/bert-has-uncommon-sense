"""
This file will:
- Read the train, test, dev splits of ontonotes (see TRAIN_FILEPATH, etc. below) into allennlp datasets
- Get BERT embeddings for words with word senses annotations in the train split
- Use a model to predict embeddings from the dev+test splits and use a similarity function to find similar instances in train
- Write the top 50 results for each instance in dev+test to a tsv with information
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import torch
import pandas as pd
import tqdm
from allennlp.data import Vocabulary
from ldg.pickle import pickle_write

from bssp.common.reading import read_dataset_cached, indexer_for_embedder, embedder_for_embedding
from bssp.common.util import dataset_stats
from bssp.common.nearest_neighbor_models import NearestNeighborRetriever, NearestNeighborPredictor, format_sentence
from bssp.ontonotes.dataset_reader import OntonotesReader, lemma_from_label

TRAIN_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/train'
DEVELOPMENT_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/development'
TEST_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/test'

# TODO: look into whether senses are from ontonotes or propbank
# - query instances to find instances that have a lemma with a certain number of senses and with a certain number of lemmas
# - Lemma restriction so we don't get distracted by other words
# - Bucket by lemma frequency and then use MAP etc to examine rarity
# TODO: refer to email and figure out how words were chosen for sense annotation


def read_datasets(embedding_name):
    train_dataset = read_dataset_cached(
        OntonotesReader, TRAIN_FILEPATH, 'ontonotes', 'train', embedding_name, with_embeddings=True
    )
    print(train_dataset[0])
    dev_dataset = read_dataset_cached(
        OntonotesReader, DEVELOPMENT_FILEPATH, 'ontonotes', 'dev', embedding_name, with_embeddings=False
    )
    test_dataset = read_dataset_cached(
        OntonotesReader, TEST_FILEPATH, 'ontonotes', 'test', embedding_name, with_embeddings=False
    )
    return train_dataset, dev_dataset + test_dataset


def stats(train_dataset, test_dataset):
    train_labels, train_lemmas = dataset_stats('train', train_dataset, "ontonotes_stats", lemma_from_label)
    testdev_labels, testdev_lemmas = dataset_stats('testdev', test_dataset, "ontonotes_stats", lemma_from_label)

    return train_labels, train_lemmas


def predict(embedding_name, distance_metric, top_n):
    predictions_path = f'cache/ontonotes_{distance_metric}_predictions/{embedding_name.replace("embeddings/", "")}.tsv'
    if os.path.isfile(predictions_path):
        print(f"Reading predictions from {predictions_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, testdev_dataset = read_datasets(embedding_name)
    train_label_counts, train_lemma_counts = stats(train_dataset, testdev_dataset)

    indexer = indexer_for_embedder(embedding_name)
    vocab, embedder = embedder_for_embedding(embedding_name)

    # we're using a `transformers` model
    label_vocab = Vocabulary.from_instances(train_dataset)
    label_vocab.extend_from_instances(testdev_dataset)
    try:
        del label_vocab._token_to_index['tokens']
    except KeyError:
        pass
    try:
        del label_vocab._index_to_token['tokens']
    except KeyError:
        pass
    vocab.extend_from_vocab(label_vocab)

    print("Constructing model")
    model = NearestNeighborRetriever(
        vocab=vocab,
        embedder=embedder,
        target_dataset=train_dataset,
        distance_metric=distance_metric,
        device=device,
        top_n=top_n,
        same_lemma=True,
    ).eval().to(device)
    dummy_reader = OntonotesReader(split='train', token_indexers={'tokens': indexer})
    predictor = NearestNeighborPredictor(model=model, dataset_reader=dummy_reader)

    os.makedirs(f'cache/ontonotes_{distance_metric}_predictions', exist_ok=True)
    with open(predictions_path, 'wt') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        header = ['sentence', 'label', 'lemma', 'label_freq_in_train']
        header += [f"label_{i+1}" for i in range(top_n)]
        header += [f"lemma_{i+1}" for i in range(top_n)]
        header += [f"sentence_{i+1}" for i in range(top_n)]
        header += [f"distance_{i+1}" for i in range(top_n)]
        tsv_writer.writerow(header)

        for instance in tqdm.tqdm([i for i in testdev_dataset if train_label_counts[i['label'].label] >= 5]):
            d = predictor.predict_instance(instance)
            sentence = [t.text for t in instance['text'].tokens]
            span = instance['label_span']
            sentence = format_sentence(sentence, span.span_start, span.span_end)
            label = instance['label'].label
            lemma = lemma_from_label(label)
            label_freq_in_train = train_label_counts[instance['label'].label]

            row = [sentence, label, lemma, label_freq_in_train]
            results = d[f'top_{top_n}']
            results += [None for _ in range(top_n - len(results))]
            row += [i['label'] if i is not None else "" for i in results]
            row += [lemma_from_label(i['label']) if i is not None else "" for i in results]
            row += [i['sentence'] if i is not None else "" for i in results]
            row += [i['distance'] if i is not None else 888888 for i in results]
            if len(row) != 204:
                print(len(row))
                print(instance)
                assert False
            tsv_writer.writerow(row)

    print(f"Wrote predictions to {predictions_path}")


def pr_at_k(df, label_freqs, lemma_freqs, top_n, pickle_path, min_train_freq, max_train_freq, min_rarity, max_rarity):
    # for pickles
    def f1():
        def f2():
            return 0
        return defaultdict(f2)
    correct_at_k = defaultdict(f1)

    no_data = True
    for _, row in tqdm.tqdm(df.iterrows()):
        label = row.label
        lemma = row.label[:row.label.rfind('_')]
        rarity = label_freqs[label] / lemma_freqs[lemma]
        if not (min_rarity <= rarity < max_rarity):
            continue
        if not (min_train_freq <= row.label_freq_in_train < max_train_freq):
            continue
        no_data = False

        num_labels_correct = 0
        for k in range(1, top_n + 1):

            label_is_correct = getattr(row, f'label_{k}') == label
            correct_at_k[k]['label'] += 1 if label_is_correct else 0
            num_labels_correct += 1 if label_is_correct else 0
            correct_at_k[k]['lemma'] += (getattr(row, f'lemma_{k}') == lemma)
            correct_at_k[k]['oracle_label_metric'] += min(label_freqs[label], k)
            correct_at_k[k]['truncated_label_metric'] += min(label_freqs[label], k)
            correct_at_k[k]['truncated_lemma_metric'] += min(lemma_freqs[lemma], k)

            correct_at_k[k]['total'] += k
            correct_at_k[k]['label_total'] += (label_freqs[label])
            correct_at_k[k]['lemma_total'] += (lemma_freqs[lemma])

    if no_data:
        print("No instances in this bin, skipping")
        return None, None


    ps_at_k = defaultdict(lambda: dict())
    for k in range(1, top_n+1):
        for l in ['label', 'lemma']:
            correct_at_k[k][l] += correct_at_k[k-1][l]
            ps_at_k[k][l] = correct_at_k[k][l] / correct_at_k[k]['total']

    recalls_at_k = defaultdict(lambda: dict())
    for k in range(1, top_n+1):
        for l in ['label', 'lemma']:
            recalls_at_k[k][l] = correct_at_k[k][l] / correct_at_k[k][f'{l}_total']

    truncated_recalls_at_k = defaultdict(lambda: dict())
    for k in range(1, top_n+1):
        for l in ['label', 'lemma']:
            truncated_recalls_at_k[k][l] = correct_at_k[k][l] / correct_at_k[k][f'truncated_{l}_metric']

    oracle_recalls_at_k = defaultdict(lambda: dict())
    for k in range(1, top_n + 1):
        oracle_recalls_at_k[k]['label'] = correct_at_k[k][f'oracle_label_metric'] / correct_at_k[k]['label_total']

    # write to pickles
    ps_at_k = dict(ps_at_k)
    for key, value in ps_at_k.items():
        ps_at_k[key] = dict(value)
    pickle_write(ps_at_k, pickle_path + '.prec')

    recalls_at_k = dict(recalls_at_k)
    for key, value in recalls_at_k.items():
        recalls_at_k[key] = dict(value)
    pickle_write(recalls_at_k, pickle_path + '.rec')

    truncated_recalls_at_k = dict(truncated_recalls_at_k)
    for key, value in truncated_recalls_at_k.items():
        truncated_recalls_at_k[key] = dict(value)
    pickle_write(truncated_recalls_at_k, pickle_path + '.trec')

    oracle_recalls_at_k = dict(oracle_recalls_at_k)
    for key, value in oracle_recalls_at_k.items():
        oracle_recalls_at_k[key] = dict(value)
    pickle_write(oracle_recalls_at_k, pickle_path + '.orec')
    return recalls_at_k, recalls_at_k


def main(embedding_name, distance_metric, top_n=50):
    predict(embedding_name, distance_metric, top_n)

    with open('cache/ontonotes_stats/train_label_freq.tsv', 'r') as f:
        label_freqs = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}
    with open('cache/ontonotes_stats/train_lemma_freq.tsv', 'r') as f:
        lemma_freqs = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}

    df = pd.read_csv(f'cache/ontonotes_{distance_metric}_predictions/{embedding_name}.tsv', sep='\t')
    for min_train_freq, max_train_freq in [[5,25], [25,100], [100,200], [200, 500000]]:
        for min_rarity, max_rarity in [[0, 0.01], [0.01,0.05], [0.05,0.15], [0.15,0.25], [0.25,1]]:
            print(f"Cutoff: [{min_train_freq},{max_train_freq}), Rarity: [{min_rarity},{max_rarity})")
            pr_at_k(df, label_freqs, lemma_freqs, top_n,
                    pickle_path=f'cache/ontonotes_{distance_metric}_predictions/{embedding_name}-{min_train_freq}to{max_train_freq}-{min_rarity}to{max_rarity}.pkl',
                    min_train_freq=min_train_freq,
                    max_train_freq=max_train_freq,
                    min_rarity=min_rarity,
                    max_rarity=max_rarity)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding",
        help="The pretrained embeddings to use.",
        choices=[
            'bert-large-cased',
            'bert-base-cased',
            'bert-base-uncased',
        ],
        default='bert-base-cased'
    )
    ap.add_argument(
        "--metric",
        help="How to measure distance between two BERT embedding vectors",
        choices=[
            'euclidean',
            'cosine'
        ],
        default='cosine'
    )
    ap.add_argument(
        '--top-n',
        type=int,
        default=50
    )
    args = ap.parse_args()
    print(args)

    main(
        args.embedding,
        args.metric,
        top_n=args.top_n
    )
