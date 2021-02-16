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
from random import shuffle

import torch
import pandas as pd
import tqdm
from allennlp.data import Vocabulary
from ldg.pickle import pickle_write

from bssp.common import paths
from bssp.common.analysis import metrics_at_k, dataset_stats
from bssp.common.const import TRAIN_FREQ_BUCKETS, PREVALENCE_BUCKETS
from bssp.common.reading import read_dataset_cached, indexer_for_embedder, embedder_for_embedding
from bssp.common.nearest_neighbor_models import NearestNeighborRetriever, NearestNeighborPredictor, format_sentence, \
    RandomRetriever
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


def batch_queries(instances, query_n, full_batches_only=True):
    instances_by_label = defaultdict(list)
    for instance in instances:
        label = instance['label'].label
        instances_by_label[label].append(instance)

    batches = []
    for label, label_instances in instances_by_label.items():
        shuffle(label_instances)
        i = 0
        while i < len(label_instances):
            if full_batches_only and i+query_n > len(label_instances):
                break
            batches.append(label_instances[i:i+query_n])
            i += query_n

    return batches


def predict(embedding_name, distance_metric, top_n, query_n):
    predictions_path = paths.predictions_tsv_path(distance_metric, embedding_name, query_n)
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
    if distance_metric == 'baseline':
        model = RandomRetriever(
            vocab=vocab,
            target_dataset=train_dataset,
            device=device,
            top_n=top_n,
            same_lemma=True,
        ).eval().to(device)
    else:
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

    # remove any super-rare instances that did not occur at least 5 times in train
    # (these are not interesting to eval on)
    instances = [i for i in testdev_dataset if train_label_counts[i['label'].label] >= 5]
    # We are abusing the batch abstraction here--really a batch should be a set of independent instances,
    # but we are using it here as a convenient way to feed in a single instance.
    batches = batch_queries(instances, query_n)

    with open(predictions_path, 'wt') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        header = ['sentence', 'label', 'lemma', 'label_freq_in_train']
        header += [f"label_{i+1}" for i in range(top_n)]
        header += [f"lemma_{i+1}" for i in range(top_n)]
        header += [f"sentence_{i+1}" for i in range(top_n)]
        header += [f"distance_{i+1}" for i in range(top_n)]
        tsv_writer.writerow(header)

        for batch in tqdm.tqdm(batches):
            # the batch results are actually all the same--just take the first one
            ds = predictor.predict_batch_instance(batch)
            d = ds[0]
            sentences = [[t.text for t in i['text'].tokens] for i in batch]
            spans = [i['label_span'] for i in batch]
            sentences = [format_sentence(sentence, span.span_start, span.span_end)
                         for sentence, span in zip(sentences, spans)]
            label = batch[0]['label'].label
            lemma = lemma_from_label(label)
            label_freq_in_train = train_label_counts[label]

            row = [" || ".join(sentences), label, lemma, label_freq_in_train]
            results = d[f'top_{top_n}']
            results += [None for _ in range(top_n - len(results))]
            row += [i['label'] if i is not None else "" for i in results]
            row += [lemma_from_label(i['label']) if i is not None else "" for i in results]
            row += [i['sentence'] if i is not None else "" for i in results]
            row += [i['distance'] if i is not None else 888888 for i in results]
            if len(row) != 204:
                print(len(row))
                assert False
            tsv_writer.writerow(row)

    print(f"Wrote predictions to {predictions_path}")


def main(embedding_name, distance_metric, top_n=50, query_n=1):
    predict(embedding_name, distance_metric, top_n, query_n)

    with open('cache/ontonotes_stats/train_label_freq.tsv', 'r') as f:
        label_freqs = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}
    with open('cache/ontonotes_stats/train_lemma_freq.tsv', 'r') as f:
        lemma_freqs = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}

    df = pd.read_csv(paths.predictions_tsv_path(distance_metric, embedding_name, query_n), sep='\t')
    for min_train_freq, max_train_freq in TRAIN_FREQ_BUCKETS:
        for min_rarity, max_rarity in PREVALENCE_BUCKETS:
            print(f"Cutoff: [{min_train_freq},{max_train_freq}), Rarity: [{min_rarity},{max_rarity})")
            metrics_at_k(
                df, label_freqs, lemma_freqs, top_n,
                path_f=lambda ext: paths.bucketed_metric_at_k_path(
                    distance_metric,
                    query_n,
                    embedding_name,
                    min_train_freq,
                    max_train_freq,
                    min_rarity,
                    max_rarity,
                    ext=ext
                ),
                min_train_freq=min_train_freq,
                max_train_freq=max_train_freq,
                min_rarity=min_rarity,
                max_rarity=max_rarity
            )


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
            'cosine',
            'baseline'
        ],
        default='cosine'
    )
    ap.add_argument(
        '--top-n',
        type=int,
        default=50
    )
    ap.add_argument(
        '--query-n',
        type=int,
        default=1,
        help="Number of sentences to draw from when formulating a query. "
             "For n>1, embeddings of the target word will be average pooled."
    )
    args = ap.parse_args()
    print(args)

    main(
        args.embedding,
        args.metric,
        top_n=args.top_n,
        query_n=args.query_n
    )
