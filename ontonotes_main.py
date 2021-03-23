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
from torch.autograd import profiler

from bssp.common import paths
from bssp.common.analysis import metrics_at_k, dataset_stats
from bssp.common.const import TRAIN_FREQ_BUCKETS, PREVALENCE_BUCKETS
from bssp.common.reading import read_dataset_cached, make_indexer, make_embedder, activate_bert_layers
from bssp.common.nearest_neighbor_models import NearestNeighborRetriever, NearestNeighborPredictor, RandomRetriever
from bssp.ontonotes.dataset_reader import OntonotesReader, lemma_from_label

TRAIN_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/train'
DEVELOPMENT_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/development'
TEST_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/test'



def read_datasets(embedding_name, bert_layers):
    train_dataset = read_dataset_cached(
        OntonotesReader, TRAIN_FILEPATH, 'ontonotes', 'train', embedding_name, bert_layers=bert_layers, with_embeddings=True
    )
    print(train_dataset[0])
    dev_dataset = read_dataset_cached(
        OntonotesReader, DEVELOPMENT_FILEPATH, 'ontonotes', 'dev', embedding_name, bert_layers=bert_layers, with_embeddings=False
    )
    test_dataset = read_dataset_cached(
        OntonotesReader, TEST_FILEPATH, 'ontonotes', 'test', embedding_name, bert_layers=bert_layers, with_embeddings=False
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


SENTENCE_CACHE = {}


def format_sentence(sentence, i, j):
    key = (tuple(sentence), i, j)
    if key in SENTENCE_CACHE:
        return SENTENCE_CACHE[key]
    formatted = " ".join(sentence[:i] + ['>>' + sentence[i] + '<<'] + sentence[j+1:])
    SENTENCE_CACHE[key] = formatted
    return formatted


def predict(embedding_name, distance_metric, top_n, query_n, bert_layers):
    predictions_path = paths.predictions_tsv_path(distance_metric, embedding_name, query_n, bert_layers=bert_layers)
    if os.path.isfile(predictions_path):
        print(f"Reading predictions from {predictions_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, testdev_dataset = read_datasets(embedding_name, bert_layers)
    train_label_counts, train_lemma_counts = stats(train_dataset, testdev_dataset)

    indexer = make_indexer(embedding_name)
    vocab, embedder = make_embedder(embedding_name, bert_layers)

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
    with open(predictions_path, 'wt') as f, torch.no_grad():
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

            labels = []
            lemmas = []
            sentences = []
            distances = []

            for result in results:
                if result is None:
                    distances.append(88888888)
                    labels.append("")
                    lemmas.append("")
                    sentences.append("")
                else:
                    index, distance = result
                    distances.append(distance)
                    instance = train_dataset[index]
                    labels.append(instance['label'].label)
                    lemmas.append(lemma_from_label(labels[-1]))
                    span = instance['label_span']
                    sentences.append(
                        format_sentence([t.text for t in instance['text'].tokens],
                                        span.span_start, span.span_end)
                    )

            row += labels
            row += lemmas
            row += sentences
            row += distances
            if len(row) != 204:
                print(len(row))
                assert False
            tsv_writer.writerow(row)

    print(f"Wrote predictions to {predictions_path}")


def main(embedding_name, distance_metric, pos='all', query_category='all', top_n=50, query_n=1, bert_layers=None):
    predict(embedding_name, distance_metric, top_n, query_n, bert_layers)

    with open('cache/ontonotes_stats/train_label_freq.tsv', 'r') as f:
        label_freqs = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}
    with open('cache/ontonotes_stats/train_lemma_freq.tsv', 'r') as f:
        lemma_freqs = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}

    df = pd.read_csv(
        paths.predictions_tsv_path(distance_metric, embedding_name, query_n, bert_layers=bert_layers),
        sep='\t',
        error_bad_lines=False
    )
    for min_train_freq, max_train_freq in TRAIN_FREQ_BUCKETS:
        for min_rarity, max_rarity in PREVALENCE_BUCKETS:
            print(f"Cutoff: [{min_train_freq},{max_train_freq}), Rarity: [{min_rarity},{max_rarity})")
            metrics_at_k(
                df, label_freqs, lemma_freqs, top_n, query_category, pos,
                path_f=lambda ext: paths.bucketed_metric_at_k_path(
                    distance_metric,
                    query_n,
                    embedding_name,
                    min_train_freq,
                    max_train_freq,
                    min_rarity,
                    max_rarity,
                    ext,
                    query_category,
                    pos,
                    bert_layers=bert_layers
                ),
                min_train_freq=min_train_freq,
                max_train_freq=max_train_freq,
                min_rarity=min_rarity,
                max_rarity=max_rarity,
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
    ap.add_argument(
        '--bert-layers',
        help="Comma-delimited list of bert layers (0-indexed) to average. "
             "For bert-base models, these range between 0 and 11.",
        default=None
    )
    ap.add_argument(
        '--query-category',
        choices=['non-nota', 'nota', 'all'],
        help="Some word senses are negatively defined 'none of the above' senses. These are presumably "
             "less coherent than normal word senses which are positively defined. `non-nota` will use "
             "only the latter senses in queries, `nota` will use only the former senses in queries, and "
             "`all` will use both kinds of senses in queries. In any case, possible results will include "
             "both NOTA and non-NOTA queries.",
        default='all'
    )
    ap.add_argument(
        '--pos',
        choices=['n','v','all'],
        help="Whether to query on nouns, verbs, or both",
        default='all'
    )
    args = ap.parse_args()
    bert_layers = None
    if args.bert_layers:
        bert_layers = [int(i) for i in args.bert_layers.split(",")]
    else:
        if 'bert' in args.embedding and args.metric != 'baseline':
            raise Exception('Specify --bert-layers (e.g., `--bert-layers 1,2,3`)')

    main(
        args.embedding,
        args.metric,
        query_category=args.query_category,
        pos=args.pos,
        top_n=args.top_n,
        query_n=args.query_n,
        bert_layers=bert_layers
    )
