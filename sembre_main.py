import argparse
import csv
import os
from collections import Counter

import tqdm
import torch
from allennlp.data import Vocabulary
from bssp.common.reading import read_dataset_cached, indexer_for_embedder, embedder_for_embedding
from bssp.semcor.model import SemcorRetriever, SemcorPredictor, format_sentence
from bssp.semcor.dataset_reader import SemcorReader


def read_datasets(embedding_name, train_filepath, test_filepath):
    train_dataset = read_dataset_cached(SemcorReader, 'semcor', 'train', embedding_name, with_embeddings=True)
    print(train_dataset[0])
    print(train_dataset[0]['span_embeddings'].array)
    test_dataset = read_dataset_cached(SemcorReader, 'semcor', 'test', embedding_name, with_embeddings=False)
    return train_dataset, test_dataset


def dataset_stats(filepath, dataset):
    labels = Counter()
    lemmas = Counter()
    synsets = Counter()

    for instance in dataset:
        label = instance['lemma_label'].label
        i = label.rfind('_')
        synset, lemma = label[:i], label[i+1:]
        labels[label] += 1
        synsets[synset] += 1
        lemmas[lemma] += 1

    os.makedirs('cache/stats', exist_ok=True)
    path = f'cache/stats/{filepath}'
    with open(path + '_label_freq.tsv', 'w', encoding='utf-8') as f:
        for item, freq in sorted(labels.items(), key=lambda x:-x[1]):
            f.write(f"{item}\t{freq}\n")
    with open(path + '_synset_freq.tsv', 'w', encoding='utf-8') as f:
        for item, freq in sorted(synsets.items(), key=lambda x:-x[1]):
            f.write(f"{item}\t{freq}\n")
    with open(path + '_lemma_freq.tsv', 'w', encoding='utf-8') as f:
        for item, freq in sorted(lemmas.items(), key=lambda x:-x[1]):
            f.write(f"{item}\t{freq}\n")

    return labels, synsets, lemmas


def stats(train_filepath, train_dataset, test_filepath, test_dataset):
    train_labels, train_synsets, train_lemmas = dataset_stats(train_filepath, train_dataset)
    test_labels, test_synsets, test_lemmas = dataset_stats(test_filepath, test_dataset)

    return train_labels, train_synsets, train_lemmas


def main(embedding_name, distance_metric, train_filepath, test_filepath, top_n=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read train and test splits of semcor, precompute embeddings on train
    train_dataset, test_dataset = read_datasets(embedding_name, train_filepath, test_filepath)
    # record some stats on train's labels
    trlabc, trsymc, trlemc = stats(train_filepath, train_dataset, test_filepath, test_dataset)

    print("Constructing vocabulary")
    indexer = indexer_for_embedder(embedding_name)
    vocab, embedder = embedder_for_embedding(embedding_name)

    # we're using a `transformers` model
    label_vocab = Vocabulary.from_instances(train_dataset)
    label_vocab.extend_from_instances(test_dataset)
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
    model = SemcorRetriever(
        vocab=vocab,
        embedder=embedder,
        target_dataset=train_dataset,
        distance_metric=distance_metric,
        device=device,
        top_n=top_n,
    ).eval().to(device)
    dummy_reader = SemcorReader(split='train', token_indexers={'tokens': indexer})
    predictor = SemcorPredictor(model=model, dataset_reader=dummy_reader)

    os.makedirs(f'cache/{distance_metric}_predictions', exist_ok=True)
    predictions_path = f'cache/{distance_metric}_predictions/{embedding_name.replace("embeddings/", "")}.tsv'
    with open(predictions_path, 'wt') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        header = ['sentence', 'label', 'synset', 'lemma', 'label_freq_in_train']
        header += [f"label_{i+1}" for i in range(top_n)]
        header += [f"synset_{i+1}" for i in range(top_n)]
        header += [f"lemma_{i+1}" for i in range(top_n)]
        header += [f"sentence_{i+1}" for i in range(top_n)]
        header += [f"distance_{i+1}" for i in range(top_n)]
        tsv_writer.writerow(header)

        for instance in tqdm.tqdm([i for i in test_dataset if trlabc[i['lemma_label'].label] >= 5]):
            d = predictor.predict_instance(instance)
            sentence = [t.text for t in instance['text'].tokens]
            span = instance['lemma_span']
            sentence = format_sentence(sentence, span.span_start, span.span_end)
            label = instance['lemma_label'].label
            synset = label[:label.rfind('_')]
            lemma = label[label.rfind('_')+1:]
            label_freq_in_train = trlabc[instance['lemma_label'].label]
            row = [sentence, label, synset, lemma, label_freq_in_train]
            row += [i['label'] for i in d[f'top_{top_n}']]
            row += [i['label'][:i['label'].rfind('_')] for i in d[f'top_{top_n}']]
            row += [i['label'][i['label'].rfind('_')+1:] for i in d[f'top_{top_n}']]
            row += [i['sentence'] for i in d[f'top_{top_n}']]
            row += [i['distance'] for i in d[f'top_{top_n}']]
            tsv_writer.writerow(row)

    print(f"Wrote predictions to {predictions_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "embedding_name",
        help="The pretrained embeddings to use.",
        choices=[
            'bert-large-cased',
            'bert-base-cased',
            'bert-base-uncased',
            'embeddings/glove.6B.300d.txt',
            # more, but above have bene tested
        ]
    )
    ap.add_argument(
        "distance_metric",
        help="How to measure distance between two BERT embedding vectors",
        choices=[
            'euclidean',
            'cosine'
        ]
    )
    ap.add_argument(
        '--small',
        action='store_true',
        help='when provided, will use a tiny slice of semcor (use for debugging)'
    )
    ap.add_argument(
        '--top-n',
        type=int,
        default=50
    )
    args = ap.parse_args()
    print(args)

    main(
        args.embedding_name,
        args.distance_metric,
        'train' + ('_small' if args.small else ''),
        'test' + ('_small' if args.small else ''),
        top_n=args.top_n
    )
