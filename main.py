import argparse
import csv
import os
import pickle
from collections import Counter

import tqdm

from sembre import SemcorReader, SemcorRetriever, SemcorPredictor, format_sentence
from sembre.embedder_model import EmbedderModelPredictor, EmbedderModel, EmbedderDatasetReader
import torch
from transformers import BertTokenizer
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder


# swappable elements --------------------------------------------------------------------------------
def indexer_for_embedder(embedding_name):
    if embedding_name.startswith('bert-'):
        return PretrainedTransformerMismatchedIndexer(embedding_name, namespace="tokens")
    else:
        return SingleIdTokenIndexer(namespace="tokens")


def embedder_for_embedding(embedding_name):
    vocab = Vocabulary()
    if embedding_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(embedding_name)
        for word in tokenizer.vocab.keys():
            vocab.add_token_to_namespace(word, "tokens")
        token_embedders = {"tokens": PretrainedTransformerMismatchedEmbedder(model_name=embedding_name)}
    else:
        with open(embedding_name, 'r', encoding='utf-8') as f:
            count = 0
            for i, line in enumerate(f.readlines()):
                vocab.add_token_to_namespace(line[0:line.find(' ')], namespace="tokens")
                count += 1
        token_embedders = {
            "tokens": Embedding(
                embedding_dim=300,
                vocab=vocab,
                pretrained_file=embedding_name,
                trainable=False
            )
        }

    return vocab, BasicTextFieldEmbedder(token_embedders)


def predictor_for_train_reader(embedding_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indexer = indexer_for_embedder(embedding_name)
    vocab, embedder = embedder_for_embedding(embedding_name)
    model = EmbedderModel(vocab=vocab, embedder=embedder).to(device).eval()
    predictor = EmbedderModelPredictor(
        model=model,
        dataset_reader=EmbedderDatasetReader(token_indexers={'tokens': indexer})
    )
    return predictor


# reading --------------------------------------------------------------------------------
def read_dataset_cached(split, filepath, embedding_name, with_embeddings=False):
    if with_embeddings:
        embedding_predictor = predictor_for_train_reader(embedding_name)
    else:
        embedding_predictor = None

    indexer = indexer_for_embedder(embedding_name)
    reader = SemcorReader(
        split=split,
        token_indexers={'tokens': indexer},
        embedding_predictor=embedding_predictor
    )

    pickle_name = filepath + ('__' + embedding_name).replace('embeddings/', '')
    pickle_path = 'dataset_cache/' + pickle_name + '.pkl'
    if os.path.isfile(pickle_path):
        print(f"Reading split {split} from cache at {pickle_path}")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    print(f"Reading split {split} from nltk")
    dataset = reader.read(pickle_name)
    os.makedirs('dataset_cache', exist_ok=True)
    with open(pickle_path, 'wb') as f:
        print(f"Caching {split} in {pickle_path}")
        pickle.dump(dataset, f)

    return dataset


def read_datasets(embedding_name, train_filepath, test_filepath):
    train_dataset = read_dataset_cached('train', train_filepath, embedding_name, with_embeddings=True)
    print(train_dataset[0])
    print(train_dataset[0]['span_embeddings'].array)
    test_dataset = read_dataset_cached('test', test_filepath, embedding_name, with_embeddings=False)
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

    os.makedirs('stats', exist_ok=True)
    path = f'stats/{filepath}'
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


def main(embedding_name, train_filepath, test_filepath):
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
        device=device
    ).eval().to(device)
    dummy_reader = SemcorReader(split='train', token_indexers={'tokens': indexer})
    predictor = SemcorPredictor(model=model, dataset_reader=dummy_reader)

    os.makedirs('predictions', exist_ok=True)
    predictions_path = f'predictions/{embedding_name.replace("embeddings/", "")}.tsv'
    with open(predictions_path, 'wt') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        header = ['sentence', 'label', 'synset', 'lemma', 'label_freq_in_train']
        header += [f"label_{i+1}" for i in range(50)]
        header += [f"synset_{i+1}" for i in range(50)]
        header += [f"lemma_{i+1}" for i in range(50)]
        header += [f"sentence_{i+1}" for i in range(50)]
        header += [f"cosine_sim_{i+1}" for i in range(50)]
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
            row += [i['label'] for i in d['top_50']]
            row += [i['label'][:i['label'].rfind('_')] for i in d['top_50']]
            row += [i['label'][i['label'].rfind('_')+1:] for i in d['top_50']]
            row += [i['sentence'] for i in d['top_50']]
            row += [i['cosine_sim'] for i in d['top_50']]
            tsv_writer.writerow(row)

    print(f"Wrote predictions to {predictions_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "embedding_name",
        help="The pretrained embeddings to use.",
        choices=[
            'bert-base-cased',
            'bert-base-uncased',
            'embeddings/glove.6B.300d.txt',
            # more, but above have bene tested
        ]
    )
    ap.add_argument(
        '--small',
        action='store_true',
        help='when provided, will use a tiny slice of semcor (use for debugging)'
    )
    args = ap.parse_args()
    print(args)

    main(
        args.embedding_name,
        'train' + ('_small' if args.small else ''),
        'test' + ('_small' if args.small else ''),
    )
