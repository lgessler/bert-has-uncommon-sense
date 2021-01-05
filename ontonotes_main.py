import argparse

from bssp.common.reading import read_dataset_cached
from bssp.ontonotes.dataset_reader import OntonotesReader
import torch

TRAIN_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/train'
DEVELOPMENT_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/development'
TEST_FILEPATH = 'data/conll-formatted-ontonotes-5.0/data/test'


def read_datasets(embedding_name):
    train_dataset = read_dataset_cached(
        OntonotesReader, TRAIN_FILEPATH, 'ontonotes', 'train', embedding_name, with_embeddings=True
    )
    print(train_dataset[0])
    print(train_dataset[0]['span_embeddings'].array)
    dev_dataset = read_dataset_cached(
        OntonotesReader, DEVELOPMENT_FILEPATH, 'ontonotes', 'dev', embedding_name, with_embeddings=False
    )
    test_dataset = read_dataset_cached(
        OntonotesReader, TEST_FILEPATH, 'ontonotes', 'test', embedding_name, with_embeddings=False
    )
    return train_dataset, dev_dataset + test_dataset


def main(embedding_name, distance_metric, top_n=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, testdev_dataset = read_datasets(embedding_name)
    print(testdev_dataset)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding",
        help="The pretrained embeddings to use.",
        choices=[
            'bert-large-cased',
            'bert-base-cased',
            'bert-base-uncased',
            'embeddings/glove.6B.300d.txt',
            # more, but above have bene tested
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
        '--small',
        action='store_true',
        help='when provided, will use a tiny slice of ontonotes (use for debugging)'
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
