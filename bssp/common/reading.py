"""
Convenience functions used in the setup of experiments. These are necessary because we're not using
allennlp's default config-based execution environment.
"""
import pickle
import os
import torch
from allennlp.data import Vocabulary
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer, SingleIdTokenIndexer
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers import Token
from transformers import BertTokenizer

from bssp.common import paths
from bssp.common.embedder_model import EmbedderModel, EmbedderDatasetReader, EmbedderModelPredictor


def indexer_for_embedder(embedding_name):
    """Get a token indexer that's appropriate for the embedding type"""
    if embedding_name.startswith('bert-'):
        return PretrainedTransformerMismatchedIndexer(embedding_name, namespace="tokens")
    else:
        return SingleIdTokenIndexer(namespace="tokens")


def embedder_for_embedding(embedding_name):
    """Given the name of an embedding, return its Vocabulary and a BasicTextFieldEmbedder on its tokens.
    (A BasicTextFieldEmbbeder can be called on a tensor with token indexes to produce embeddings.)"""
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
    """
    When we are reading data from a train split, we want to store the embedding of the target word
    with the instance. This method returns a predictor that will simply allow us to predict embeddings.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indexer = indexer_for_embedder(embedding_name)
    vocab, embedder = embedder_for_embedding(embedding_name)
    model = EmbedderModel(vocab=vocab, embedder=embedder).to(device).eval()
    predictor = EmbedderModelPredictor(
        model=model,
        dataset_reader=EmbedderDatasetReader(token_indexers={'tokens': indexer})
    )
    return predictor


def read_dataset_cached(reader_cls, data_path, corpus_name, split, embedding_name, with_embeddings=False):
    if with_embeddings:
        embedding_predictor = predictor_for_train_reader(embedding_name)
    else:
        embedding_predictor = None

    indexer = indexer_for_embedder(embedding_name)
    reader = reader_cls(
        split=split,
        token_indexers={'tokens': indexer},
        embedding_predictor=embedding_predictor
    )

    pickle_path = paths.dataset_path(corpus_name, embedding_name, split)
    if os.path.isfile(pickle_path):
        print(f"Reading split {split} from cache at {pickle_path}")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    print(f"Reading split {split}")
    dataset = list(reader.read(data_path))
    with open(pickle_path, 'wb') as f:
        print(f"Caching {split} in {pickle_path}")
        pickle.dump(dataset, f)

    return dataset
