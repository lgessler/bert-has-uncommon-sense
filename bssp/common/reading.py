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
from transformers import BertTokenizer

from bssp.common import paths
from bssp.common.embedder_model import EmbedderModel, EmbedderDatasetReader, EmbedderModelPredictor


def make_indexer(embedding_name):
    """Get a token indexer that's appropriate for the embedding type"""
    if embedding_name.startswith('bert-'):
        return PretrainedTransformerMismatchedIndexer(embedding_name, namespace="tokens")
    else:
        return SingleIdTokenIndexer(namespace="tokens")


def activate_bert_layers(embedder, bert_layers):
    """
    The Embedder has params deep inside that produce a scalar mix of BERT layers via a softmax
    followed by a dot product. Activate the ones specified in `layers` and deactivate the rest
    """
    # whew!
    scalar_mix = embedder.token_embedder_tokens._matched_embedder._scalar_mix.scalar_parameters

    with torch.no_grad():
        for i, param in enumerate(scalar_mix):
            param.requires_grad = False
            # These parameters will be softmaxed, so get the layer(s) we want close to +inf,
            # and the layers we don't want close to -inf
            param.fill_(1e9 if i in bert_layers else -1e9)


def make_embedder(embedding_name, bert_layers=None):
    """Given the name of an embedding, return its Vocabulary and a BasicTextFieldEmbedder on its tokens.
    (A BasicTextFieldEmbbeder can be called on a tensor with token indexes to produce embeddings.)"""
    vocab = Vocabulary()
    if embedding_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(embedding_name)
        for word in tokenizer.vocab.keys():
            vocab.add_token_to_namespace(word, "tokens")
        token_embedders = {"tokens": PretrainedTransformerMismatchedEmbedder(model_name=embedding_name,
                                                                             last_layer_only=False)}
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

    embedder = BasicTextFieldEmbedder(token_embedders)
    if embedding_name.startswith('bert-'):
        activate_bert_layers(embedder, bert_layers)

    return vocab, embedder


def make_predictor_for_train_reader(embedding_name, bert_layers=None):
    """
    When we are reading data from a train split, we want to store the embedding of the target word
    with the instance. This method returns a predictor that will simply allow us to predict embeddings.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indexer = make_indexer(embedding_name)
    vocab, embedder = make_embedder(embedding_name, bert_layers=bert_layers)
    if bert_layers is not None:
        activate_bert_layers(embedder, bert_layers)
    model = EmbedderModel(vocab=vocab, embedder=embedder).to(device).eval()
    predictor = EmbedderModelPredictor(
        model=model,
        dataset_reader=EmbedderDatasetReader(token_indexers={'tokens': indexer})
    )
    return predictor


def read_dataset_cached(reader_cls, data_path, corpus_name, split, embedding_name, bert_layers=None, with_embeddings=False):
    if with_embeddings:
        embedding_predictor = make_predictor_for_train_reader(embedding_name, bert_layers=bert_layers)
    else:
        embedding_predictor = None

    indexer = make_indexer(embedding_name)
    reader = reader_cls(
        split=split,
        token_indexers={'tokens': indexer},
        embedding_predictor=embedding_predictor
    )

    pickle_path = paths.dataset_path(corpus_name, embedding_name, split, bert_layers=bert_layers)
    if os.path.isfile(pickle_path):
        print(f"Reading split {split} from cache at {pickle_path}")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)

    print(f"Reading split {split}")
    dataset = sorted(list(reader.read(data_path)), key=lambda x: x['label'].label)
    with open(pickle_path, 'wb') as f:
        print(f"Caching {split} in {pickle_path}")
        pickle.dump(dataset, f)

    return dataset
