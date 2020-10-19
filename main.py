import os
import pickle

from sembre import SemcorReader, SemcorRetriever, SemcorPredictor
from sembre.embedder_model import EmbedderModelPredictor, EmbedderModel, EmbedderDatasetReader
import torch
from allennlp.data import Vocabulary, DatasetReader
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer, TokenIndexer
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder, Embedding
from allennlp.data.tokenizers import PretrainedTransformerTokenizer



#predictor.predict(
#    sentence='Basil is easy to grow , and transformers ordinary meals into culinary treasure !'.split(),
#    lemma_span_start=0,
#    lemma_span_end=1,
#    lemma_label='basil.n.01'
#)


def indexer_for_embedder(embedding_name):
    if embedding_name.startswith('bert-'):
        return PretrainedTransformerMismatchedIndexer(embedding_name, namespace="tokens")
    else:
        raise Exception('bad embedding_name: ' + embedding_name)


def embedder_for_embedding(embedding_name):
    if embedding_name.startswith('bert-'):
        return PretrainedTransformerMismatchedEmbedder(model_name=embedding_name)
    else:
        raise Exception('bad embedding_name: ' + embedding_name)


def predictor_for_train_reader(embedding_name, device):
    indexer = indexer_for_embedder(embedding_name)
    embedder = embedder_for_embedding(embedding_name)
    # TODO: should really be using the embedding vocabulary, but i'm not sure how to get that here
    model = EmbedderModel(vocab=Vocabulary(), embedder=embedder).to(device).eval()
    predictor = EmbedderModelPredictor(
        model=model,
        dataset_reader=EmbedderDatasetReader(token_indexers={'tokens': indexer})
    )
    return predictor


def read_dataset_cached(split, filepath, embedding_name, with_embeddings=False):
    if with_embeddings:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        embedding_predictor = predictor_for_train_reader(embedding_name, device)
    else:
        embedding_predictor = None

    indexer = indexer_for_embedder(embedding_name)
    reader = SemcorReader(
        split=split,
        token_indexers={'tokens': indexer},
        embedding_predictor=embedding_predictor
    )

    pickle_name = filepath + '__' + embedding_name
    pickle_path = 'dataset_cache/' + pickle_name + '.pkl'
    if os.path.isfile(pickle_path):
        print(f"Reading split {split} from cache")
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
    test_dataset = read_dataset_cached('test', test_filepath, embedding_name, with_embeddings=False)
    return train_dataset, test_dataset


def main(embedding_name, train_filepath, test_filepath):
    train_dataset, test_dataset = read_datasets(embedding_name, train_filepath, test_filepath)

    print("Constructing vocabulary")
    vocab = Vocabulary.from_instances(train_dataset)
    vocab.extend_from_instances(test_dataset)

    print("Constructing model")
    indexer = indexer_for_embedder(embedding_name)
    embedder = embedder_for_embedding(embedding_name)
    model = SemcorRetriever(vocab=vocab, embedder=embedder, target_dataset=train_dataset)
    dummy_reader = SemcorReader(split='train', token_indexers={'tokens': indexer})
    predictor = SemcorPredictor(model=model, dataset_reader=dummy_reader)

    predictor.predict(
        sentence='UN inspectors who had been dispatched two weeks before claimed that there had been " no irregularities " .'.split(),
        lemma_span_start=16,
        lemma_span_end=17,
        lemma_label='abnormality.n.04_irregularity'
    )

if __name__ == '__main__':
    embedding_name = "bert-base-cased"
    #main(embedding_name, 'train_small', 'test_small')
    main(embedding_name, 'train', 'test')
