
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


def read_datasets(embedding_name, train_filepath, test_filepath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_embedding_predictor = predictor_for_train_reader(embedding_name, device)
    indexer = indexer_for_embedder(embedding_name)

    print("Loading data")
    # note: filenames passed to read() are actually dummies (data is read from nltk's copy of semcor)
    # but we need them there for caching to work
    train_reader = SemcorReader(
        split='train',
        token_indexers={'tokens': indexer},
        embedding_predictor=train_embedding_predictor,
        cache_directory=f'instance_cache/{embedding_name}'
    )
    test_reader = SemcorReader(split='test', token_indexers={'tokens': indexer}, cache_directory='instance_cache')
    print("Reading train")
    train_dataset = train_reader.read(train_filepath + '__' + embedding_name)
    print("Reading test")
    test_dataset = test_reader.read(test_filepath + '__' + embedding_name)
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
