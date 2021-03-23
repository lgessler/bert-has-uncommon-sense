import os


def dataset_path(corpus_name, embedding_name, split, bert_layers=None):
    os.makedirs('cache/dataset/', exist_ok=True)
    pickle_name = corpus_name
    if bert_layers is not None:
        pickle_name += '_' + ",".join(str(l) for l in bert_layers)
    pickle_name += "_" + split + ('__' + embedding_name).replace('cache/embeddings/', '')
    pickle_path = 'cache/dataset/' + pickle_name + '.pkl'
    return pickle_path


def freq_tsv_path(directory, split, token_type):
    os.makedirs(f'cache/{directory}', exist_ok=True)
    return f'cache/{directory}/{split}_{token_type}_freq.tsv'


def freq_tsv_path2(corpus, distance_metric, query_n, split, token_type):
    directory = f'{corpus}_{distance_metric}_q{query_n}_predictions'
    return freq_tsv_path(directory, split, token_type)


def model_dir(corpus, distance_metric, query_n):
    return f'cache/{corpus}_{distance_metric}_q{query_n}_predictions/'


def predictions_tsv_path(corpus, distance_metric, embedding_name, query_n, bert_layers=None):
    mdir = model_dir(corpus, distance_metric, query_n)
    os.makedirs(mdir, exist_ok=True)
    return mdir + (f'{embedding_name.replace("embeddings/", "")}'
                   f'{("_" + ",".join(map(str, bert_layers))) if bert_layers else ""}'
                   f'.tsv')


def bucketed_metric_at_k_path(
        corpus,
        distance_metric,
        query_n,
        embedding_name,
        min_train_freq,
        max_train_freq,
        min_rarity,
        max_rarity,
        ext,
        query_category=None,
        pos=None,
        bert_layers=None
):
    mdir = model_dir(corpus, distance_metric, query_n)
    return mdir + (f'{embedding_name}'
                   f'{("_" + ",".join(map(str, bert_layers))) if bert_layers else ""}'
                   + (f'_{pos}' if pos else "")
                   + (f'_{query_category}' if query_category else "") +
                   f'_{min_train_freq}-{max_train_freq}'
                   f'_{min_rarity}-{max_rarity}'
                   f'.{ext}')