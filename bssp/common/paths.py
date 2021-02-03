import os


def dataset_path(corpus_name, embedding_name, split):
    os.makedirs('cache/dataset/', exist_ok=True)
    pickle_name = corpus_name + "_" + split + ('__' + embedding_name).replace('cache/embeddings/', '')
    pickle_path = 'cache/dataset/' + pickle_name + '.pkl'
    return pickle_path


def freq_tsv_path(directory, split, token_type):
    os.makedirs(f'cache/{directory}', exist_ok=True)
    return f'cache/{directory}/{split}_{token_type}_freq.tsv'


def model_dir(distance_metric, query_n):
    return f'cache/ontonotes_{distance_metric}_q{query_n}_predictions/'


def predictions_tsv_path(distance_metric, embedding_name, query_n):
    mdir = model_dir(distance_metric, query_n)
    os.makedirs(mdir, exist_ok=True)
    return mdir + f'{embedding_name.replace("embeddings/", "")}.tsv'


def bucketed_metric_at_k_path(
        distance_metric,
        query_n,
        embedding_name,
        min_train_freq,
        max_train_freq,
        min_rarity,
        max_rarity,
        ext
):
    mdir = model_dir(distance_metric, query_n)
    return mdir + f'{embedding_name}_{min_train_freq}-{max_train_freq}_{min_rarity}-{max_rarity}.{ext}'