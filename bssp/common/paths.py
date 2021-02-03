import os


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