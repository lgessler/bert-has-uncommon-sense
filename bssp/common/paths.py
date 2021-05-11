import os


def dataset_path(cfg, split):
    os.makedirs("cache/dataset/", exist_ok=True)
    pickle_name = cfg.corpus_name
    if cfg.bert_layers is not None:
        pickle_name += "_" + ",".join(str(l) for l in cfg.bert_layers)
    pickle_name += "_" + split + ("__" + cfg.embedding_model).replace("cache/embeddings/", "")
    override_weights_piece = (
        "_" + cfg.override_weights_path.replace(os.sep, "__") if cfg.override_weights_path is not None else ""
    )
    pickle_name += override_weights_piece
    pickle_path = "cache/dataset/" + pickle_name + ".pkl"
    return pickle_path


def freq_tsv_path(directory, split, token_type):
    os.makedirs(f"cache/{directory}", exist_ok=True)
    return f"cache/{directory}/{split}_{token_type}_freq.tsv"


def freq_tsv_path2(cfg, split, token_type):
    directory = f"{cfg.corpus_name}_{cfg.metric}_q{cfg.query_n}_predictions"
    return freq_tsv_path(directory, split, token_type)


def model_dir(cfg):
    return f"cache/{cfg.corpus_name}_{cfg.metric}_q{cfg.query_n}_predictions/"


def predictions_tsv_path(cfg):
    mdir = model_dir(cfg)
    os.makedirs(mdir, exist_ok=True)
    override_weights_piece = (
        "_" + cfg.override_weights_path.replace(os.sep, "__") if cfg.override_weights_path is not None else ""
    )
    return mdir + (
        f'{cfg.embedding_model.replace("embeddings/", "")}'
        f"{override_weights_piece}"
        f'{("_" + ",".join(map(str, cfg.bert_layers))) if cfg.bert_layers else ""}'
        f".tsv"
    )


def bucketed_metric_at_k_path(
    cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, ext, query_category=None, pos=None,
):
    mdir = model_dir(cfg)
    override_weights_piece = (
        "_" + cfg.override_weights_path.replace(os.sep, "__") if cfg.override_weights_path is not None else ""
    )
    return mdir + (
        f"{cfg.embedding_model}"
        f"{override_weights_piece}"
        f'{("_" + ",".join(map(str, cfg.bert_layers))) if cfg.bert_layers else ""}'
        + (f"_{pos}" if pos else "")
        + (f"_{query_category}" if query_category else "")
        + f"_{min_train_freq}-{max_train_freq}"
        f"_{min_rarity}-{max_rarity}"
        f".{ext}"
    )
