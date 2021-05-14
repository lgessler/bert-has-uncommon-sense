"""
This file will:
- Read the train, test, dev splits of clres (see TRAIN_FILEPATH, etc. below) into allennlp datasets
- Get BERT embeddings for words with word senses annotations in the train split
- Use a model to predict embeddings from the dev+test splits and use a similarity function to find similar instances in train
- Write the top 50 results for each instance in dev+test to a tsv with information
"""

import csv
import os
from copy import copy

import click
import torch
import pandas as pd
import tqdm
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.training import GradientDescentTrainer
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer

import bssp
from bssp.common import paths
from bssp.common.analysis import metrics_at_k, dataset_stats
from bssp.common.config import Config
from bssp.common.pickle import pickle_read
from bssp.common.reading import read_dataset_cached, make_indexer, make_embedder
from bssp.common.nearest_neighbor_models import NearestNeighborRetriever, NearestNeighborPredictor, RandomRetriever
from bssp.common.util import batch_queries, format_sentence
from bssp.clres.dataset_reader import lemma_from_label, ClresConlluReader
from bssp.fews.dataset_reader import FewsReader
from bssp.fine_tuning.models import StreusleFineTuningModel
from bssp.fine_tuning.streusle import StreusleJsonReader
from bssp.ontonotes.dataset_reader import OntonotesReader
from bssp.semcor.dataset_reader import SemcorReader


@click.group()
def cli():
    pass


@cli.command(help="run a trial on a corpus")
@click.argument("corpus_slug")
@click.option("--embedding-model", help="`transformers` model slug to use", default="bert-base-cased")
@click.option(
    "--metric",
    help="how to measure embedding distance",
    default="cosine",
    type=click.Choice(["euclidean", "cosine", "baseline"], case_sensitive=False),
)
@click.option("--override-weights", help="Path to override weights from fine-tuning to use with the model")
@click.option("--top-n", type=int, default=50)
@click.option(
    "--query-n",
    type=int,
    default=1,
    help="Number of sentences to draw from when formulating a query. "
    "For n>1, embeddings of the target word will be average pooled.",
)
@click.option("--bert-layer", type=int, help="BERT layer (0-indexed) to average.", default=7)
def trial(corpus_slug, embedding_model, metric, override_weights, top_n, query_n, bert_layer):
    cfg = Config(
        corpus_slug,
        embedding_model=embedding_model,
        override_weights_path=override_weights,
        metric=metric,
        top_n=top_n,
        query_n=query_n,
        bert_layers=[bert_layer],
    )
    predict(cfg)
    label_freqs, lemma_freqs = read_stats(cfg)
    df = pd.read_csv(paths.predictions_tsv_path(cfg), sep="\t", error_bad_lines=False)
    lemma_f = get_lemma_f(cfg)
    for min_train_freq, max_train_freq in cfg.train_freq_buckets:
        for min_rarity, max_rarity in cfg.prevalence_buckets:
            print(f"Cutoff: [{min_train_freq},{max_train_freq}), Rarity: [{min_rarity},{max_rarity})")
            metrics_at_k(
                cfg,
                df,
                label_freqs,
                lemma_freqs,
                lemma_f,
                min_train_freq=min_train_freq,
                max_train_freq=max_train_freq,
                min_rarity=min_rarity,
                max_rarity=max_rarity,
            )


def read_datasets(cfg):
    if cfg.corpus_name == "clres":
        train_filepath = "data/pdep/pdep_train.conllu"
        test_filepath = "data/pdep/pdep_test.conllu"
        train_dataset = read_dataset_cached(cfg, ClresConlluReader, "train", train_filepath, with_embeddings=True)
        test_dataset = read_dataset_cached(cfg, ClresConlluReader, "test", test_filepath, with_embeddings=False)
    elif cfg.corpus_name == "ontonotes":
        train_filepath = "data/conll-formatted-ontonotes-5.0/data/train"
        dev_filepath = "data/conll-formatted-ontonotes-5.0/data/development"
        test_filepath = "data/conll-formatted-ontonotes-5.0/data/test"
        train_dataset = read_dataset_cached(cfg, OntonotesReader, "train", train_filepath, with_embeddings=True)
        dev_dataset = read_dataset_cached(cfg, OntonotesReader, "dev", dev_filepath, with_embeddings=False)
        test_dataset = read_dataset_cached(cfg, OntonotesReader, "test", test_filepath, with_embeddings=False)
        test_dataset = dev_dataset + test_dataset
    elif cfg.corpus_name == "semcor":
        train_dataset = read_dataset_cached(cfg, SemcorReader, "train", None, with_embeddings=True)
        test_dataset = read_dataset_cached(cfg, SemcorReader, "test", None, with_embeddings=False)
    elif cfg.corpus_name == "fews":
        train_dataset = read_dataset_cached(cfg, FewsReader, "train", "data/fews/train/train.txt", with_embeddings=True)
        dev_dataset = read_dataset_cached(
            cfg, FewsReader, "dev", "data/fews/dev/dev.few-shot.txt", with_embeddings=False
        )
        test_dataset = read_dataset_cached(
            cfg, FewsReader, "test", "data/fews/test/test.few-shot.txt", with_embeddings=False
        )
        test_dataset = dev_dataset + test_dataset
    else:
        raise Exception(f"Unknown corpus: {cfg.corpus_name}")

    print(train_dataset[0])
    return train_dataset, test_dataset


def read_stats(cfg):
    readf = lambda f: {k: int(v) for k, v in map(lambda l: l.strip().split("\t"), f)}
    with open(paths.freq_tsv_path(f"{cfg.corpus_name}_stats", "train", "label"), "r") as f:
        label_freqs = readf(f)
    with open(paths.freq_tsv_path(f"{cfg.corpus_name}_stats", "train", "lemma"), "r") as f:
        lemma_freqs = readf(f)
    return label_freqs, lemma_freqs


def get_lemma_f(cfg):
    if cfg.corpus_name == "clres":
        lemma_f = bssp.clres.dataset_reader.lemma_from_label
    elif cfg.corpus_name == "ontonotes":
        lemma_f = bssp.ontonotes.dataset_reader.lemma_from_label
    elif cfg.corpus_name == "semcor":
        lemma_f = bssp.semcor.dataset_reader.lemma_from_label
    elif cfg.corpus_name == "fews":
        lemma_f = bssp.fews.dataset_reader.lemma_from_label
    else:
        raise Exception(f"Unknown corpus: {cfg.corpus_name}")
    return lemma_f


def write_stats(cfg, train_dataset, test_dataset):
    lemma_f = get_lemma_f(cfg)
    train_labels, train_lemmas = dataset_stats("train", train_dataset, f"{cfg.corpus_name}_stats", lemma_f)
    test_labels, test_lemmas = dataset_stats("test", test_dataset, f"{cfg.corpus_name}_stats", lemma_f)
    return train_labels, train_lemmas


def predict(cfg):
    predictions_path = paths.predictions_tsv_path(cfg)
    if os.path.isfile(predictions_path):
        print(f"Reading predictions from {predictions_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = read_datasets(cfg)
    train_label_counts, train_lemma_counts = write_stats(cfg, train_dataset, test_dataset)

    indexer = make_indexer(cfg)
    vocab, embedder = make_embedder(cfg)

    # we're using a `transformers` model
    label_vocab = Vocabulary.from_instances(train_dataset)
    label_vocab.extend_from_instances(test_dataset)
    try:
        del label_vocab._token_to_index["tokens"]
    except KeyError:
        pass
    try:
        del label_vocab._index_to_token["tokens"]
    except KeyError:
        pass
    vocab.extend_from_vocab(label_vocab)

    print("Constructing model")
    if cfg.metric == "baseline":
        model = (
            RandomRetriever(
                vocab=vocab,
                target_dataset=train_dataset,
                device=device,
                top_n=cfg.top_n,
                same_lemma=True,
            )
            .eval()
            .to(device)
        )
    else:
        model = (
            NearestNeighborRetriever(
                vocab=vocab,
                embedder=embedder,
                target_dataset=train_dataset,
                distance_metric=cfg.metric,
                device=device,
                top_n=cfg.top_n,
                same_lemma=True,
            )
            .eval()
            .to(device)
        )
    dummy_reader = ClresConlluReader(split="train", token_indexers={"tokens": indexer})
    predictor = NearestNeighborPredictor(model=model, dataset_reader=dummy_reader)

    # remove any super-rare instances that did not occur at least 5 times in train
    # (these are not interesting to eval on)
    instances = [i for i in test_dataset if train_label_counts[i["label"].label] >= 5]
    # We are abusing the batch abstraction here--really a batch should be a set of independent instances,
    # but we are using it here as a convenient way to feed in a single instance.
    batches = batch_queries(instances, cfg.query_n)
    with open(predictions_path, "wt") as f, torch.no_grad():
        tsv_writer = csv.writer(f, delimiter="\t")
        header = ["sentence", "label", "lemma", "label_freq_in_train"]
        header += [f"label_{i+1}" for i in range(cfg.top_n)]
        header += [f"lemma_{i+1}" for i in range(cfg.top_n)]
        header += [f"sentence_{i+1}" for i in range(cfg.top_n)]
        header += [f"distance_{i+1}" for i in range(cfg.top_n)]
        tsv_writer.writerow(header)

        for batch in tqdm.tqdm(batches):
            # the batch results are actually all the same--just take the first one
            ds = predictor.predict_batch_instance(batch)
            d = ds[0]
            sentences = [[t.text for t in i["text"].tokens] for i in batch]
            spans = [i["label_span"] for i in batch]
            sentences = [
                format_sentence(sentence, span.span_start, span.span_end) for sentence, span in zip(sentences, spans)
            ]
            label = batch[0]["label"].label
            lemma = lemma_from_label(label)
            label_freq_in_train = train_label_counts[label]

            row = [" || ".join(sentences), label, lemma, label_freq_in_train]
            results = d[f"top_{cfg.top_n}"]
            results += [None for _ in range(cfg.top_n - len(results))]

            labels = []
            lemmas = []
            sentences = []
            distances = []

            for result in results:
                if result is None:
                    distances.append(88888888)
                    labels.append("")
                    lemmas.append("")
                    sentences.append("")
                else:
                    index, distance = result
                    distances.append(distance)
                    instance = train_dataset[index]
                    labels.append(instance["label"].label)
                    lemmas.append(lemma_from_label(labels[-1]))
                    span = instance["label_span"]
                    sentences.append(
                        format_sentence([t.text for t in instance["text"].tokens], span.span_start, span.span_end)
                    )

            row += labels
            row += lemmas
            row += sentences
            row += distances
            if len(row) != 204:
                print(len(row))
                assert False
            tsv_writer.writerow(row)

    print(f"Wrote predictions to {predictions_path}")


@cli.command(help="Calculate summary stats for a trial after it's been run")
@click.argument("corpus_slug")
@click.option("--embedding-model", help="`transformers` model slug to use", default="bert-base-cased")
@click.option(
    "--metric",
    help="how to measure embedding distance",
    default="cosine",
    type=click.Choice(["euclidean", "cosine", "baseline"], case_sensitive=False),
)
@click.option("--override-weights", help="Path to override weights from fine-tuning to use with the model")
@click.option("--top-n", type=int, default=50)
@click.option(
    "--query-n",
    type=int,
    default=1,
    help="Number of sentences to draw from when formulating a query. "
    "For n>1, embeddings of the target word will be average pooled.",
)
@click.option("--bert-layer", type=int, help="BERT layer (0-indexed) to average.", default=7)
def summarize(corpus_slug, embedding_model, metric, override_weights, top_n, query_n, bert_layer):
    cfg = Config(
        corpus_slug,
        embedding_model=embedding_model,
        override_weights_path=override_weights,
        metric=metric,
        top_n=top_n,
        query_n=query_n,
        bert_layers=[bert_layer],
    )
    label_freqs, lemma_freqs = read_stats(cfg)
    df = pd.read_csv(paths.predictions_tsv_path(cfg), sep="\t", error_bad_lines=False)
    lemma_f = get_lemma_f(cfg)

    low_freq, high_freq = (5, 500), (500, 1e9)
    low_rarity, high_rarity = (0.0, 0.25), (0.25, 1.0)

    for min_train_freq, max_train_freq in [low_freq, high_freq]:
        for min_rarity, max_rarity in [low_rarity, high_rarity]:
            metrics_at_k(
                cfg,
                df,
                label_freqs,
                lemma_freqs,
                lemma_f,
                min_train_freq=min_train_freq,
                max_train_freq=max_train_freq,
                min_rarity=min_rarity,
                max_rarity=max_rarity,
            )

            def pathf(ext):
                return paths.bucketed_metric_at_k_path(cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, ext)

            def baseline_pathf(ext):
                new_cfg = copy(cfg)
                new_cfg.metric = "baseline"
                return paths.bucketed_metric_at_k_path(cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, ext)

            def mean_average(d):
                sum = 0.0
                for k, v in d.items():
                    sum += v["label"]
                return sum / len(d)

            def get_f1d(precd, recd):
                d = {}
                for i in range(1, cfg.top_n + 1):
                    d[i] = {"label": 2 / (1 / recd[i]["label"] + 1 / precd[i]["label"])}
                return d

            def write_row(prec, rec, baseline_kind=None):
                f1d = get_f1d(prec, rec)
                if cfg.override_weights_path is not None:
                    finetuning_count = cfg.override_weights_path[
                        cfg.override_weights_path.rfind("_") + 1 : cfg.override_weights_path.rfind(".")
                    ]
                else:
                    finetuning_count = 0
                fname_prefix = ""
                fname_prefix += "low_freq" if min_train_freq == low_freq[0] else "high_freq"
                fname_prefix += "_"
                fname_prefix += "low_rarity" if min_rarity == low_rarity[0] else "all_rarity"
                fname_prefix += "_"

                with open(fname_prefix + "results.tsv", "a") as f:
                    # todo: query_n, bert_layers
                    vals = [
                        cfg.corpus_name,
                        baseline_kind if baseline_kind is not None else cfg.embedding_model,
                        ""
                        if cfg.metric == "baseline"
                        else (",".join(str(x) for x in cfg.bert_layers) if cfg.bert_layers is not None else ""),
                        finetuning_count,
                        mean_average(prec),
                        mean_average(rec),
                        mean_average(f1d),
                    ]
                    f.write("\t".join(str(x) for x in vals) + "\n")

            if cfg.metric == "baseline":
                oprec = pickle_read(pathf("oprec"))
                bprec = pickle_read(baseline_pathf("prec"))
                orec = pickle_read(pathf("orec"))
                brec = pickle_read(baseline_pathf("rec"))
                write_row(bprec, brec, baseline_kind="random baseline")
                write_row(oprec, orec, baseline_kind="oracle")
            else:
                prec = pickle_read(pathf("prec"))
                rec = pickle_read(pathf("rec"))
                write_row(prec, rec)


@cli.command(
    help="fine-tune a `transformers` model identified by transformer_model_name, "
    "and save the weights in PyTorch format to serialization_path"
)
@click.argument("transformer_model_name")
@click.argument("serialization_path")
@click.option("--corpus", default="streusle", help="Corpus to fine tune on")
@click.option("--num_insts", default=100, help="Number of instances to fine tune on")
def finetune(transformer_model_name, serialization_path, corpus, num_insts):
    # same number for each pos
    num_n = num_v = num_p = num_insts // 3

    if corpus == "streusle":
        # Read the data
        json_path = "data/streusle/train/streusle.ud_train.json"
        reader = make_streusle_reader(transformer_model_name, num_n, num_v, num_p)
        instances = list(reader.read(json_path))

        # Check that we were able to meet the quota
        required_number = (num_insts // 3) * 3
        if len(instances) < required_number:
            raise Exception(f"Requested {required_number} instances, but only got {len(instances)}")

        vocab = Vocabulary.from_instances(instances)
        loader = make_streusle_data_loader(instances, vocab)
    else:
        raise Exception(f"Unknown corpus: {corpus}")

    model = build_model(vocab, transformer_model_name)
    model.to("cuda:0")
    trainer = build_trainer(model, loader)
    trainer.train()
    transformer_model = model.embedder._token_embedders["tokens"]._matched_embedder.transformer_model
    torch.save(transformer_model.state_dict(), serialization_path)
    print(f"Saved fine-tuned weights for {transformer_model_name} on {num_insts} instances " f"to {serialization_path}")


def make_streusle_reader(transformer_model_name, num_n, num_v, num_p):
    indexer = PretrainedTransformerMismatchedIndexer(transformer_model_name)
    reader = StreusleJsonReader(
        tokenizer=None, token_indexers={"tokens": indexer}, max_n=num_n, max_v=num_v, max_p=num_p
    )
    return reader


def make_streusle_data_loader(instances, vocab):
    loader = SimpleDataLoader(instances, batch_size=8, vocab=vocab)
    return loader


def build_model(vocab, transformer_model_name):
    token_embedder = PretrainedTransformerMismatchedEmbedder(transformer_model_name, train_parameters=True)
    embedder = BasicTextFieldEmbedder({"tokens": token_embedder})
    model = StreusleFineTuningModel(vocab, embedder)
    return model


def build_trainer(model, loader):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = HuggingfaceAdamWOptimizer(parameters, lr=2e-5)
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=loader,
        validation_data_loader=loader,
        num_epochs=40,
        patience=5,
        optimizer=optimizer,
        run_sanity_checks=False
    )
    return trainer


if __name__ == "__main__":
    cli()
