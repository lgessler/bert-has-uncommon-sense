import os

import click
import json

import torch
from allennlp.data import Vocabulary
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.token_indexers import PretrainedTransformerMismatchedIndexer
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from transformers import AdamW, BertForTokenClassification, BertTokenizer, AutoTokenizer, AutoModel

from bssp.fine_tuning.models import StreusleFineTuningModel
from bssp.fine_tuning.streusle import StreusleJsonReader

STREUSLE_PATH = "data/streusle"


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
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=loader,
        validation_data_loader=loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer


@click.command()
@click.argument("transformer_model_name")
@click.argument("serialization_path")
@click.option("--corpus", default="streusle", help="Corpus to fine tune on")
@click.option("--num_n", default=100, help="Number of noun instances to fine tune on")
@click.option("--num_v", default=100, help="Number of verb instances to fine tune on")
@click.option("--num_p", default=100, help="Number of preposition instances to fine tune on")
def main(transformer_model_name, serialization_path, corpus, num_n, num_v, num_p):

    if corpus == "streusle":
        json_path = "data/streusle/dev/streusle.ud_dev.json"
        reader = make_streusle_reader(transformer_model_name, num_n, num_v, num_p)
        instances = list(reader.read(json_path))
        vocab = Vocabulary.from_instances(instances)
        loader = make_streusle_data_loader(instances, vocab)
        print(len(instances))
    else:
        raise Exception(f"Unknown corpus: {corpus}")

    model = build_model(vocab, transformer_model_name)
    model.to("cuda:0")
    trainer = build_trainer(model, loader)
    trainer.train()
    transformer_model = model.embedder._token_embedders["tokens"]._matched_embedder.transformer_model
    torch.save(transformer_model.state_dict(), serialization_path)
    # transformer_model.save_pretrained(serialization_path)
    # AutoModel.from_pretrained(serialization_path)
    # tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    # tokenizer.save_pretrained(serialization_path)
    # - after it's done, save the embedder, which should just be bert... should it be allennlp wrapped or not?


if __name__ == "__main__":
    main()
