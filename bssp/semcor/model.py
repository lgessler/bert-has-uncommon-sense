import random
from collections import defaultdict
from pprint import pprint
from typing import Dict, Any, List, Union, Literal

import torch
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from torch.nn.functional import cosine_similarity, pairwise_distance
from allennlp.data import Vocabulary, TokenIndexer, Instance, AllennlpDataset
from allennlp.models import Model
from allennlp.modules import TokenEmbedder, TextFieldEmbedder
from allennlp.predictors import Predictor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.util import logger, JsonDict
from tqdm import tqdm


def format_sentence(sentence, i, j):
    return " ".join(sentence[:i] + ['>>' + sentence[i] + '<<'] + sentence[j+1:])


def is_bert(embedder: TextFieldEmbedder):
    return isinstance(embedder._token_embedders['tokens'], PretrainedTransformerMismatchedEmbedder)


def function_for_distance_metric(metric_name):
    """Return a function that accepts 2 tensors and returns a pairwise distance"""
    if metric_name == "cosine":
        return lambda x1, x2: 1 - cosine_similarity(x1, x2)
    elif metric_name == "euclidean":
        return lambda x1, x2: pairwise_distance(x1, x2, p=2)
    else:
        raise Exception(f"Invalid distance metric: \"{metric_name}\"")


@Model.register('semcor_retriever')
class SemcorRetriever(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 target_dataset: AllennlpDataset,
                 device: torch.device,
                 distance_metric: Literal["cosine", "euclidean"],
                 top_n: int,
                 same_lemma: bool = False):
        super().__init__(vocab)
        self.embedder = embedder
        self.accuracy = CategoricalAccuracy()
        self.target_dataset = target_dataset
        self.top_n = top_n
        self.distance_function = function_for_distance_metric(distance_metric)
        self.same_lemma = same_lemma

        if not all(inst['span_embeddings'].array.shape[0] == 1 for inst in self.target_dataset):
            raise NotImplemented("All spans must be length 1")
        self.target_dataset_embeddings = torch.stack(
            tuple(torch.tensor(inst['span_embeddings'].array[0]) for inst in target_dataset)
        ).to(device)

        # build index from lemma to all instances that have it
        self.lemma_index = defaultdict(list)
        for i, instance in enumerate(target_dataset):
            lemma = str(instance['lemma_label'].label).split('_')[0]
            self.lemma_index[lemma].append(i)


    def forward(self,
                text: Dict[str, Dict[str, torch.Tensor]],
                lemma_span: torch.Tensor,
                lemma_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # note the lemma of the query
        query_lemma_label_string = self.vocab.get_token_from_index(lemma_label.item(), namespace='labels')
        query_lemma_string = query_lemma_label_string[:query_lemma_label_string.find('_')]

        # get the sentence embedding
        # contextualized and uncontextualized embedders need separate handling
        embedded_text = self.embedder(text)

        # get information about the query
        if embedded_text.shape[0] > 1:
            raise NotImplemented("Use single-item batches")
        span_start = lemma_span[0][0].item()
        span_end = lemma_span[0][1].item()
        if span_end - span_start > 1:
            raise NotImplemented("Only single-word spans are currently supported")

        # if same_lemma is set to true, enforce the constraint that the lemma (but not necessarily
        # the sense of the lemma) be the same for both the word in the query and the word we're
        # looking at in the results
        if self.same_lemma:
            lemma_indexes = self.lemma_index[query_lemma_string]
            target_embeddings = self.target_dataset_embeddings[lemma_indexes]
            target_instances = [self.target_dataset[index] for index in lemma_indexes]
        else:
            target_embeddings = self.target_dataset_embeddings
            target_instances = self.target_dataset

        # compute similarities and rank them
        query_embedding = embedded_text[0][span_start]
        query_embedding = query_embedding.reshape((1, -1))
        distances = self.distance_function(target_embeddings, query_embedding)
        ranked_indices = torch.argsort(distances, descending=False)

        # return top n
        top_n_results = []
        for index in ranked_indices:
            if len(top_n_results) >= self.top_n:
                break

            instance = target_instances[index]
            span = instance['lemma_span']
            instance_lemma_label = str(instance['lemma_label'].label)

            result_dict = {
                'sentence': format_sentence([t.text for t in instance['text'].tokens],
                                            span.span_start, span.span_end),
                'label': instance_lemma_label,
                'distance': distances[index].item()
            }
            top_n_results.append(result_dict)
        # wrap in another list because we have a batch size of 1
        result = {f'top_{self.top_n}': [top_n_results]}
        return result


class SemcorPredictor(Predictor):
    def predict(self,
                sentence: List[str],
                lemma_span_start: int,
                lemma_span_end: int,
                lemma_label: str) -> JsonDict:
        return self.predict_json({
            "sentence": sentence,
            "lemma_span_start": lemma_span_start,
            "lemma_span_end": lemma_span_end,
            "lemma_label": lemma_label
        })

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            tokens=json_dict['sentence'],
            span_start=json_dict['lemma_span_start'],
            span_end=json_dict['lemma_span_end'],
            lemma=json_dict['lemma_label'],
        )
