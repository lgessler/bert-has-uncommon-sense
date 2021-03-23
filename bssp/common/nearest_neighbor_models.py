import random
from collections import defaultdict
from pprint import pprint
from typing import Dict, Any, List, Union, Literal, Iterator

import torch
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from torch.nn.functional import cosine_similarity, pairwise_distance
from allennlp.data import Vocabulary, TokenIndexer, Instance
from allennlp.models import Model
from allennlp.modules import TokenEmbedder, TextFieldEmbedder
from allennlp.predictors import Predictor
from allennlp.common.util import logger, JsonDict



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


@Model.register('nearest_neighbor_retriever')
class NearestNeighborRetriever(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 target_dataset: Iterator[Instance],
                 device: torch.device,
                 distance_metric: Literal["cosine", "euclidean"],
                 top_n: int,
                 same_lemma: bool = False):
        super().__init__(vocab)
        print(vocab)
        self.embedder = embedder
        self.target_dataset = target_dataset
        self.device = device
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
            lemma = str(instance['label'].label).split('_')[0]
            self.lemma_index[lemma].append(i)

        # needed for Predictor to work with this
        self.dummy_param = torch.nn.parameter.Parameter(torch.tensor([1.]))

    def forward(self,
                text: Dict[str, Dict[str, torch.Tensor]],
                label_span: torch.Tensor,
                label: torch.Tensor,
                lemma: torch.Tensor) -> Dict[str, Any]:
        # note the lemma of the query
        query_label_string = self.vocab.get_token_from_index(label[0].item(), namespace='labels')
        query_lemma_string = query_label_string[:query_label_string.find('_')]

        # get the sentence embedding
        # contextualized and uncontextualized embedders need separate handling
        embedded_text = self.embedder(text)

        # validation
        for i in range(len(label_span)):
            span_start = label_span[i][0].item()
            span_end = label_span[i][1].item()
            if span_end - span_start > 1:
                raise NotImplemented("Only single-word spans are currently supported")

        # Get the query embedding: in the general case, we have n words in context, and we take their average pool
        target_word_embeddings = [
            embedded_text[i][label_span[i][0].item()]
            for i in range(len(label_span))
        ]
        target_word_embeddings = torch.stack(target_word_embeddings, 0)
        query_embedding = torch.mean(target_word_embeddings, 0)
        query_embedding = query_embedding.reshape((1, -1))

        # if same_lemma is set to true, enforce the constraint that the lemma (but not necessarily
        # the sense of the lemma) be the same for both the word in the query and the word we're
        # looking at in the results
        target_embeddings = self.target_dataset_embeddings
        if self.same_lemma:
            lemma_indexes = self.lemma_index[query_lemma_string]
            target_embeddings = target_embeddings[lemma_indexes]

        # compute similarities and rank them
        distances = self.distance_function(target_embeddings, query_embedding)
        ranked_indices = torch.argsort(distances, descending=False)

        # return top n
        top_indices = ranked_indices[:50].to('cpu')
        top_distances = distances[top_indices].to('cpu')
        # need to map back to dataset indices if we were lemma-restricted
        if self.same_lemma:
            top_indices.apply_(lambda i: lemma_indexes[i])
        top_n_results = list(zip(
            top_indices,
            top_distances
        ))
        # cooperate with allennlp by pretending we have batch results
        result = {
            f'top_{self.top_n}': [top_n_results] * len(label_span)
        }
        return result


class NearestNeighborPredictor(Predictor):
    def predict(self,
                sentence: List[str],
                label_span_start: int,
                label_span_end: int,
                label: str) -> JsonDict:
        return self.predict_json({
            "sentence": sentence,
            "label_span_start": label_span_start,
            "label_span_end": label_span_end,
            "label": label
        })

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            tokens=json_dict['sentence'],
            span_start=json_dict['label_span_start'],
            span_end=json_dict['label_span_end'],
            lemma=json_dict['label'],
        )


@Model.register('random_retriever')
class RandomRetriever(Model):
    """
    Randomized baseline counterpart of NearestNeighborRetriever
    """
    def __init__(self,
                 vocab: Vocabulary,
                 target_dataset: Iterator[Instance],
                 device: torch.device,
                 top_n: int,
                 same_lemma: bool = False):
        super().__init__(vocab)
        self.target_dataset = target_dataset
        self.device = device
        self.top_n = top_n
        self.same_lemma = same_lemma

        if not all(inst['span_embeddings'].array.shape[0] == 1 for inst in self.target_dataset):
            raise NotImplemented("All spans must be length 1")
        self.target_dataset_embeddings = torch.stack(
            tuple(torch.tensor(inst['span_embeddings'].array[0]) for inst in target_dataset)
        ).to(device)

        # build index from lemma to all instances that have it
        self.lemma_index = defaultdict(list)
        for i, instance in enumerate(target_dataset):
            lemma = str(instance['label'].label).split('_')[0]
            self.lemma_index[lemma].append(i)

        # needed for Predictor to work with this
        self.dummy_param = torch.nn.parameter.Parameter(torch.tensor([1.]))

    def forward(self,
                text: Dict[str, Dict[str, torch.Tensor]],
                label_span: torch.Tensor,
                label: torch.Tensor = None,
                lemma: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # note the lemma of the query
        query_label_string = self.vocab.get_token_from_index(label.item(), namespace='labels')
        query_lemma_string = query_label_string[:query_label_string.find('_')]

        # get the sentence embedding
        # contextualized and uncontextualized embedders need separate handling

        span_start = label_span[0][0].item()
        span_end = label_span[0][1].item()
        if span_end - span_start > 1:
            raise NotImplemented("Only single-word spans are currently supported")

        # if same_lemma is set to true, enforce the constraint that the lemma (but not necessarily
        # the sense of the lemma) be the same for both the word in the query and the word we're
        # looking at in the results
        if self.same_lemma:
            indexes = self.lemma_index[query_lemma_string]
        else:
            indexes = torch.tensor(range(self.target_dataset_embeddings.shape[0])).to(self.device)

        # compute similarities and rank them
        random.shuffle(indexes)

        # return top n
        top_n_results = []
        for index in indexes:
            if len(top_n_results) >= self.top_n:
                break

            result_dict = [index, None]
            top_n_results.append(result_dict)
        # wrap in another list because we have a batch size of 1
        result = {f'top_{self.top_n}': [top_n_results]}
        return result
