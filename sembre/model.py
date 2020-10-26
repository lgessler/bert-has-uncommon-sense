import random
from pprint import pprint
from typing import Dict, Any, List, Union

import torch
from allennlp.modules.token_embedders import PretrainedTransformerMismatchedEmbedder
from torch.nn.functional import cosine_similarity
from allennlp.data import Vocabulary, TokenIndexer, Instance, AllennlpDataset
from allennlp.models import Model
from allennlp.modules import TokenEmbedder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.predictors import Predictor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.util import logger, JsonDict


def format_sentence(sentence, i, j):
    return " ".join(sentence[:i] + ['>>' + sentence[i] + '<<'] + sentence[j:])


def is_bert(embedder: TextFieldEmbedder):
    return isinstance(embedder._token_embedders['tokens'], PretrainedTransformerMismatchedEmbedder)


@Model.register('semcor_retriever')
class SemcorRetriever(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 target_dataset: AllennlpDataset,
                 device: Union[str, int],
                 top_n: int):
        super().__init__(vocab)
        self.embedder = embedder
        self.accuracy = CategoricalAccuracy()
        self.target_dataset = target_dataset
        self.top_n = top_n

        if not all(inst['span_embeddings'].array.shape[0] == 1 for inst in self.target_dataset):
            raise NotImplemented("All spans must be length 1")
        self.target_dataset_embeddings = torch.stack(
            tuple(torch.tensor(inst['span_embeddings'].array[0]) for inst in target_dataset)
        ).to(device)

    def forward(self,
                text: Dict[str, Dict[str, torch.Tensor]],
                lemma_span: torch.Tensor,
                lemma_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

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

        # compute similarities
        query_embedding = embedded_text[0][span_start]
        query_embedding = query_embedding.reshape((1, -1))
        dist = cosine_similarity(self.target_dataset_embeddings, query_embedding)
        indices = torch.argsort(dist, descending=True)

        # return top n
        result = {f'top_{self.top_n}': []}
        top_n = indices[:self.top_n]
        for i in top_n:
            instance = self.target_dataset[i]
            span = instance['lemma_span']
            result[f'top_{self.top_n}'].append({
                'sentence': format_sentence([t.text for t in instance['text'].tokens],
                                            span.span_start, span.span_end),
                'label': str(instance['lemma_label'].label),
                'cosine_sim': dist[i].item()
            })
        result[f'top_{self.top_n}'] = [result[f'top_{self.top_n}']]
        #pprint(result)

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
