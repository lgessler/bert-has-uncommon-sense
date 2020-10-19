"""Quick and dirty classes for getting plain old embeddings for text"""
from typing import Dict, Any, List, Iterable

import torch
from allennlp.data import Vocabulary, TokenIndexer, Instance, AllennlpDataset, DatasetReader, Token
from allennlp.data.fields import TextField
from allennlp.models import Model
from allennlp.modules import TokenEmbedder
from allennlp.predictors import Predictor
from allennlp.common.util import logger, JsonDict


class EmbedderModel(Model):
    def __init__(self, vocab: Vocabulary, embedder: TokenEmbedder):
        super().__init__(vocab)
        self.embedder = embedder

    def forward(self, text: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if 'wordpiece_mask' in text['tokens']:
            embedded_text = self.embedder(
                token_ids=text['tokens']['token_ids'],
                mask=text['tokens']['mask'],
                wordpiece_mask=text['tokens']['wordpiece_mask'],
                offsets=text['tokens']['offsets'],
            )
        else:
            embedded_text = self.embedder(
                tokens=text['tokens']['tokens']
            )

        return {
            "embeddings": embedded_text
        }


class EmbedderModelPredictor(Predictor):
    def predict(self, sentence: List[str]) -> JsonDict:
        return self.predict_json({
            "sentence": sentence,
        })

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(
            tokens=json_dict['sentence'],
        )


class EmbedderDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer], **kwargs):
        super().__init__(**kwargs)
        self.token_indexers = token_indexers

    def text_to_instance(self, tokens: List[str]) -> Instance:
        tokens = [Token(t) for t in tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {
            'text': text_field,
        }
        return Instance(fields)

