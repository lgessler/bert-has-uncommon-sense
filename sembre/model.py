from typing import Dict, Any, List

import torch
from allennlp.data import Vocabulary, TokenIndexer, Instance, AllennlpDataset
from allennlp.models import Model
from allennlp.modules import TokenEmbedder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.predictors import Predictor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.util import logger, JsonDict

from sembre.dataset_reader import SemcorReader


@Model.register('semcor_retriever')
class SemcorRetriever(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TokenEmbedder,
                 target_dataset: AllennlpDataset):
        super().__init__(vocab)
        self.embedder = embedder
        self.accuracy = CategoricalAccuracy()
        self.target_dataset = target_dataset
        print('111', target_dataset[0])
        # TODO: build a matrix for cosining against from this

    def forward(self,
                text: Dict[str, Dict[str, torch.Tensor]],
                lemma_span: torch.Tensor,
                lemma_label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        print()
        print(text['tokens']['token_ids'].shape, end='\n\n')
        for batch in range(1):
            print()
            print('received sentence:', [self.vocab.get_token_from_index(i.item()) for i in text['tokens']['token_ids'][batch]])
            span_start = lemma_span[batch][0].item()
            span_end = lemma_span[batch][1].item()
            print(text['tokens'])
            wp_span_start = text['tokens']['offsets'][batch][span_start][0].item()
            wp_span_end = text['tokens']['offsets'][batch][span_end][0].item()
            print('normal offset, and wordpiece offset: ', (span_start, span_end), (wp_span_start, wp_span_end))
            print('target span:', [self.vocab.get_token_from_index(text['tokens']['token_ids'][batch][i].item())
                   for i in range(wp_span_start, wp_span_end)])
            print('span label:', self.vocab.get_token_from_index(lemma_label[batch].item(), namespace='labels'))
            print()
        print(text['tokens'].keys())
        print()
        print(lemma_label.shape, end="\n\n")
        # Shape: (batch_size, num_tokens, embedding_dim)
        # TODO: genericize this to non-wordpiece embedders
        embedded_text = self.embedder(
            token_ids=text['tokens']['token_ids'],
            mask=text['tokens']['mask'],
            wordpiece_mask=text['tokens']['wordpiece_mask'],
            offsets=text['tokens']['offsets'],
        )
        print(embedded_text.shape)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)

        # TODO:
        # - read train split (in constructor)
        # - handle case where single word has multiple wordpieces
        # - cosine sim over it and implement the metric

        assert False
        ## Shape: (batch_size, encoding_dim)
        #encoded_text = self.encoder(embedded_text, mask)
        ## Shape: (batch_size, num_labels)
        #logits = self.classifier(encoded_text)
        ## Shape: (batch_size, num_labels)
        #probs = torch.nn.functional.softmax(logits, dim=-1)
        ## Shape: (1,)
        #output = {'probs': probs}
        #if label is not None:
        #    self.accuracy(logits, label)
        #    output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


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
