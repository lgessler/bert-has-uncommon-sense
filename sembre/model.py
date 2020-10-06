from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register('semcor_retriever')
class SemcorRetriever(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder):
        super().__init__(vocab)
        self.embedder = embedder
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                text: Dict[str, Dict[str, torch.Tensor]],
                lemma_span: torch.Tensor,
                lemma_label: torch.Tensor) -> Dict[str, torch.Tensor]:

        print()
        print(text['tokens']['token_ids'].shape, end='\n\n')
        print(self.vocab)
        for batch in range(8):
            print()
            print([self.vocab.get_token_from_index(i.item()) for i in text['tokens']['token_ids'][batch]])
            span_start = lemma_span[batch][0].item()
            span_end = lemma_span[batch][1].item()
            wp_span_start = text['tokens']['offsets'][batch][span_start][0].item()
            wp_span_end = text['tokens']['offsets'][batch][span_end][0].item()
            print((span_start, span_end), (wp_span_start, wp_span_end))
            print([self.vocab.get_token_from_index(text['tokens']['token_ids'][batch][i].item())
                   for i in range(wp_span_start, wp_span_end)])
            print(self.vocab.get_token_from_index(lemma_label[batch].item(), namespace='labels'))
            print()
        print(text['tokens'].keys())
        print(lemma_span, lemma_span.shape, lemma_span[0], lemma_span[1])
        print()
        print(lemma_label.shape, end="\n\n")
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
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
