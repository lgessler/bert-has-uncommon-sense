import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy


class StreusleFineTuningModel(Model):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder):
        super().__init__(vocab)
        self.embedder = embedder
        self.label_projection_layer = torch.nn.Linear(embedder.get_output_dim(), self.vocab.get_vocab_size("labels"))
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }

    def forward(self, text: TextFieldTensors, labeled_span: torch.Tensor, ss: torch.LongTensor):
        embedded_text = self.embedder(text)

        # labeled_span is (batch_size, 2)
        token_indices = labeled_span[:, 0].unsqueeze(-1)

        # use gather to select the embedding corresponding to every target token efficiently
        # see: https://discuss.pytorch.org/t/batch-index-select/62621
        dummy = token_indices.unsqueeze(2).expand(token_indices.size(0), token_indices.size(1), embedded_text.size(2))
        target_embeddings = torch.gather(embedded_text, 1, dummy).squeeze(1)

        ss_logits = self.label_projection_layer(target_embeddings)
        output = {"ss_logits": ss_logits}
        if ss is not None:
            for metric in self.metrics.values():
                metric(ss_logits, ss)
            output["loss"] = F.cross_entropy(ss_logits, ss)

        return output
