import re
from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.fields import ArrayField, LabelField, SpanField, TextField
import numpy as np
from tqdm import tqdm

from bssp.common.embedder_model import EmbedderModelPredictor


def lemma_from_label(label):
    return ".".join(label.split(".")[:2])


@DatasetReader.register("fews")
class FewsReader(DatasetReader):
    def __init__(
        self,
        split: str,
        token_indexers: Dict[str, TokenIndexer] = None,
        embedding_predictor: EmbedderModelPredictor = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_indexers = token_indexers
        self.embedding_predictor = embedding_predictor
        self.tokenizer = SpacyTokenizer("en_core_web_md")

    def text_to_instance(
        self, tokens: List[str], span_start: int, span_end: int, label: str, embeddings: np.ndarray = None
    ) -> Instance:
        tokens = [Token(t) for t in tokens]
        text_field = TextField(tokens, self.token_indexers)
        lemma_span_field = SpanField(span_start, span_end, text_field)
        label_field = LabelField(label)
        lemma_field = LabelField(lemma_from_label(label), label_namespace="lemma_labels")
        fields = {"text": text_field, "label_span": lemma_span_field, "label": label_field, "lemma": lemma_field}
        if self.embedding_predictor:
            fields["span_embeddings"] = ArrayField(embeddings[span_start : span_end + 1, :])

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as f:
            multiword_skipped = 0
            too_long_skipped = 0
            pbar = tqdm(f)
            for line in pbar:
                sent, label = line.strip().split("\t")
                match = re.search(r"<WSD>([^ ]*)</WSD>", sent)
                if match is None:
                    multiword_skipped += 1
                    pbar.set_postfix({"multiword_skipped": multiword_skipped, "too_long_skipped": too_long_skipped})
                    continue
                lefti, righti = match.span()
                sent_left = sent[:lefti]
                sent_right = sent[righti:]

                tokens = [str(t) for t in self.tokenizer.tokenize(sent_left)]
                index = len(tokens)
                tokens += [match.groups()[0]]
                tokens += [str(t) for t in self.tokenizer.tokenize(sent_right)]
                if len(tokens) > 300:
                    too_long_skipped += 1
                    pbar.set_postfix({"multiword_skipped": multiword_skipped, "too_long_skipped": too_long_skipped})
                    continue
                pbar.set_postfix({"multiword_skipped": multiword_skipped, "too_long_skipped": too_long_skipped})
                if self.embedding_predictor:
                    embeddings = np.array(self.embedding_predictor.predict(tokens)["embeddings"])
                else:
                    embeddings = None

                yield self.text_to_instance(tokens, index, index, label, embeddings=embeddings)
