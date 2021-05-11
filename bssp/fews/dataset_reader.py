import os
from random import random
from typing import Dict, Iterable, List, Literal
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.fields import ArrayField, LabelField, SpanField, TextField
from allennlp_models.common.ontonotes import Ontonotes, OntonotesSentence
import numpy as np
from tqdm import tqdm

from bssp.common.embedder_model import EmbedderModelPredictor


def lemma_from_label(label):
    """Turn something like "make_v_1.0" into "make_v" """
    return label[: label.rfind("_")]


@DatasetReader.register("ontonotes")
class OntonotesReader(DatasetReader):
    def __init__(
        self,
        split: Literal["train", "test", "all", "none"],
        token_indexers: Dict[str, TokenIndexer] = None,
        embedding_predictor: EmbedderModelPredictor = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.token_indexers = token_indexers
        self.embedding_predictor = embedding_predictor

    def text_to_instance(
        self, tokens: List[str], span_start: int, span_end: int, label: str, embeddings: np.ndarray = None
    ) -> Instance:
        tokens = [Token(t) for t in tokens]
        # The text of the sentence in which the word sense appears
        text_field = TextField(tokens, self.token_indexers)
        # The word sense-annotated span, always of length 1 in ontonotes
        lemma_span_field = SpanField(span_start, span_end, text_field)
        # a label like "make_v_1.0"
        label_field = LabelField(label)
        # like above but without the sense number
        lemma_field = LabelField(lemma_from_label(label), label_namespace="lemma_labels")
        fields = {"text": text_field, "label_span": lemma_span_field, "label": label_field, "lemma": lemma_field}
        if self.embedding_predictor:
            fields["span_embeddings"] = ArrayField(embeddings[span_start : span_end + 1, :])

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        reader = Ontonotes()
        for doc_path in tqdm(Ontonotes.dataset_path_iterator(file_path)):
            for doc in reader.dataset_document_iterator(doc_path):
                for sent in doc:
                    if all(sense is None for sense in sent.word_senses):
                        continue
                    else:
                        tokens = sent.words
                        if self.embedding_predictor:
                            embeddings = np.array(self.embedding_predictor.predict(tokens)["embeddings"])
                        else:
                            embeddings = None
                        if not (len(sent.words) == len(sent.word_senses) == len(sent.predicate_lemmas)):
                            print("!!!!!!")
                            print(sent.words)
                            print(sent.word_senses)
                            print(sent.predicate_lemmas)
                            print(len(sent.words), len(sent.word_senses), len(sent.predicate_lemmas))

                        for i, sense in enumerate(sent.word_senses):
                            if sense is not None:
                                lemma = sent.predicate_lemmas[i]
                                pos_tag = sent.pos_tags[i]
                                simplified_pos = (
                                    "n" if pos_tag.startswith("N") else "v" if pos_tag.startswith("V") else None
                                )
                                # if simplified_pos is None:
                                #    raise Exception(f"POS tag not for noun or verb: {pos_tag}")
                                instance = self.text_to_instance(
                                    tokens, i, i, f"{lemma}_{simplified_pos}_{sense}", embeddings=embeddings
                                )
                                if random() < 0.0001:
                                    print(instance)
                                yield instance
