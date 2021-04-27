import os
from glob import glob
from typing import Dict, Iterable, List, Literal, Any

import conllu
from allennlp.common.logging import logger
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.fields import ArrayField, LabelField, SpanField, TextField
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup as BS

from bssp.common.embedder_model import EmbedderModelPredictor


def lemma_from_label(label):
    """Turn something like "make_v_1.0" into "make_v" """
    return label[: label.rfind("_")]


@DatasetReader.register("clres")
class ClresReader(DatasetReader):
    def __init__(
        self,
        split: Literal["train", "test", "all"],
        token_indexers: Dict[str, TokenIndexer] = None,
        embedding_predictor: EmbedderModelPredictor = None,
        answers: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.token_indexers = token_indexers
        self.embedding_predictor = embedding_predictor
        self.answers = answers

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
        xml_file_paths = sorted(glob(file_path + os.sep + "*.xml"))
        for i, xml_file_path in enumerate(xml_file_paths):
            print(f"[{i}/{len(xml_file_paths)}] Processing", xml_file_path)
            with open(xml_file_path, "r", encoding="utf8") as f:
                soup = BS(f.read(), "html.parser")
            lemma = soup.lexelt["item"]
            for instance in tqdm(soup.find_all("instance")):
                if instance.answer:
                    label = lemma + "_" + instance.answer["senseid"]
                elif instance["id"] in self.answers:
                    label = lemma + "_" + self.answers[instance["id"]]["sense_id"]
                else:
                    logger.warn(f"{instance['id']} not found in keys! skipping")
                    continue

                left_str, target, right_str = list(instance.context.children)
                left_tokens = str(left_str).strip().split(" ")
                right_tokens = str(right_str).strip().split(" ")
                i = len(left_tokens)
                tokens = left_tokens + [target.text] + right_tokens
                if self.embedding_predictor:
                    embeddings = np.array(self.embedding_predictor.predict(tokens)["embeddings"])
                else:
                    embeddings = None
                yield self.text_to_instance(tokens, i, i, label, embeddings=embeddings)


@DatasetReader.register("clres_conllu")
class ClresConlluReader(DatasetReader):
    def __init__(
        self,
        split: Literal["train", "test", "all"],
        token_indexers: Dict[str, TokenIndexer] = None,
        embedding_predictor: EmbedderModelPredictor = None,
        answers: Dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_indexers = token_indexers
        self.embedding_predictor = embedding_predictor
        self.answers = answers

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
        with open(file_path, "r") as f:
            tokenlists = conllu.parse(f.read())

        for tokenlist in tokenlists:
            prep_index = int(tokenlist.metadata["prep_id"]) - 1
            tokens = [t["form"] for t in tokenlist]
            lemma = tokenlist[prep_index]["lemma"]
            label = tokenlist[prep_index]["misc"]["Sense"]
            if self.embedding_predictor:
                embeddings = np.array(self.embedding_predictor.predict(tokens)["embeddings"])
            else:
                embeddings = None
            yield self.text_to_instance(tokens, prep_index, prep_index, lemma + "_" + label, embeddings=embeddings)
