import os
from typing import Dict, Iterable, List, Literal
from allennlp.data import DatasetReader, Instance, TokenIndexer, Token
from allennlp.data.fields import ArrayField, LabelField, SpanField, TextField
from allennlp_models.common.ontonotes import Ontonotes, OntonotesSentence
import numpy as np

from bssp.common.embedder_model import EmbedderModelPredictor


@DatasetReader.register('ontonotes')
class OntonotesReader(DatasetReader):
    def __init__(self,
                 split: Literal['train', 'test', 'all', 'none'],
                 token_indexers: Dict[str, TokenIndexer] = None,
                 embedding_predictor: EmbedderModelPredictor = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.token_indexers = token_indexers
        self.embedding_predictor = embedding_predictor

    def text_to_instance(self,
                         tokens: List[str],
                         span_start: int,
                         span_end: int,
                         lemma: str,
                         embeddings: np.ndarray = None) -> Instance:
        Tokens = [Token(t) for t in tokens]
        text_field = TextField(Tokens, self.token_indexers)
        lemma_span_field = SpanField(span_start, span_end, text_field)
        lemma_label_field = LabelField(lemma)
        fields = {
            'text': text_field,
            'lemma_span': lemma_span_field,
            'lemma_label': lemma_label_field
        }
        if self.embedding_predictor:
            fields['span_embeddings'] = ArrayField(embeddings[span_start:span_end + 1, :])

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        reader = Ontonotes()
        for doc_path in Ontonotes.dataset_path_iterator(file_path):
            for doc in reader.dataset_document_iterator(doc_path):
                for sent in doc:
                    if all(sense is None for sense in sent.word_senses):
                        continue
                    else:
                        tokens = sent.words
                        if self.embedding_predictor:
                            embeddings = np.array(self.embedding_predictor.predict(tokens)['embeddings'])
                        else:
                            embeddings = None
                        if not (len(sent.words) == len(sent.word_senses) == len(sent.predicate_lemmas)):
                            print("!!!!!!")
                            print(sent.words)
                            print(sent.word_senses)
                            print(sent.predicate_lemmas)
                            print(len(sent.words), len(sent.word_senses), len(sent.predicate_lemmas))

                        for word, sense, lemma, i in zip(sent.words,
                                                         sent.word_senses,
                                                         sent.predicate_lemmas,
                                                         range(len(sent.words))):
                            if sense is not None:
                                yield self.text_to_instance(
                                    tokens, i, i, lemma + "_" + str(sense), embeddings=embeddings
                                )



