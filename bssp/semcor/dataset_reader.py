from typing import Dict, Iterable, List, Literal
from itertools import islice
import random

import numpy as np
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, SpanField, ArrayField
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.common.util import logger
from nltk.corpus.reader import Lemma
from nltk.corpus import semcor as sc
from bssp.common.embedder_model import EmbedderModelPredictor


def tokens_of_sentence(sentence) -> List[str]:
    tokens = []
    for item in sentence:
        if type(item) is list and all(isinstance(t, str) for t in item):
            tokens += item
        else:
            tokens += item.flatten().leaves()
    return tokens


def spans_of_sentence(sentence):
    tokens = []
    for item in sentence:
        if type(item) is list and all(isinstance(t, str) for t in item):
            tokens += item
        else:
            item = item.flatten()
            label = item.label()
            span_tokens = item.leaves()
            yield span_tokens, (len(tokens), len(tokens) + len(span_tokens)), label
            tokens += span_tokens


def lemma_to_string(lemma: Lemma) -> str:
    if type(lemma) is str:
        logger.warning("lemma was a string instead of a Lemma: " + lemma)
        return lemma
    return lemma.synset().name() + '_' + lemma.name()


@DatasetReader.register('semcor')
class SemcorReader(DatasetReader):
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
        sentences = list(sc.tagged_sents(tag='sem'))
        # TODO: be wary that this is here
        if 'small' in file_path:
            sentences = list(islice(sentences, 50))

        # deterministically shuffle sentences
        random.Random(42).shuffle(sentences)

        if self.split == 'train':
            sentences = sentences[:int(len(sentences)*.8)]
            logger.info("Reading train split of semcor: " + str(len(sentences)))
        elif self.split == 'test':
            sentences = sentences[int(len(sentences)*.8):]
            logger.info("Reading test split of semcor: " + str(len(sentences)))
        elif self.split == 'all':
            logger.info("Reading all splits of semcor: " + str(len(sentences)))
        elif self.split == 'none':
            return []
        else:
            raise Exception("Unknown split: " + self.split)

        print(f"Beginning to read {len(sentences)} sentences")
        for sentence in sentences:
            tokens = tokens_of_sentence(sentence)
            spans = spans_of_sentence(sentence)
            if self.embedding_predictor:
                embeddings = np.array(self.embedding_predictor.predict(tokens)['embeddings'])
            else:
                embeddings = None
            for span_tokens, (i, j), lemma in spans:
                # j is currently exclusive, but SpanField is inclusive--subtract 1 from j
                j -= 1

                # don't consider multi-token words
                if j != i:
                    print('Skipping multiword instance ' + ' '.join(span_tokens))
                    continue
                if j >= len(tokens) - 1:
                    print(f"out of j out of bounds!\n\ttext: {tokens}\n\t{(i,j)}\n\t{lemma_to_string(lemma)}")
                    continue
                yield self.text_to_instance(tokens, i, j, lemma_to_string(lemma), embeddings=embeddings)
