import logging
from typing import Dict, Iterable, List, Literal
from itertools import islice

from nltk.corpus import semcor as sc

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, SpanField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from nltk.corpus.reader import Lemma

logger = logging.getLogger(__name__)


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
    return lemma.synset().name() + lemma.name()


@DatasetReader.register('semcor')
class SemcorReader(DatasetReader):
    def __init__(self,
                 split: Literal['train', 'test'],
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, tokens: List[str], span_start: int, span_end: int, lemma: str):
        text_field = TextField([Token(t) for t in tokens], self.token_indexers)
        lemma_span_field = SpanField(span_start, span_end, text_field)
        lemma_label_field = LabelField(lemma)
        return Instance({
            'text': text_field,
            'lemma_span': lemma_span_field,
            'lemma_label': lemma_label_field
        })

    def _read(self, file_path: str) -> Iterable[Instance]:
        sentences = sc.tagged_sents(tag='sem')
        sentences = list(islice(sentences, 5))
        if self.split == 'train':
            sentences = sentences[:int(len(sentences)*.8)]
            logger.info("Reading train split of semcor: " + str(len(sentences)))
        else:
            sentences = sentences[int(len(sentences)*.8):]
            logger.info("Reading test split of semcor: " + str(len(sentences)))

        for sentence in sentences:
            tokens = tokens_of_sentence(sentence)
            spans = spans_of_sentence(sentence)
            for span_tokens, (i, j), lemma in spans:
                # don't consider multi-token words
                if j - i != 1:
                    logging.info('Skipping multiword instance ' + ' '.join(span_tokens))
                    continue
                print(lemma, type(lemma))
                yield self.text_to_instance(tokens, i, j, lemma_to_string(lemma))
