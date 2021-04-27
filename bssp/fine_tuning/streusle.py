import json
from random import shuffle
from typing import Dict, Iterable

from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance, Field
from allennlp.data.fields import TextField, LabelField, SpanField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer


@DatasetReader.register("streusle-json")
class StreusleJsonReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_n: int = 100,
        max_v: int = 100,
        max_p: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_n = max_n
        self.max_v = max_v
        self.max_p = max_p

    def text_to_instance(self, text: str, token_index: int, ss: str = None, ss2: str = None) -> Instance:  # type: ignore
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self.token_indexers)
        fields: Dict[str, Field] = {
            "text": text_field,
            "labeled_span": SpanField(token_index, token_index, text_field),
        }
        if ss is not None:
            # Portmanteau by default for now
            if ss2 is not None:
                ss += "_" + ss2
            fields["ss"] = LabelField(ss)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as f:
            sentences = json.load(f)
        n_count = 0
        v_count = 0
        p_count = 0
        shuffle(sentences)
        for sentence in sentences:
            tokens = [t["word"] for t in sentence["toks"]]
            for _, swe in sentence["swes"].items():
                token_id = swe["toknums"][0]
                lexcat = swe["lexcat"]
                if swe["ss"] is not None:
                    match = False
                    if lexcat == "N" and n_count < self.max_n:
                        match = True
                        n_count += 1
                    elif lexcat == "V" and v_count < self.max_v:
                        match = True
                        v_count += 1
                    elif lexcat == "P" and p_count < self.max_p:
                        match = True
                        p_count += 1
                    if match:
                        yield self.text_to_instance(
                            text=" ".join(tokens), token_index=token_id - 1, ss=swe["ss"], ss2=swe["ss2"]
                        )


xs = StreusleJsonReader().read("data/streusle/dev/streusle.ud_dev.json")
