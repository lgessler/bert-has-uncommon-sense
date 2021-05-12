class Config:
    def __init__(
        self,
        corpus_name,
        embedding_model="bert-base-cased",
        override_weights_path=None,
        metric="cosine",
        top_n=50,
        query_n=1,
        bert_layers=None,
        train_freq_buckets=((5, 25), (25, 100), (100, 200), (200, 500000)),
        prevalence_buckets=((0.0, 0.05), (0.05, 0.15), (0.15, 0.25), (0.25, 0.5), (0.5, 1.0)),
    ):
        self.corpus_name = corpus_name
        self.embedding_model = embedding_model
        self.override_weights_path = override_weights_path
        self.metric = metric
        self.top_n = top_n
        self.query_n = query_n
        self.bert_layers = bert_layers
        self.train_freq_buckets = train_freq_buckets
        self.prevalence_buckets = prevalence_buckets

    def is_transformer(self):
        return self.embedding_model in ['gpt2'] or any(
            self.embedding_model.startswith(m) for m in ["roberta-", "bert-", "distilbert-", "distilroberta-", "xlnet-", "albert-"]
        )
