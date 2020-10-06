local bert_model = 'bert-base-cased';

{
    "dataset_reader" : {
        "type": "semcor",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": bert_model,
                "namespace": "tokens"
            }
        },
        "split": "test"
    },
    "train_data_path": "UNUSED",
    "validation_data_path": "UNUSED",
    "model": {
        "type": "semcor_retriever",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": bert_model
                }
            }
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5,
        "cuda_device": 0
    },
}
