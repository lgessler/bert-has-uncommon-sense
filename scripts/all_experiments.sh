#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate sembre

for CORPUS in "ontonotes" "clres"; do
  python main.py trial --embedding-model "bert-base-cased" --metric "baseline" --query-n 1 --bert-layer 7 "$CORPUS"
  python main.py summarize --embedding-model "bert-base-cased" --metric "baseline" --query-n 1 --bert-layer 7 "$CORPUS"
  for BERT_MODEL in "bert-base-cased" "distilbert-base-cased" "roberta-base" "distilroberta-base" "albert-base-v2" "xlnet-base-cased" "gpt2"; do
    # last layer for distilled models is 6th
    if [ "$BERT_MODEL" == "distilbert-base-cased" ] || [ "$BERT_MODEL" == "distilroberta-base" ]; then
      BERT_LAYER=5
    else
      BERT_LAYER=11
    fi
    echo "$BERT_MODEL $BERT_LAYER"
    for NUM_INSTS in 0 100 250 500 1000 2500; do
      if [ "$NUM_INSTS" -eq "0" ];
      then
        python main.py trial --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER "$CORPUS"
        python main.py summarize --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER "$CORPUS"
      else
        WEIGHTS_LOCATION="models/${BERT_MODEL}_${NUM_INSTS}.pt"
        if [ ! -f "$WEIGHTS_LOCATION" ]; then
          python main.py finetune "$BERT_MODEL" "$WEIGHTS_LOCATION"
        fi
        python main.py trial --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER --override-weights "$WEIGHTS_LOCATION" "$CORPUS"
        python main.py summarize --embedding-model "$BERT_MODEL" --metric "cosine" --query-n 1 --bert-layer $BERT_LAYER --override-weights "$WEIGHTS_LOCATION" "$CORPUS"
      fi
    done
  done
done
