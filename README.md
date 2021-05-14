# Usage
## Data
In order to reproduce the OntoNotes data, you will need to have a copy of OntoNotes 5.0 in 
[CONLL format](https://cemantix.org/data/ontonotes.html). Cf. https://docs.allennlp.org/models/main/models/common/ontonotes/

## Setup
1. Initiate submodules:

```bash
git submodule init
git submodule update
```

2. Create a new Anaconda environment:

```bash
$ conda create --name bhus python=3.8
```

3. Install dependencies:

```
conda activate bhus
pip install -r requirements.txt
```

4. Perform a test run to ensure everything's working:

```
mkdir models
# finetune the model on a small number of STREUSLE instances
python main.py finetune --num_insts 100 distilbert-base-cased models/distilbert-base-cased_100.pt
# run the trials using the finetuned model we just created on pdep--note `clres` is an alias we use for pdep
python main.py trial \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    --override-weights models/distilbert-base-cased_100.pt \
    clres
# analyze results by bucket
python main.py summarize \
    --embedding-model distilbert-base-cased \
    --metric cosine \
    --query-n 1 \
    --bert-layer 5 \
    --override-weights models/distilbert-base-cased_100.pt \
    clres
```

## Execution

Type `bash scripts/all_experiments.sh` to yield the numbers used in the paper's tables, which will appear in TSV 
files by bucket. For individual results on each trial, see `.tsv` files under `cache/`.
In between runs, clear `cache/`, `models/`, and `*.tsv`.

# Acknowledgments
Thanks to Ken Litkowski for allowing us to distribute a subset of the 
[PDEP corpus](https://www.aclweb.org/anthology/P14-1120.pdf) with our code. 