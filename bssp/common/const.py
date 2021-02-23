import os

TRAIN_FREQ_BUCKETS = [[5,25], [25,100], [100,200], [200, 500000]]
PREVALENCE_BUCKETS = [[0, 0.01], [0.01,0.05], [0.05,0.15], [0.15,0.25], [0.25,0.5], [0.5,1]]


def read_nota_senses():
    nota_filepath = 'data/ontonotes_nota_senses.txt'
    if not os.path.isfile(nota_filepath):
        raise Exception("Populate `data/ontonotes_nota_senses.txt` by running `notebooks/senses.ipynb`")
    with open(nota_filepath, 'r') as f:
        return [s for s in f.read().split('\n') if s.strip()]


NOTA_SENSES = read_nota_senses()
