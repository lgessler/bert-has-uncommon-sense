import os


def read_nota_senses():
    nota_filepath = "data/ontonotes_nota_senses.txt"
    if not os.path.isfile(nota_filepath):
        raise Exception("Populate `data/ontonotes_nota_senses.txt` by running `notebooks/senses.ipynb`")
    with open(nota_filepath, "r") as f:
        return [s for s in f.read().split("\n") if s.strip()]


NOTA_SENSES = read_nota_senses()
