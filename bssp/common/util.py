import os
from collections import Counter


def dataset_stats(split, dataset, directory, lemma_function):
    labels = Counter()
    lemmas = Counter()

    for instance in dataset:
        label = instance['label'].label
        lemma = lemma_function(label)
        labels[label] += 1
        lemmas[lemma] += 1

    os.makedirs(f'cache/{directory}', exist_ok=True)
    path = f'cache/{directory}/{split}'
    with open(path + '_label_freq.tsv', 'w', encoding='utf-8') as f:
        for item, freq in sorted(labels.items(), key=lambda x:-x[1]):
            f.write(f"{item}\t{freq}\n")
    with open(path + '_lemma_freq.tsv', 'w', encoding='utf-8') as f:
        for item, freq in sorted(lemmas.items(), key=lambda x:-x[1]):
            f.write(f"{item}\t{freq}\n")

    return labels, lemmas
