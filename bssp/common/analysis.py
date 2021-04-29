from collections import defaultdict, Counter
from ldg.pickle import pickle_write
from tqdm import tqdm

from bssp.common import paths
from bssp.common.const import NOTA_SENSES


def metrics_at_k(
    cfg,
    df,
    label_freqs,
    lemma_freqs,
    lemma_f,
    min_train_freq,
    max_train_freq,
    min_rarity,
    max_rarity,
    query_category=None,
    pos=None,
):
    # for pickles
    def f1():
        def f2():
            return 0

        return defaultdict(f2)

    score_dict = defaultdict(f1)

    count = 0
    lemmas = set()
    # Computation of metrics at k would be more straightforward with k in the outer loop and rows in the inner loop, but
    # this is very inefficient. Instead, we reverse the loop order.
    for _, row in tqdm(df.iterrows()):
        label = row.label
        lemma = lemma_f(label)

        # By default for ontonotes, only consider non-nota senses
        if cfg.corpus_name == 'ontonotes' and label in NOTA_SENSES:
            continue
        # skipped named entities for semcor
        if cfg.corpus_name == 'semcor' and (label == 'NE' or '_' not in label):
            continue

        rarity = label_freqs[label] / lemma_freqs[lemma]

        # Handling for query_category and pos--disabled for now
        # if query_category and (
        #     query_category == "non-nota"
        #     and label in NOTA_SENSES
        #     or query_category == "nota"
        #     and label not in NOTA_SENSES
        # ):
        #     continue
        # if pos and (pos == "n" and "_n_" not in label) or (pos == "v" and "_v_" not in label):
        #     continue

        # take care that we're in the proper bucket
        if not (min_rarity <= rarity < max_rarity):
            continue
        if not (min_train_freq <= row.label_freq_in_train < max_train_freq):
            continue

        count += 1
        lemmas.add(lemma)

        num_labels_correct = 0
        num_lemmas_correct = 0
        for k in range(1, cfg.top_n + 1):

            # Do we have the correct label/lemma?
            label_is_correct = getattr(row, f"label_{k}") == label
            lemma_is_correct = getattr(row, f"lemma_{k}") == lemma
            num_labels_correct += label_is_correct
            num_lemmas_correct += lemma_is_correct

            # accumulate the numerator for precision
            score_dict[k]["label"] += num_labels_correct
            score_dict[k]["lemma"] += num_lemmas_correct

            # accumulate denominators for precision
            score_dict[k]["total"] += k
            # ... and recall
            score_dict[k]["label_total"] += label_freqs[label]
            score_dict[k]["lemma_total"] += lemma_freqs[lemma]

            # oracle recall and precision: simulate if every item was gold. this is the numerator
            score_dict[k]["oracle_recall"] += min(k, label_freqs[label])
            score_dict[k]["oracle_precision"] += min(k, label_freqs[label])

    if count == 0:
        print("No instances in this bin, skipping")
        return None, None

    ps_at_k = defaultdict(lambda: dict())
    for k in range(1, cfg.top_n + 1):
        for l in ["label", "lemma"]:
            ps_at_k[k][l] = score_dict[k][l] / score_dict[k]["total"]

    recalls_at_k = defaultdict(lambda: dict())
    for k in range(1, cfg.top_n + 1):
        for l in ["label", "lemma"]:
            recalls_at_k[k][l] = score_dict[k][l] / score_dict[k][f"{l}_total"]

    oracle_recalls_at_k = defaultdict(lambda: dict())
    for k in range(1, cfg.top_n + 1):
        oracle_recalls_at_k[k]["label"] = score_dict[k][f"oracle_recall"] / score_dict[k]["label_total"]

    oracle_precisions_at_k = defaultdict(lambda: dict())
    for k in range(1, cfg.top_n + 1):
        oracle_precisions_at_k[k]["label"] = score_dict[k][f"oracle_precision"] / score_dict[k]["total"]

    # write to pickles
    def path_f(ext):
        return paths.bucketed_metric_at_k_path(
            cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, ext, query_category=query_category, pos=pos
        )

    ps_at_k = dict(ps_at_k)
    for key, value in ps_at_k.items():
        ps_at_k[key] = dict(value)
    pickle_write(ps_at_k, path_f("prec"))

    recalls_at_k = dict(recalls_at_k)
    for key, value in recalls_at_k.items():
        recalls_at_k[key] = dict(value)
    pickle_write(recalls_at_k, path_f("rec"))

    oracle_recalls_at_k = dict(oracle_recalls_at_k)
    for key, value in oracle_recalls_at_k.items():
        oracle_recalls_at_k[key] = dict(value)
    pickle_write(oracle_recalls_at_k, path_f("orec"))

    oracle_precisions_at_k = dict(oracle_precisions_at_k)
    for key, value in oracle_precisions_at_k.items():
        oracle_precisions_at_k[key] = dict(value)
    pickle_write(oracle_precisions_at_k, path_f("oprec"))

    pickle_write(lemmas, path_f("lemmas"))
    pickle_write(count, path_f("count"))

    return recalls_at_k, recalls_at_k


def dataset_stats(split, dataset, directory, lemma_function):
    labels = Counter()
    lemmas = Counter()

    for instance in dataset:
        label = instance["label"].label
        lemma = lemma_function(label)
        labels[label] += 1
        lemmas[lemma] += 1

    with open(paths.freq_tsv_path(directory, split, "label"), "w", encoding="utf-8") as f:
        for item, freq in sorted(labels.items(), key=lambda x: -x[1]):
            f.write(f"{item}\t{freq}\n")
    with open(paths.freq_tsv_path(directory, split, "lemma"), "w", encoding="utf-8") as f:
        for item, freq in sorted(lemmas.items(), key=lambda x: -x[1]):
            f.write(f"{item}\t{freq}\n")

    return labels, lemmas
