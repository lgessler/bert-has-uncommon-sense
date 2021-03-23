from collections import defaultdict
from random import shuffle


def batch_queries(instances, query_n, full_batches_only=True):
    instances_by_label = defaultdict(list)
    for instance in instances:
        label = instance['label'].label
        instances_by_label[label].append(instance)

    batches = []
    for label, label_instances in instances_by_label.items():
        shuffle(label_instances)
        i = 0
        while i < len(label_instances):
            if full_batches_only and i+query_n > len(label_instances):
                break
            batches.append(label_instances[i:i+query_n])
            i += query_n

    return batches


SENTENCE_CACHE = {}


def format_sentence(sentence, i, j):
    key = (tuple(sentence), i, j)
    if key in SENTENCE_CACHE:
        return SENTENCE_CACHE[key]
    formatted = " ".join(sentence[:i] + ['>>' + sentence[i] + '<<'] + sentence[j+1:])
    SENTENCE_CACHE[key] = formatted
    return formatted
