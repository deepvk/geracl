import random
from collections import Counter

import numpy as np


def random_derangement(n):
    """
    Return a random derangement of the list [0, 1, ..., n-1].
    That is, a permutation p such that p[i] != i for all i.
    If n=1, no derangement is possible; return [0] as a fallback.
    """
    if n == 1:
        return [0]

    indices = list(range(n))
    while True:
        random.shuffle(indices)
        # Check if this shuffle has no fixed points
        if all(indices[i] != i for i in range(n)):
            return indices


def generate_classes_multiclass(classes):
    flat_classes = [item for sample_classes in classes for item in sample_classes]
    frequency_counter = Counter(flat_classes)
    new_classes = [[] for _ in range(len(classes))]
    weighted_classes = list(frequency_counter.elements())
    negatives_count = [random.randint(1, 3) for _ in range(len(classes))]

    for i in range(len(classes)):
        selected_positive = random.choice(classes[i])
        new_classes[i] = [selected_positive]

    derangement = random_derangement(len(classes))

    fixed = True
    max_attempts = 200
    attempts = 0

    while True:
        fixed = True
        for i in range(len(classes)):
            negative_index = derangement[i]
            if new_classes[i][0] in classes[negative_index]:
                # We have a collision: the item for i is the same as for j
                # Re-pick a new random item for i.
                new_item = random.choice(classes[i])
                tries = 0
                while new_item in classes[negative_index] and tries < max_attempts:
                    new_item = random.choice(classes[i])
                    tries += 1
                new_classes[i][0] = new_item
                fixed = False
        attempts += 1
        # Break if no collisions or we tried too many times
        if fixed or attempts >= max_attempts:
            if attempts >= max_attempts:
                raise Exception(
                    "Could not distribute one positive class from each sample to different samples without repetition of classes."
                )
            break

    for i in range(len(classes)):
        new_classes[derangement[i]].append(new_classes[i][0])

    for i in range(len(new_classes)):
        existing_set = set(new_classes[i]).union(set(classes[i]))  # to ensure uniqueness

        added = 0
        while added < negatives_count[i]:
            candidate = random.choice(weighted_classes)
            if candidate not in existing_set:
                new_classes[i].append(candidate)
                existing_set.add(candidate)
                added += 1
    return new_classes


def generate_classes_multilabel(classes):
    flat_classes = [item for sample_classes in classes for item in sample_classes]
    frequency_counter = Counter(flat_classes)
    new_classes = [[] for _ in range(len(classes))]
    weighted_classes = list(frequency_counter.elements())
    positives_count = [random.randint(2, 3) for _ in range(len(classes))]
    negatives_count = [random.randint(2, 3) for _ in range(len(classes))]
    for i in range(len(classes)):
        added = 0
        existing_set = set()
        if len(classes[i]) == 2:
            new_classes[i] += classes[i]
            positives_count[i] = 2
        else:
            while added < positives_count[i]:
                candidate = random.choice(classes[i])
                if candidate not in existing_set:
                    new_classes[i].append(candidate)
                    existing_set.add(candidate)
                    added += 1

    for i in range(len(new_classes)):
        existing_set = set(new_classes[i]).union(set(classes[i]))  # to ensure uniqueness

        added = 0
        while added < negatives_count[i]:
            candidate = random.choice(weighted_classes)
            if candidate not in existing_set:
                new_classes[i].append(candidate)
                existing_set.add(candidate)
                added += 1
    return new_classes


def shuffle_classes(classes, positives):
    rng = np.random.default_rng()
    labels = [[] for _ in range(len(classes))]
    for i, sample_classes in enumerate(classes):
        lower_positives = [positive.lower() for positive in positives[i]]
        for sample_class in sample_classes:
            if sample_class.lower() in lower_positives:
                labels[i].append(1)
            else:
                labels[i].append(0)

    for i, sample_classes in enumerate(classes):
        permutation = rng.permutation(len(sample_classes))
        classes[i] = [sample_classes[k] for k in permutation]
        labels[i] = [labels[i][k] for k in permutation]

    new_labels = [[] for _ in range(len(classes))]
    for i, sample_labels in enumerate(labels):
        for k, sample_label in enumerate(sample_labels):
            if sample_label == 1:
                new_labels[i].append(k)
    return classes, new_labels


def choose_synthetic_classes(batch, include_scenarios=False) -> list | tuple[list, list]:
    new_classes = [[] for _ in range(len(batch))]
    if include_scenarios:
        scenarios = []
    for i in range(len(batch)):
        negative_ids = [0, 1, 2, 3, 4]
        negative_idx = random.choice(negative_ids)
        while not batch[i][f"classes_{negative_idx}"]:
            negative_ids.remove(negative_idx)
            negative_idx = random.choice(negative_ids)
        new_classes[i] = batch[i][f"classes_{negative_idx}"]
        if include_scenarios:
            scenario_idx = 0
            for k in range(negative_idx):
                if not batch[i][f"classes_{negative_idx}"]:
                    continue
                else:
                    scenario_idx += 1
            scenarios.append(batch[i]["scenarios"][scenario_idx])
    if include_scenarios:
        return new_classes, scenarios
    else:
        return new_classes


def generate_classes(classes: list, config: str):
    if config == "synthetic_positives_multiclass":
        new_classes = generate_classes_multiclass(classes)
    elif config == "synthetic_positives_multilabel":
        new_classes = generate_classes_multilabel(classes)

    new_classes, labels = shuffle_classes(new_classes.copy(), classes)
    return new_classes, labels
