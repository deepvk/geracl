import random
from collections import Counter


def random_derangement(n):
    """
    Return a random derangement of the list [0, 1, ..., n-1].
    That is, a permutation p such that p[i] != i for all i.
    If n=1, no derangement is possible; we'll return [0] as a fallback.
    """
    if n == 1:
        return [0]

    indices = list(range(n))
    while True:
        random.shuffle(indices)
        # Check if this shuffle has no fixed points
        if all(indices[i] != i for i in range(n)):
            return indices


def make_new_classes_multiclass(classes):
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
    max_attempts = 100
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

        # Pick 'classes_count[i]' number of classes from weighted_classes, skipping duplicates
        added = 0
        while added < negatives_count[i]:
            candidate = random.choice(weighted_classes)
            if candidate not in existing_set:
                new_classes[i].append(candidate)
                existing_set.add(candidate)
                added += 1
    return new_classes


def make_new_classes_multilabel(classes):
    flat_classes = [item for sample_classes in classes for item in sample_classes]
    frequency_counter = Counter(flat_classes)
    new_classes = [[] for _ in range(len(classes))]
    weighted_classes = list(frequency_counter.elements())
    classes_count = [random.randint(2, 3) for _ in range(len(classes))]
    for i in range(len(classes)):
        added = 0
        existing_set = set()
        if len(classes[i]) == 2:
            new_classes[i] += classes[i]
            classes_count[i] = 2
        else:
            while added < classes_count[i]:
                candidate = random.choice(classes[i])
                if candidate not in existing_set:
                    new_classes[i].append(candidate)
                    existing_set.add(candidate)
                    added += 1

    for i in range(len(new_classes)):
        existing_set = set(classes[i])  # to ensure uniqueness

        # Pick 'classes_count[i]' number of classes from weighted_classes, skipping duplicates
        added = 0
        while added < classes_count[i]:
            candidate = random.choice(weighted_classes)
            if candidate not in existing_set:
                new_classes[i].append(candidate)
                existing_set.add(candidate)
                added += 1
    return new_classes
