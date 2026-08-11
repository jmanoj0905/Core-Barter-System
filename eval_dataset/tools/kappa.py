from collections import Counter

LABELS = ("correct", "weakly_correct", "incorrect")


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    observed_agreement = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    chance_agreement = sum((count_a[l] / n) * (count_b[l] / n) for l in LABELS)

    if chance_agreement >= 1.0:
        return 0.0
    return (observed_agreement - chance_agreement) / (1 - chance_agreement)


def fleiss_kappa(labels_by_annotator: list[list[str]]) -> float:
    num_annotators = len(labels_by_annotator)
    num_items = len(labels_by_annotator[0])

    # counts[item][label] = number of annotators assigning that label
    counts = []
    for item_idx in range(num_items):
        item_labels = [labels_by_annotator[a][item_idx] for a in range(num_annotators)]
        counts.append(Counter(item_labels))

    p_item = []
    for item_counts in counts:
        agreements = sum(c * (c - 1) for c in item_counts.values())
        p_item.append(agreements / (num_annotators * (num_annotators - 1)))
    p_bar = sum(p_item) / num_items

    label_totals = Counter()
    for item_counts in counts:
        label_totals.update(item_counts)
    total_ratings = num_items * num_annotators
    p_e = sum((label_totals[l] / total_ratings) ** 2 for l in LABELS)

    if p_bar == 1.0:
        return 1.0
    if p_e >= 1.0:
        return 0.0
    return (p_bar - p_e) / (1 - p_e)


def session_kappa(annotator_label_lists: list[list[str]]) -> float:
    if len(annotator_label_lists) == 2:
        return cohens_kappa(*annotator_label_lists)
    return fleiss_kappa(annotator_label_lists)
