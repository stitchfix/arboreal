import random


def bootstrap(iterable, random_seed=None):
    if random_seed:
        random.seed(random_seed)

    results = []
    for _ in iterable:
        results.append(random.choice([i for i in iterable]))
    return results
