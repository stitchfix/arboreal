from core.datatype import Datatype


def gini_impurity(labels):
    # apple apple orange banana banana

    count = len(labels)
    unique = set(labels)
    contributions = []
    for label in unique:
        p = labels.count(label) / count
        impurity = p * (1.0 - p)
        contributions.append(impurity)
    return sum(contributions)


def mean_squared_error(targets):
    # 0.1, 0.3, 0.9

    mean = sum(targets) / len(targets)
    mse = sum([(target - mean) ** 2 for target in targets])
    return mse


def get_criterion(target_counter, target_type):
    # unoptimized pass-through to gini and mse; #TODO optimize

    unraveled_targets = []
    for target in target_counter:
        unraveled_targets.extend([target] * target_counter[target])

    if target_type == Datatype.categorical:
        criterion = gini_impurity(unraveled_targets)
    elif target_type == Datatype.numerical:
        criterion = mean_squared_error(unraveled_targets)
    else:
        raise ValueError("Unrecognized target_type")

    return criterion
