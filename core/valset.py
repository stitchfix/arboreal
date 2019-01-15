import math
import itertools
from core.datatype import Datatype
import random


def powersubset(s, size_limit):
    return frozenset(itertools.combinations(s, size_limit))


def replace_NaNs_with_None(s):
    no_nans = {v for v in s if v == v}  # strip NaNs
    if len(no_nans) < len(s):  # if NaNs were stripped
        no_nans.add(None)
    return no_nans


def _create_val_set_set_set_numerical(
    numerical_set, max_values_considered=100
):
    # take in a set of unique numerical values and generate a set of all
    # valsetsets, where a single valsetset corresponds to a split to evaluate

    # current implementation creates binary or trinary splits: two windowed
    # branches, and, if there are missing values present, a third None branch.
    # future versions could create multiple windowed branches without too
    # much trouble.

    # input validation
    assert isinstance(numerical_set, set), "numerical_set should be a set"

    # We treat NaN as a missing value, like None, but since NaN logic works
    # confusingly with sets (NaN being != NaN, and therefore set membership
    # being difficult), we treat create a version of numerical_set with NaN
    # replaced by None
    numerical_set = replace_NaNs_with_None(numerical_set)
    contains_missing = None in numerical_set

    candidate_values = sorted(numerical_set.difference([None]))
    if max_values_considered:
        random.shuffle(candidate_values)
        candidate_values = candidate_values[:max_values_considered]

    val_set_set_set = set()
    for val in candidate_values:
        lower_val_set = (float("-inf"), val)  # -inf -> val
        upper_val_set = (val, float("inf"))  # val -> +inf
        if contains_missing:
            none_val_set = frozenset([None])
            val_set_set = frozenset(
                (lower_val_set, upper_val_set, none_val_set)
            )
        else:
            val_set_set = frozenset((lower_val_set, upper_val_set))
        val_set_set_set.add(val_set_set)
    return frozenset(val_set_set_set)


# def _deprecated_create_trinary_val_set_set_set_categorical(categorical_set):
#     assert isinstance(categorical_set, set), "categorical_set should be a set"
#     # take in a set of unique categorical values and generate a set of all
#     # valsetsets, where a single valsetset corresponds to a split to evaluate

#     # current implementation creates trinary splits
#     categorical_set = categorical_set.union(
#         set([None])
#     )  # make None an option if it's not already

#     val_set_set_set = set()
#     for first_two_categories in itertools.combinations(categorical_set, 2):
#         first = frozenset([first_two_categories[0]])
#         second = frozenset([first_two_categories[1]])
#         remainder = frozenset(
#             categorical_set.difference(set(first_two_categories))
#         )
#         val_set_set = frozenset((first, second, remainder))
#         val_set_set_set.add(val_set_set)

#     return frozenset(val_set_set_set)


def _create_val_set_set_set_categorical(categorical_set):
    assert isinstance(categorical_set, set), "categorical_set should be a set"

    # We treat NaN as a missing value, like None, but since NaN logic works
    # confusingly with sets (NaN being != NaN, and therefore set membership
    # being difficult), we treat create a version of categorical_set with NaN
    # replaced by None
    categorical_set = replace_NaNs_with_None(categorical_set)

    val_set_set_set = set()

    for cat in categorical_set:
        cat_branch = frozenset([cat])
        remainder_branch = frozenset(categorical_set.difference(cat_branch))
        val_set_set = frozenset((cat_branch, remainder_branch))
        val_set_set_set.add(val_set_set)

    return frozenset(val_set_set_set)


def create_val_set_set_set(categorical_or_numerical_set, datatype):
    if datatype == Datatype.numerical:
        return _create_val_set_set_set_numerical(categorical_or_numerical_set)
    elif datatype == Datatype.categorical:
        return _create_val_set_set_set_categorical(
            categorical_or_numerical_set
        )
    else:
        raise ValueError("Unrecognized datatype")


def val_set_contains(val_set, val):
    # We treat NaN as a missing value, like None, but since NaN logic works
    # confusingly with sets (NaN being != NaN, and therefore set membership
    # being difficult), we treat NaN as None
    try:
        if math.isnan(val):  # errors if val is a str
            val = None
    except TypeError:
        pass

    if isinstance(val_set, frozenset):
        return val in val_set
    elif isinstance(val_set, tuple) and len(val_set) == 2:
        if val is None:
            return False
        else:
            return val_set[0] <= val < val_set[1]
    else:
        raise ValueError("Unrecognized val_set")


def get_val_set_for_feature_value(val_set_set, feature_value):
    matching_val_set_count = 0
    matching_val_set = None
    for val_set in val_set_set:
        if val_set_contains(val_set, feature_value):
            matching_val_set_count += 1
            matching_val_set = val_set
    assert (
        matching_val_set_count == 1
    ), "Exactly one val_set should match a feature value"
    return matching_val_set
