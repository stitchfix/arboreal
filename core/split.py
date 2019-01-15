from collections import defaultdict
from core.criterion import get_criterion
from core.valset import create_val_set_set_set, get_val_set_for_feature_value
import math
import random
from loguru import logger


class Split:
    def __init__(
        self,
        feature_subset_fraction=1.0,
        random_seed=None,
        reward_comparison_tolerance=1e-10,
    ):

        logger.debug("Split __init__() called")

        self._reward = None  # some reward, like information gain
        self._feature_name = None
        self._val_set_set = None  # val_set_set representing the fit split
        self._pre_split_criterion = None
        self._feature_subset_fraction = feature_subset_fraction
        self._random_seed = random_seed
        self._reward_comparison_tolerance = reward_comparison_tolerance

    @property
    def reward(self):
        return self._reward

    @property
    def feature_name(self):
        return self._feature_name

    @property
    def val_set_set(self):
        return self._val_set_set

    @property
    def pre_split_criterion(self):
        return self._pre_split_criterion

    @property
    def is_fit(self):
        return bool(self._reward and self._feature_name and self._val_set_set)

    def __str__(self):
        if self._reward is None:
            return f"<Split of feature {self._feature_name} (unfit)>"
        else:
            ret = f"<Split of feature {self._feature_name}"
            ret += f" with val_set_set {self._val_set_set}>"
            return ret

    def fit(self, dataset, verbose=False):

        logger.debug("Split fit() called")

        return self._fit(dataset, verbose=verbose)

    def _fit(self, dataset, verbose=False):
        # Split creates a candidate ValSetSetSet and loops through each
        # ValSetSet in it, evaluating the best ValSetSet.  A ValSetSet is, not
        # surprisingly, a set of ValSets.  A ValSet, in turn, is a set of
        # values; this set can contain all values explicitly, as in the
        # categorical case, or can contain bounding endpoints on a line, as in
        # the numerical case.

        logger.debug("Split _fit() called")

        # Track max reward of candidate splits
        max_reward = 0.0
        max_reward_feature_name = None
        max_reward_val_set_set = None

        # Get the parent criterion pre proposed split
        target_type = dataset.metadata.datatype(dataset.metadata.target)
        parent_criterion = get_criterion(dataset.targets, target_type)
        total_datapoints = dataset.datapoints_count
        logger.debug(
            f"Split fitting beginning: target_type {target_type}, parent_criterion {parent_criterion}, total_datapoints {total_datapoints}"
        )

        # Determine the candidate features to consider as a function
        # of feature_subset_fraction
        all_features = sorted(
            list(dataset.metadata.feature_names)
        )  # sort by default for consistency; if random subset selection is enabled, that's handled next
        index = math.ceil(len(all_features) * self._feature_subset_fraction)

        if self._random_seed:
            random.seed(self._random_seed)
        random.shuffle(all_features)  # random selection...
        # print(f"all_features[:1]: {all_features[:1]}")
        considered_candidate_features = all_features[:index]  # up to the frac

        # For considered candidate features in the current dataset
        for feature_name in considered_candidate_features:

            logger.debug(f"evaluating feature {feature_name}")

            # Build a set of candidate splits
            feature_datatype = dataset.metadata.datatype(feature_name)
            feature_values = dataset.unique_values_of_feature(feature_name)
            val_set_set_set = create_val_set_set_set(
                feature_values, feature_datatype
            )

            logger.debug(
                f"len(set of candidate splits): {len(val_set_set_set)}"
            )

            # For each val_set_set, which is a single candidate split, ask the
            # dataset, for each val_set, which is a single branch of a
            # candidate split, for the target (counts) of datapoints whose
            # feature value is in the val_set, which is to say branch, of the
            # proposed split
            for val_set_set in val_set_set_set:

                if verbose:
                    pass
                    # print(f"\t\tevaluating val_set_set {val_set_set}")

                # Get the criteria for each branch of the proposed split
                weighted_criteria = []
                for val_set in val_set_set:
                    targets = dataset.val_set_targets(feature_name, val_set)
                    group_size = sum(targets.values())

                    if group_size > 0:
                        criterion = get_criterion(targets, target_type)
                        weighted_criterion = criterion * group_size
                        weighted_criteria.append(weighted_criterion)

                # Sum and normalize weighted criterion across children of proposed split
                children_weighted_criteria = (
                    sum(weighted_criteria) / total_datapoints
                )

                # Calc the reward from this proposed split
                reward = parent_criterion - children_weighted_criteria

                if verbose:
                    pass
                    # print(f"\t\treward of proposed split: {reward}")

                # Record the results if this is the best reward so far
                if reward > max_reward + self._reward_comparison_tolerance:
                    max_reward = reward
                    max_reward_feature_name = feature_name
                    max_reward_val_set_set = val_set_set

                    logger.debug(
                        f"New max reward found, reward: {reward}, feature_name: {feature_name}, val_set_set: {val_set_set}"
                    )

        # Record the best fit
        self._reward = max_reward
        self._feature_name = max_reward_feature_name
        self._val_set_set = max_reward_val_set_set
        self._pre_split_criterion = parent_criterion  # and parent crit

        if self.is_fit:
            logger.debug(
                f"Split fit; chosen feature_name {self._feature_name} and chosen val_set_set {self._val_set_set}"
            )
        else:
            logger.debug(
                f"Split remains unfit; no candidate splits yielded a reward"
            )

        return

    def transform(self, dataset):
        val_set_to_ids = defaultdict(list)

        for dp in dataset.datapoints:
            dp_id = dp[dataset.metadata.identifier]
            dp_feature_value = dp.get(self.feature_name)
            val_set = get_val_set_for_feature_value(
                self._val_set_set, dp_feature_value
            )
            val_set_to_ids[val_set].append(dp_id)

        return val_set_to_ids
