from core.tree import Tree
from core.split import Split
from core.node import Node
from core.valset import val_set_contains
from core.datatype import Datatype
from collections import Counter
from core.bootstrap import bootstrap
from core.dataset import Dataset
from loguru import logger

# logger.disable("arboreal_tree")

import math
import inspect

import pandas as pd
import numpy as np

NODE_DATASET_KEY = "dataset"
NODE_SPLIT_KEY = "split"
NODE_VAL_SET_KEY = "val_set"
NODE_TARGETS_REDUCED_KEY = "targets_reduced"
NODE_TARGETS_COUNT_KEY = "targets_count"
NODE_TARGETS_REDUCED_TYPE_KEY = "targets_reduced_type"
NODE_IS_BOOTSTRAP_CANDIDATE_KEY = "is_bootstrap_candidate"

ESTIMATOR_TYPE_CLASSIFIER = "classifier"
ESTIMATOR_TYPE_REGRESSOR = "regressor"


def _reduce_node(node):
    # 'reduce' a node by storing the reduced targets of the dataset,
    # which becomes the prediction that this leaf node makes
    dataset = node.data[NODE_DATASET_KEY]

    node.data[NODE_TARGETS_REDUCED_KEY] = dataset.targets_reduced
    node.data[NODE_TARGETS_COUNT_KEY] = dataset.datapoints_count
    node.data[NODE_TARGETS_REDUCED_TYPE_KEY] = dataset.metadata.datatype(
        dataset.metadata.target
    )
    # TODO: remove dataset from storage of this node, and store instead the
    # datapoints_count of dataset for use in weighting this node's predictions;
    # ...unless, that is, we need to keep around the datapoint ids for
    # use in causal trees...
    return


def _fit_recursive(
    node,
    min_samples_split,
    ceiling_criterion,
    bootstrap_criterion_fraction_threshold,
    maximum_bootstrap_branching_factor,
    feature_subset_fraction,
    random_seed,
    verbose=False,
):

    logger.debug("arboreal_tree _fit_recursive() called")

    if verbose:
        print(f"Beginning _fit_recursive with node {node}")

    # receive a node that has a dataset in its data field,
    # fit a split to the dataset at the node,
    # generate children nodes from the split,
    # and update the children's data fields with the dataset
    # resulting from their split branch;
    # then call recursively on each child, ending on one
    # of 3 conditions:
    # 1) there are fewer than min_samples_split datapoints
    # left at a node (min_samples_split is 'the minimum number
    # of datapoints required to generate a split')
    # 2) there is no (information) gain to be had
    # by splitting because all the targets of the
    # present dataset are now the same
    # 3) there is no (information) gain to be had
    # by splitting because no splits improve the criterion.
    # when we choose to end, we 'reduce' up the current
    # node by storing the reduced targets of the dataset
    # which becomes the prediction that this leaf node
    # makes

    dataset = node.data[NODE_DATASET_KEY]

    # base case 1, there are fewer than min_samples_split:
    # (currently this is the supported stopping logic)
    current_datapoints_count = dataset.datapoints_count
    if current_datapoints_count < min_samples_split:
        _reduce_node(node)
        return

    # base case 2, all targets are the same
    # (this is always a stopping logic piece)
    if len(dataset.targets) == 1:  # remember, .targets is a counter
        _reduce_node(node)
        return

    # to know whether we're at base case 3, we must try to fit a split,
    # which if we're not at base case 3 we will use to recurse,
    # so the next necessary step is fitting a split
    s = Split(
        feature_subset_fraction=feature_subset_fraction,
        random_seed=random_seed,
    )
    s.fit(dataset)
    node.data[NODE_SPLIT_KEY] = s

    # base case 3, no (information) gain to be had
    if not s.is_fit:
        _reduce_node(node)
        return

    if verbose:
        print(f"Fit split {s} and saved to node")

    # if we've gotten this far, no base case applies and we are in recursive
    # case, and we have a split fit already

    # now, we have the option to either split to generate children, or
    # not split and bootstrap to generate children from the current dataset

    # only consider bootstrapping if parent node is not bootstrapped
    # (must never have two bootstrap nodes in direct parent-child relationship)
    bootstrapped = (
        False
    )  # TODO: refactor code branch logic so flag not necessary
    if node.data[NODE_IS_BOOTSTRAP_CANDIDATE_KEY]:
        if verbose:
            print(
                f"Node is a bootstrap candidate; evaluating bootstrap criteria"
            )

        # get the criterion of the dataset at the current node, which is
        # calculated and saved for us by the split that was fit, as well
        # as the ceiling/maximum it could be, which is set for us by the
        # tree at initialization and passed down through the recursion calls
        current_criterion = s.pre_split_criterion
        criterion_fraction = current_criterion / ceiling_criterion
        if criterion_fraction > bootstrap_criterion_fraction_threshold:
            # we are still above our criterion threshold to stop
            # bootstrapping, so we will bootstrap

            number_of_times_to_bootstrap = math.ceil(
                criterion_fraction * maximum_bootstrap_branching_factor
            )

            if verbose:
                print(f"Bootstrapping {number_of_times_to_bootstrap} times")

            bootstraps = []
            for _ in range(number_of_times_to_bootstrap):
                id_list = bootstrap(
                    dataset.identifiers, random_seed=random_seed
                )
                bootstraps.append(id_list)

            # Create bootstrapped children
            for b in bootstraps:

                child_identifier_visibility = Counter(b)
                child_Dataset = dataset.sub_Dataset(
                    identifier_visibility=child_identifier_visibility
                )

                child = ArborealTreeNode()
                child.data[NODE_DATASET_KEY] = child_Dataset
                child.data[NODE_IS_BOOTSTRAP_CANDIDATE_KEY] = False
                node.add_child(child)

            bootstrapped = True

    if (
        not bootstrapped
    ):  # node is not a bootstrap candidate, make children from split
        if verbose:
            print(f"Node is not a bootstrap candidate; ")
        # use the fit split to transform the dataset
        # to generate children nodes from the split,
        # and update the children's data fields with the results of the transformation, wrapped as new Datasets via sub_Dataset
        val_set_to_ids = s.transform(dataset)

        if verbose:
            print(f"Splitting dataset with val_set_to_ids: {val_set_to_ids}")

        for val_set, ids in val_set_to_ids.items():

            child_identifier_visibility = Counter(ids)
            child_Dataset = dataset.sub_Dataset(
                identifier_visibility=child_identifier_visibility
            )

            child = ArborealTreeNode()
            child.data[NODE_DATASET_KEY] = child_Dataset
            child.data[NODE_VAL_SET_KEY] = val_set
            child.data[NODE_IS_BOOTSTRAP_CANDIDATE_KEY] = True
            node.add_child(child)

    if verbose:
        print(f"Children created, node.children: {node.children}")

    # now that children have been created and their data fields populated
    # with the appropriate sub_Datasets, we recursively call fit on them
    if verbose:
        print(f"Recursing on children")

    # import pdb

    # pdb.set_trace()

    for child in node.children.values():

        _fit_recursive(
            child,
            min_samples_split,
            ceiling_criterion,
            bootstrap_criterion_fraction_threshold,
            maximum_bootstrap_branching_factor,
            feature_subset_fraction,
            random_seed,
        )
    return


def _transform_recursive(node, datapoint):

    # base case, the current node is a leaf node
    if len(node.children) == 0:

        targets_reduced = node.data[NODE_TARGETS_REDUCED_KEY]
        datapoints_count = node.data[NODE_TARGETS_COUNT_KEY]
        prediction_datatype = node.data[NODE_TARGETS_REDUCED_TYPE_KEY]

        return targets_reduced, datapoints_count, prediction_datatype

    # get the datapoint's value for the node's fit feature
    feature_name = node.data[NODE_SPLIT_KEY].feature_name
    dp_feature_value = datapoint.get(feature_name)

    # if _any_ child has a val_set that this dp_feature_value belongs to, then
    # use only that child's response. otherwise, ask all children for their
    # predictions, and aggregate their responses according to the children's
    # weight (the number of data points they have)

    children_predictions_and_weights = []
    prediction_datatypes = set()
    for child in node.children.values():
        targets_reduced, datapoints_count, prediction_datatype = _transform_recursive(
            child, datapoint
        )

        child_val_set = child.data.get(NODE_VAL_SET_KEY)
        if child_val_set and val_set_contains(child_val_set, dp_feature_value):
            # this child has a matching valset for this datapoint; this is the
            # only child response we need; return early
            return targets_reduced, datapoints_count, prediction_datatype
        else:
            # validate all child prediction datatypes are consistent
            prediction_datatypes.add(prediction_datatype)
            assert (
                len(prediction_datatypes) == 1
            ), "All prediction datatypes from subtrees must be equal"

            children_predictions_and_weights.append(
                (targets_reduced, datapoints_count)
            )

    # reduce the children's responses
    total_datapoints_count = sum(
        [c[1] for c in children_predictions_and_weights]
    )
    prediction_datatype = prediction_datatypes.pop()
    if prediction_datatype == Datatype.numerical:
        # if datatype is numerical, get a weighted sum of children's
        # predictions
        targets_reduced = sum(
            [c[0] * c[1] for c in children_predictions_and_weights]
        )
        targets_reduced /= total_datapoints_count
    elif prediction_datatype == Datatype.categorical:
        # if datatype is categorical, add the counters of the children's
        # predictions
        targets_reduced = Counter()
        for c in children_predictions_and_weights:
            prediction, weight = c
            targets_reduced += prediction  # add counters
        assert total_datapoints_count == sum(
            targets_reduced.values()
        ), "aggregated counters should maintain consistent total weight"
    else:
        raise ValueError("Unrecognized prediction datatype")

    return targets_reduced, total_datapoints_count, prediction_datatype


class ArborealTreeNode(Node):
    def _is_bootstrapped_child(self):
        return not bool(self.data.get(NODE_VAL_SET_KEY))

    def __str__(self):
        if self.is_leaf:
            s = f"<Leaf Node with "
            s += f"val_set {self.data.get(NODE_VAL_SET_KEY)} and "
            s += f"targets_reduced {self.data.get(NODE_TARGETS_REDUCED_KEY)} and "
            s += f"targets_reduced_type {self.data.get(NODE_TARGETS_REDUCED_TYPE_KEY)} and "
            s += f"dataset size {self.data.get(NODE_DATASET_KEY).datapoints_count}"
            s += ">"
        else:
            if self._is_bootstrapped_child():
                if self.parent:
                    s = f"<Bootstrap Child Node with "
                else:
                    s = f"<Root Node with "
                s += f"dataset size {self.data.get(NODE_DATASET_KEY).datapoints_count}"
                s += ">"
            else:
                s = f"<Split Child Node with "
                s += f"val_set {self.data.get(NODE_VAL_SET_KEY)} and "
                s += f"dataset size {self.data.get(NODE_DATASET_KEY).datapoints_count}"
                s += ">"

        return s


class ArborealTree(Tree):
    def __init__(
        self,
        bootstrap_criterion_fraction_threshold=0.8,
        maximum_bootstrap_branching_factor=10,
        feature_subset_fraction=None,
        min_samples_split=2,
        random_seed=None,
        disable_logging_parallel=False,
    ):

        # set all client/caller parameters (non-'self') via set_params
        # (we use set_params to ensure that init and set_params both follow the
        # same codepath for setting parameters)
        loc = inspect.getargvalues(inspect.currentframe()).locals

        # note that the locals includes 'self', so calling something like
        # self.set_params(**loc) would require us to first remove 'self' from
        # 'loc' with eg loc.pop("self"); alternatively, we can avoid the
        # self.function prefix notation and pass self directly inside loc:
        ArborealTree.set_params(**loc)

    def set_params(self, **parameters):
        # before sklearn interface, super call was in init:
        # create a Tree (creates root Node, etc)
        super(ArborealTree, self).__init__()

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        if self.disable_logging_parallel:
            logger.debug("Disabling logging in core")
            logger.disable("core")

        return self

    def get_params(self, deep=True):
        # private vars are named with _varname convention,
        # vars set in fit() are named with varname_ convention,
        # and vars set by client are named with varname convention;
        # therefore, to return client-set vars as per get_params spec,
        # we return vars neither prepended nor appended with _
        client_var_names = sorted(
            [
                n
                for n in vars(self)
                if not (n.startswith("_") or n.endswith("_"))
            ]
        )
        client_set_parameters = {n: vars(self)[n] for n in client_var_names}

        return client_set_parameters

    def fit(self, *args):

        logger.debug("ArborealTree fit() called")

        # to be compatible with sklearn interface, fit() must accept an (X, y)
        # function call signature. to be nice with Arboreal Datasets, we also
        # want to accept a (Dataset) signature. this fit() function is mostly
        # about handling incoming data of different types and preparing it for
        # the _fit() function which does the actual fitting work on a Dataset.
        # determine whether we're using a scikit-learn or Arboreal signature:
        if (
            len(args) == 2
            and isinstance(args[0], pd.DataFrame)
            and isinstance(args[1], pd.Series)
        ):
            # we are using X, y dataframe function signature
            # we'll construct a Dataset from these X, y dataframes
            dataset = Dataset.from_pandas_Xy(args[0], args[1])

        elif len(args) == 1 and isinstance(args[0], Dataset):
            # we are using Dataset function signature, good to pass directly to _fit()
            dataset = args[0]

        else:

            import pdb

            pdb.set_trace()

            # we are using an unknown signature
            raise ValueError(
                "fit() requires either X,y paired dataframes(/series) or a Dataset"
            )

        # sklearn parameters:
        # set _estimator_type for sklearn compatibility; use the target metadata
        # to determine estimator type
        if dataset.metadata.is_categorical(dataset.metadata.target):
            self._estimator_type = ESTIMATOR_TYPE_CLASSIFIER
        elif dataset.metadata.is_numerical(dataset.metadata.target):
            self._estimator_type = ESTIMATOR_TYPE_REGRESSOR
        else:
            raise ValueError(
                "target must have type set, and must be either categorical or numerical"
            )

        # set classes_ for sklearn compatibility (if classifier); also used
        # as class ordering in predict and predict_proba
        if dataset.metadata.is_categorical(dataset.metadata.target):
            self.classes_ = sorted(set(dataset.targets.keys()))

        return self._fit(dataset)

    def _fit(self, dataset):

        logger.debug("ArborealTree _fit() called")

        # fit this decision tree to the dataset
        # recursive fit pseudo:
        # init:
        # take the current node (start w root)
        # save the current dataset to this node's data field
        # recursion:
        # fit a split to the cur dataset at cur node (via node's data field)
        # save the split to the current node's data field (alongside dataset)
        # create one child node for each group from the split
        # save to each child node the dataset that represents that group
        # call the recursive fit on each of the child nodes

        # init:
        self.root = ArborealTreeNode()
        self.root_ = self.root  # make sklearn happy with a root_ fitted param
        # get the current node
        current_node = self.root
        current_node.name = "root"
        # save the current dataset to this node's data field
        current_node.data[NODE_DATASET_KEY] = dataset
        # the root node is a bootstrap candidate
        current_node.data[NODE_IS_BOOTSTRAP_CANDIDATE_KEY] = True
        # set the ceiling criterion to the (parent/root)
        # criterion before any splits are generated; do this by fitting a split
        # (without saving), and accessing its pre_split_criterion
        s = Split(
            feature_subset_fraction=1.0, random_seed=self.random_seed
        )  # use full feature fit to set ceiling; random_seed is essentially not used here, as we are not randomly selecting a subset (given we're using the full feature_set)
        s.fit(dataset)
        self.ceiling_criterion_ = s.pre_split_criterion

        logger.debug("Initial Split fit")

        # auto-set feature_subset_fraction if necessary
        if not self.feature_subset_fraction:
            num_features = len(dataset.metadata.feature_names)
            default_fraction = math.sqrt(num_features)
            self.feature_subset_fraction = default_fraction

        logger.debug("Beginning recursive fitting")

        # fit recursively:
        _fit_recursive(
            current_node,
            self.min_samples_split,
            self.ceiling_criterion_,
            self.bootstrap_criterion_fraction_threshold,
            self.maximum_bootstrap_branching_factor,
            self.feature_subset_fraction,
            self.random_seed,
        )

    def predict(self, *args):
        assert len(args) == 1

        if isinstance(args[0], pd.DataFrame):
            X = args[0]
            m = self.root.data[NODE_DATASET_KEY].metadata
            dataset = Dataset.from_pandas_X(X, m)
        elif isinstance(args[0], Dataset):
            dataset = args[0]
        else:
            raise ValueError(
                "Arg to predict() must be a Pandas DataFrame or Arboreal Dataset"
            )

        # return predictions in the format sklearn expects
        predictions = []
        for transformed in self.transform(dataset):
            targets_reduced, total_datapoints_count, prediction_datatype = (
                transformed
            )
            if prediction_datatype == Datatype.numerical:
                prediction = targets_reduced
            elif prediction_datatype == Datatype.categorical:
                prediction = targets_reduced.most_common()[0][0]
            predictions.append(prediction)

        return predictions

    def predict_proba(self, *args):
        # predict_proba conforms to the sklearn estimator interface

        # predict_proba is only available for classifiers
        m = self.root.data[NODE_DATASET_KEY].metadata
        assert m.is_categorical(m.target)
        assert self._estimator_type == ESTIMATOR_TYPE_CLASSIFIER

        # if being called with a dataframe so we convert the dataframe to a
        # Dataset before calling transform on it
        assert len(args) == 1 and isinstance(args[0], pd.DataFrame)
        X = args[0]
        m = self.root.data[NODE_DATASET_KEY].metadata
        dataset = Dataset.from_pandas_X(X, m)

        # gather predictions and probabilities in the format sklearn expects
        all_datapoints_class_probabilities = []
        for transformed in self.transform(dataset):
            targets_reduced, total_datapoints_count, prediction_datatype = (
                transformed
            )
            # for each class in order, get the proba
            class_probabilities = []
            for klass in self.classes_:
                fraction_of_class = targets_reduced[klass] / sum(
                    targets_reduced.values()
                )
                class_probabilities.append(fraction_of_class)
            all_datapoints_class_probabilities.append(class_probabilities)

        return np.array(all_datapoints_class_probabilities)

    def transform(self, *args):
        # This function aims to be compatible with both arboreal Datasets
        # directly as well as Pandas dataframes, to implement the sklearn
        # interface.  If a DataFrame is passed in, we create a Dataset, and
        # then continue handling as usual.  Note, if we need to create a
        # dataset, rather than do inference on the types of the passed-in
        # dataframe, we use the metadata determined in fit() (called before
        # transform()) to determine which columns are of which types.
        if len(args) == 1 and isinstance(args[0], Dataset):
            dataset = args[0]
        elif len(args) == 1 and isinstance(args[0], pd.DataFrame):
            X = args[0]
            # use the fit metadata rather than re-inferring (and potentially
            # being inconsistent with the previous inference)
            m = self.root.data[NODE_DATASET_KEY].metadata
            dataset = Dataset.from_pandas_X(X, m)
        else:
            raise ValueError(
                "transform requires either an Arboreal Dataset or Pandas DataFrame"
            )

        return self._transform(dataset)

    def _transform(self, dataset):
        predictions = []
        for dp in dataset.datapoints:
            predictions.append(self._transform_recursive_wrapper(dp))
        return predictions

    def _transform_recursive_wrapper(self, datapoint):
        start_node = self.root
        return _transform_recursive(start_node, datapoint)


class DecisionTree(ArborealTree):
    # a special case of ArborealTree that is equivalent to a traditional
    # Decision Tree, with by default minimum samples per split required
    # to split as 2

    def __init__(
        self,
        bootstrap_criterion_fraction_threshold=1.0,
        maximum_bootstrap_branching_factor=1,
        feature_subset_fraction=1.0,
        min_samples_split=2,
        random_seed=None,
        disable_logging_parallel=False,
    ):

        # initialize superclass with passed in (/default) parameters
        loc = inspect.getargvalues(inspect.currentframe()).locals
        loc.pop("self")
        loc.pop("__class__")
        super(DecisionTree, self).__init__(**loc)

    def score(self, X, y):
        predictions = self.predict(X)
        compare = list(zip(predictions, y))
        accuracy = len([r for r in compare if r[0] == r[1]]) / len(compare)
        return accuracy


class RandomForest(ArborealTree):
    # a special case of ArborealTree that is equivalent to a traditional
    # Random Forest, with by default 100 trees, sqrt(num_features) as the
    # feature subset amongst which features are considered for splits,
    # and minimum samples per split required to split as 2

    def __init__(
        self,
        bootstrap_criterion_fraction_threshold=0.999_999_999,
        maximum_bootstrap_branching_factor=100,
        feature_subset_fraction=None,
        min_samples_split=2,
        random_seed=None,
        disable_logging_parallel=False,
    ):

        logger.debug("RandomForest __init__() called")

        # initialize superclass with passed in (/default) parameters
        loc = inspect.getargvalues(inspect.currentframe()).locals
        loc.pop("self")
        loc.pop("__class__")
        super(RandomForest, self).__init__(**loc)

    def score(self, X, y):
        predictions = self.predict(X)
        compare = list(zip(predictions, y))
        accuracy = len([r for r in compare if r[0] == r[1]]) / len(compare)
        return accuracy
