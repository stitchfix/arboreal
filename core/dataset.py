from collections import Counter
from core.datatype import Datatype
from core.valset import val_set_contains
from core.abstract_dataset import AbstractDataset, AbstractMetadata


class Metadata(AbstractMetadata):
    def __init__(self):
        self._datatype = dict()
        self._target_name = None
        self._identifier = None

    # Data type metadata
    def set_datatype(self, feature_name, datatype):
        self._datatype[feature_name] = datatype

    @property
    def categoricals(self):
        return [
            f
            for f in self._datatype
            if self._datatype[f] == Datatype.categorical
        ]

    @categoricals.setter
    def categoricals(self, feature_names):
        for f in feature_names:
            self.set_datatype(f, Datatype.categorical)

    @property
    def numericals(self):
        return [
            f
            for f in self._datatype
            if self._datatype[f] == Datatype.numerical
        ]

    @numericals.setter
    def numericals(self, feature_names):
        for f in feature_names:
            self.set_datatype(f, Datatype.numerical)

    def is_numerical(self, feature):
        return self._datatype[feature] == Datatype.numerical

    def is_categorical(self, feature):
        return self._datatype[feature] == Datatype.categorical

    def datatype(self, feature):
        return self._datatype[feature]

    # Target / feature metadata
    @property
    def target(self):
        return self._target_name

    @target.setter
    def target(self, val):
        self._target_name = val

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, val):
        self._identifier = val

    @property
    def feature_names(self):
        s = set(self._datatype.keys())

        if self.target:
            s.discard(self.target)
        if self.identifier:
            s.discard(self.identifier)

        return s


class Dataset(AbstractDataset):
    def __init__(
        self,
        metadata,
        datapoints=None,
        identifier_visibility=None,
        validate_identifier=True,
        validate_target=True,
        validate_features=True,
    ):

        self._metadata = metadata
        self._datapoints = list()

        if identifier_visibility:
            self._identifier_visibility = identifier_visibility
        else:
            self._identifier_visibility = Counter()

        if datapoints:
            self.add_datapoints(
                datapoints,
                validate_identifier=validate_identifier,
                validate_target=validate_target,
                validate_features=validate_features,
            )

    def __str__(self):
        return f"<Dataset with {len(self.datapoints)} datapoints: {self.identifiers}>"

    def sub_Dataset(self, identifier_visibility):
        # return a new Dataset instance which is a 'child' or subset of this
        # instance, with different identifier visibility. does NOT deepcopy
        # other data structures; for efficiency, we just wire them up
        # to this (parent) instance's attributes

        # don't hand the datapoints to the Dataset constructor directly
        # in order to avoid both duplicate additions to internal _datapoints
        # as well as duplicate validation; instead, create the new Dataset
        # and overwrite the _datapoints field.  #TODO consider refactoring the
        # Dataset constructor so this post-construction override isn't
        # necessary
        new_Dataset = Dataset(
            metadata=self.metadata,
            datapoints=None,  # override after construction
            identifier_visibility=identifier_visibility,
        )

        new_Dataset._datapoints = self._datapoints  # the override
        return new_Dataset

    @property
    def metadata(self):
        return self._metadata

    @property
    def datapoints(self):
        # all the datapoints 'visible' in this dataset, based on
        # identifier visibility; some datapoints may not be visible at all,
        # while others may be visible multiple times (as in the case of
        # bootstrapping)
        dps = []
        for dp in self._datapoints:
            for _ in range(
                self._identifier_visibility[dp[self.metadata.identifier]]
            ):
                dps.append(dp)
        return dps

    @property
    def datapoints_count(self):
        return len(self.datapoints)  # can be optimized

    @property
    def targets(self):
        return Counter([dp[self.metadata.target] for dp in self.datapoints])

    @property
    def identifiers(self):
        return list(self._identifier_visibility.elements())

    @property
    def targets_reduced(self, verbose=False):
        # used for prediction; 'reduce' the targets down to one single class or number. implemented in dataset because it can be potentially optimized depending on dataset backend to avoid moving data

        # if numeric, store the count and mean of the targets
        if self.metadata.is_numerical(self.metadata.target):
            mean = (
                sum([dp[self.metadata.target] for dp in self.datapoints])
                / self.datapoints_count
            )  # TODO: optimize
            ret = mean

        # if categorical, store the count and each target sub-count
        elif self.metadata.is_categorical(self.metadata.target):
            class_counter = self.targets
            ret = class_counter

        if verbose:
            print(f"targets_reduced: {ret}")

        return ret

    def add_datapoints(
        self,
        datapoints,
        validate_identifier=True,
        validate_target=True,
        validate_features=True,
    ):

        for dp in datapoints:
            self.add_datapoint(
                dp,
                validate_identifier=validate_identifier,
                validate_target=validate_target,
                validate_features=validate_features,
            )

    def add_datapoint(
        self,
        datapoint,
        validate_identifier=True,
        validate_target=True,
        validate_features=True,
    ):

        if validate_identifier:
            # verify the datapoint has an identifier
            if self.metadata.identifier not in datapoint:
                raise ValueError(
                    f"""Cannot add Datapoint without identifier"""
                    f"""{self.metadata.identifier} to dataset"""
                )

        if validate_target:
            # verify the datapoint has a target value
            if self.metadata.target not in datapoint:
                raise ValueError(
                    f"""Cannot add Datapoint without target """
                    f"""{self.metadata.target} to dataset"""
                )

        if validate_features:
            # verify each key present in datapoint is known in metadata
            # (it's ok if not all known-in-metadata features are present
            # in datapoint, which happens when the feature is missing/none)
            known_keys = self.metadata.feature_names.union(
                [self.metadata.target, self.metadata.identifier]
            )

            for key in datapoint:
                if key not in known_keys:
                    raise ValueError(
                        f"""Cannot add Datapoint with unknown-to-metadata key """
                        f"""{key} to dataset"""
                    )

        # once validated, add datapoint to dataset
        self._datapoints.append(datapoint)
        # and update visibility of this datapoint's identifier
        self._identifier_visibility[datapoint[self.metadata.identifier]] += 1

    def unique_values_of_feature(self, feature):
        # get unique non-None values of this feature in the dataset
        values = set()
        for dp in self.datapoints:
            values.add(dp.get(feature))
        values.discard(None)  # we account for None elsewhere
        return values

    def val_set_targets(self, feature_name, val_set):
        targets = Counter()
        for dp in self.datapoints:
            dp_feature_value = dp.get(feature_name)
            dp_target = dp[self.metadata.target]

            if val_set_contains(val_set, dp_feature_value):
                targets[dp_target] += 1

        return targets

    @staticmethod
    def from_pandas_Xy(X, y):
        # TODO this is a VERY unoptimized way to create a Dataset from
        # pandas dataframe+series. This can and should be optimized in the future.

        # join together X and y into one DataFrame (via copying, very inefficient #TODO)
        Xy = X.join(y)

        # add an explicit identifier column
        identifier_literal = "identifier"
        Xy[identifier_literal] = range(1, len(Xy) + 1)

        # use DataFrame's types to determine handling as categorical or numeric
        categorical_feature_names = [
            c for c in Xy.select_dtypes(exclude="number").columns
        ]
        numerical_feature_names = [
            c for c in Xy.select_dtypes(include="number").columns
        ]

        # create the Metadata object by inspecting the DataFrame's column types
        m = Metadata()
        m.identifier = identifier_literal
        m.numericals = numerical_feature_names
        m.categoricals = categorical_feature_names
        m.target = y.name

        # wrap this up in a Dataset
        dataset = Dataset(metadata=m, datapoints=Xy.to_dict(orient="records"))
        return dataset

    @staticmethod
    def from_pandas_X(X, metadata):
        # add an explicit identifier column
        identifier_literal = "identifier"
        X = X.copy(deep=True)
        X[identifier_literal] = range(1, len(X) + 1)
        dataset = Dataset(
            metadata=metadata,
            datapoints=X.to_dict(orient="records"),
            validate_identifier=True,
            validate_target=False,
            validate_features=False,
        )
        return dataset
