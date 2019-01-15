from core.abstract_dataset import AbstractDataset
from core.datatype import Datatype


IDENTIFIER_LITERAL = "identifier_literal"


class DictBackendMetadata(AbstractMetadata):
    def __init__(self):
        self._datatype = dict()
        self._target_name = None
        self._identifier = None

    def set_datatype(self, feature_name, datatype):
        self._datatype[feature_name] = datatype

    def datatype(self, feature_name):
        return self._datatype[feature]

    def is_numerical(self, feature_name):
        return self._datatype[feature] == Datatype.numerical

    def is_categorical(self, feature_name):
        return self._datatype[feature] == Datatype.categorical

    @property
    def feature_names(self):
        s = set(self._datatype.keys())

        if self.target:
            s.discard(self.target)
        if self.identifier:
            s.discard(self.identifier)

        return s

    @property
    def categoricals(self):
        return [f for f in self._datatype if self.is_categorical(f)]

    @categoricals.setter
    def categoricals(self, feature_names):
        for f in feature_names:
            self.set_datatype(f, Datatype.categorical)

    @property
    def numericals(self):
        return [f for f in self._datatype if self.is_numerical(f)]

    @numericals.setter
    def numericals(self, feature_names):
        for f in feature_names:
            self.set_datatype(f, Datatype.numerical)

    @property
    def target(self):
        return self._target_name

    @target.setter
    def target(self, target_name):
        self._target_name = target_name

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier_name):
        self._identifier = identifier_name

    @classmethod
    def from_pandas_Xy(cls, X, y):
        # construct a Metadata object from explicitly typed-via-column-name
        # pandas dataframes
        pass


class PandasBackendDataset(AbstractDataset):
    def __init__(self, metadata, df, identifier_visibility=None):
        self._metadata = metadata
        self._df = df

        if identifier_visibility:
            self._identifier_visibility = identifier_visibility
        else:
            self._identifier_visibility = Counter()

    @property
    def metadata(self):
        return self._metadata

    @property
    def datapoints(self):
        pass

    @property
    def datapoints_count(self):
        return len(self._df)

    @property
    def identifiers(self):
        return list(self._identifier_visibility.elements())

    @property
    def targets(self):
        pass

    @property
    def targets_reduced(self):
        pass

    def unique_values_of_feature(self, feature):
        pass

    def val_set_targets(self, feature_name, val_set):
        pass

    def sub_Dataset(self, identifier_visibility):
        new_Dataset = PandasBackendDataset(
            metadata=self.metadata,
            df=self.df,
            identifier_visibility=identifier_visibility,
        )
        return new_Dataset

    @classmethod
    def from_pandas_Xy(cls, X, y):

        # validate X and y are of the same length
        assert len(X) == len(y), "Length of X and y must match"

        # add an explicit identifier column (used for masking in eg
        # bootstrapping contexts)
        X = X.copy(deep=True)
        X[IDENTIFIER_LITERAL] = range(1, len(X) + 1)

        # join X and y into one df
        df = X.join(y)

        # use DataFrame's column dtypes to determine handling as categorical or
        # numeric for both features and target
        categoricals = [c for c in df.select_dtypes(exclude="number").columns]
        numericals = [c for c in df.select_dtypes(include="number").columns]

        # construct a Metadata object
        m = DictBackendMetadata()
        m.identifier = IDENTIFIER_LITERAL
        m.numericals = numericals
        m.categoricals = categoricals
        m.target = y.name

        return cls(metadata=m, df=df)

    @classmethod
    def from_pandas_X(X, metadata):
        # add an explicit identifier column (used for masking in eg
        # bootstrapping contexts)
        X = X.copy(deep=True)
        X[IDENTIFIER_LITERAL] = range(1, len(X) + 1)
        df = X

        return cls(metadata=metadata, df=df)
