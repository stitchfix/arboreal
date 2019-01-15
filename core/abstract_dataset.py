from abc import ABCMeta, abstractmethod


class AbstractMetadata(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def set_datatype(self, feature_name, datatype):
        pass

    @abstractmethod
    def datatype(self, feature_name):
        pass

    @abstractmethod
    def is_numerical(self, feature_name):
        pass

    @abstractmethod
    def is_categorical(self, feature_name):
        pass

    @property
    @abstractmethod
    def feature_names(self):
        pass

    @property
    @abstractmethod
    def categoricals(self):
        pass

    @categoricals.setter
    @abstractmethod
    def categoricals(self, feature_names):
        pass

    @property
    @abstractmethod
    def numericals(self):
        pass

    @numericals.setter
    @abstractmethod
    def numericals(self, feature_names):
        pass

    @property
    @abstractmethod
    def target(self):
        pass

    @target.setter
    @abstractmethod
    def target(self, target_name):
        pass

    @property
    @abstractmethod
    def identifier(self):
        pass

    @identifier.setter
    @abstractmethod
    def identifier(self, target_name):
        pass


class AbstractDataset(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args):
        pass

    @property
    @abstractmethod
    def metadata(self):
        pass

    @property
    @abstractmethod
    def datapoints(self):
        pass

    @property
    @abstractmethod
    def datapoints_count(self):
        return len(self.datapoints)  # can be optimized

    @property
    @abstractmethod
    def identifiers(self):
        pass

    @property
    @abstractmethod
    def targets(self):
        pass

    @property
    @abstractmethod
    def targets_reduced(self):
        pass

    @abstractmethod
    def unique_values_of_feature(self, feature):
        pass

    @abstractmethod
    def val_set_targets(self, feature_name, val_set):
        pass

    @abstractmethod
    def sub_Dataset(self, identifier_visibility):
        pass
