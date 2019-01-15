from nose_focus import focus  # enable with @focus
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import copy
import math
import numpy as np
import pandas as pd
import random
import unittest

from core.arboreal_tree import DecisionTree, RandomForest
from core.dataset import Metadata, Dataset


class TestIris(unittest.TestCase):
    def setUp(self):
        self.random_seed = 43

    def test_iris_with_arboreal_dataset(self):
        # Load iris dataset and convert to Pandas DataFrame
        iris = load_iris()
        iris_df = pd.DataFrame(
            data=np.c_[iris["data"], iris["target"]],
            columns=iris["feature_names"] + ["target"],
        )
        # Add an explicit identifier column
        iris_df["identifier"] = range(1, len(iris_df) + 1)

        # Create the Metadata for this dataset
        m = Metadata()
        m.identifier = "identifier"
        m.numericals = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "target",
        ]
        m.target = "target"

        # Create the datapoints by converting Pandas rows to dictionaries
        datapoints = iris_df.to_dict(orient="records")

        # Train/test split
        train_fraction = 0.8
        random.seed(42)  # set random seed
        random.shuffle(datapoints)  # shuffle dataset
        split_index = math.ceil(train_fraction * len(datapoints))
        train_datapoints, test_datapoints = (
            datapoints[:split_index],
            datapoints[split_index:],
        )
        # Double check we're not cheating by letting the target into the test data
        test_datapoints_with_targets = copy.deepcopy(test_datapoints)
        for dp in test_datapoints:
            del dp["target"]

        # Create train and test datasets
        train_dataset = Dataset(metadata=m, datapoints=train_datapoints)
        test_dataset = Dataset(
            metadata=m, datapoints=test_datapoints, validate_target=False
        )

        # Fit a Decision Tree on the train set
        tree = DecisionTree()
        tree.fit(train_dataset)

        # Predict data points in the test set
        predictions = tree.predict(test_dataset)

        # Compare predictions to actual (for test set)
        actual_targets = [dp["target"] for dp in test_datapoints_with_targets]
        self.assertEqual(len(predictions), len(actual_targets))
        accuracy = len(
            [x for x in zip(predictions, actual_targets) if x[0] == x[1]]
        ) / len(predictions)
        # print(f"Accuracy: {accuracy}")

        # Assert accuracy > 0.5
        self.assertTrue(accuracy > 0.5)

    def test_iris_with_pandas_dataframes(self):
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df["species"] = pd.Categorical.from_codes(
            iris.target, iris.target_names
        )
        train_fraction = 0.8
        iris_df["is_train"] = (
            np.random.uniform(0, 1, len(iris_df)) <= train_fraction
        )
        feature_names = iris_df.columns[:4]
        target_name = iris_df.columns[4]

        train, test = (
            iris_df[iris_df["is_train"] == True],
            iris_df[iris_df["is_train"] == False],
        )
        X_train, y_train = train[feature_names], train[target_name]
        X_test, y_test = test[feature_names], test[target_name]

        tree = DecisionTree()
        tree.fit(X_train, y_train)

        # get predicted targets
        y_predictions = tree.predict(X_test)

        results = [r for r in zip(y_predictions, y_test)]

        accuracy = len([r for r in results if r[0] == r[1]]) / len(results)
        # print(f"Accuracy: {accuracy}")

        self.assertTrue(accuracy > 0.5)

    def test_arboreal_and_sklearn_decision_tree_accuracy_is_equal(self):
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df["species"] = pd.Categorical.from_codes(
            iris.target, iris.target_names
        )
        train_fraction = 0.8
        # set random seed for train/test split
        np.random.seed(42)
        iris_df["is_train"] = (
            np.random.uniform(0, 1, len(iris_df)) <= train_fraction
        )
        feature_names = iris_df.columns[:4]
        target_name = iris_df.columns[4]

        train, test = (
            iris_df[iris_df["is_train"] == True],
            iris_df[iris_df["is_train"] == False],
        )
        X_train, y_train = train[feature_names], train[target_name]
        X_test, y_test = test[feature_names], test[target_name]

        sklearn_tree = DecisionTreeClassifier()
        sklearn_tree.fit(X_train, y_train)
        sklearn_accuracy = sklearn_tree.score(X_test, y_test)

        arboreal_tree = DecisionTree()
        arboreal_tree.fit(X_train, y_train)
        arboreal_accuracy = arboreal_tree.score(X_test, y_test)

        self.assertEqual(
            arboreal_accuracy,
            sklearn_accuracy,
            "arboreal and sklearn decision trees should have equal accuracy",
        )

    def test_arboreal_and_sklearn_random_forest_accuracy_is_almost_equal(self):
        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df["species"] = pd.Categorical.from_codes(
            iris.target, iris.target_names
        )
        train_fraction = 0.8
        # set random seed for train/test split
        np.random.seed(self.random_seed)
        iris_df["is_train"] = (
            np.random.uniform(0, 1, len(iris_df)) <= train_fraction
        )
        # print(f"iris_df is_train: {iris_df['is_train']}")
        feature_names = iris_df.columns[:4]
        target_name = iris_df.columns[4]

        train, test = (
            iris_df[iris_df["is_train"] == True],
            iris_df[iris_df["is_train"] == False],
        )
        X_train, y_train = train[feature_names], train[target_name]
        X_test, y_test = test[feature_names], test[target_name]

        # call w random seed/state for consistency
        sklearn_rf = RandomForestClassifier(
            n_estimators=10, random_state=self.random_seed
        )
        sklearn_rf.fit(X_train, y_train)
        sklearn_accuracy = sklearn_rf.score(X_test, y_test)
        # print(f"\nsklearn_accuracy: {sklearn_accuracy}")

        # call w random seed for consistency
        arboreal_rf = RandomForest(
            maximum_bootstrap_branching_factor=10, random_seed=self.random_seed
        )
        arboreal_rf.fit(X_train, y_train)
        arboreal_accuracy = arboreal_rf.score(X_test, y_test)
        # print(f"\narboreal_accuracy: {arboreal_accuracy}")

        self.assertAlmostEqual(
            arboreal_accuracy,
            sklearn_accuracy,
            places=8,
            msg="arboreal and sklearn random forests should have almost equal accuracy",
        )
