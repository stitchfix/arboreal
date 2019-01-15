from nose_focus import focus  # enable with @focus
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import unittest

from core.arboreal_tree import ArborealTree, DecisionTree, RandomForest

from loguru import logger

logger.disable("core")  # Toggle to enable/disable logging in core module


class TestGridSearchInterface(unittest.TestCase):
    def setUp(self):
        # set random seed for consistent tests
        self.random_seed = 43

        # set n_jobs to 1 for easiest debugging, -1 for faster test suite runs
        # (but this may enable more logs since logger.disable() only applies to
        # one process :/
        self.n_jobs = 1

        # load iris dataset, split into train/test
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
        self.X_train, self.y_train = train[feature_names], train[target_name]
        self.X_test, self.y_test = test[feature_names], test[target_name]

    def test_grid_search_interface_for_decision_tree_min_samples_split(self):
        param_grid = {"min_samples_split": [2, 3, 4]}
        tree = DecisionTree()
        gs = GridSearchCV(
            tree, param_grid, cv=3, iid=True, verbose=0, n_jobs=self.n_jobs
        )  # verbose = 2 for helpful info during the search

        gs.fit(self.X_train, self.y_train)
        gs_accuracy = gs.score(self.X_test, self.y_test)

        expected_accuracy = 0.90
        self.assertAlmostEqual(gs_accuracy, expected_accuracy, places=2)

    def test_grid_search_interface_for_random_forest(self):
        param_grid = {
            "maximum_bootstrap_branching_factor": [1, 5, 10],
            "feature_subset_fraction": [None, 0.4, 0.6],
            "min_samples_split": [2, 3],
        }
        arb = RandomForest(random_seed=self.random_seed)
        gs = GridSearchCV(
            arb,
            param_grid,
            cv=3,
            scoring="accuracy",
            iid=True,
            verbose=0,
            n_jobs=self.n_jobs,
        )  # verbose = 2 for helpful info during the search

        gs.fit(self.X_train, self.y_train)
        gs_accuracy = gs.score(self.X_test, self.y_test)

        expected_accuracy = 0.90
        self.assertAlmostEqual(gs_accuracy, expected_accuracy, places=2)

    def test_grid_search_interface_for_arboreal_tree(self):
        param_grid = {
            "bootstrap_criterion_fraction_threshold": [0.8, 0.999_999_999],
            "maximum_bootstrap_branching_factor": [1, 2, 3],
            "feature_subset_fraction": [None, 0.4],
            "min_samples_split": [2, 3],
        }
        arb = ArborealTree(random_seed=self.random_seed)
        gs = GridSearchCV(
            arb,
            param_grid,
            cv=3,
            scoring="accuracy",
            iid=True,
            verbose=0,
            n_jobs=self.n_jobs,
        )  # verbose = 2 for helpful info during the search

        gs.fit(self.X_train, self.y_train)
        gs_accuracy = gs.score(self.X_test, self.y_test)

        expected_accuracy = 0.90
        self.assertAlmostEqual(gs_accuracy, expected_accuracy, places=2)

    def test_grid_search_interface_for_arboreal_tree_single_grid_cell(self):
        param_grid = {
            "bootstrap_criterion_fraction_threshold": [0.8],
            "maximum_bootstrap_branching_factor": [2],
            "feature_subset_fraction": [0.4],
            "min_samples_split": [2],
        }
        arb = ArborealTree(random_seed=self.random_seed)
        gs = GridSearchCV(
            arb,
            param_grid,
            cv=3,
            scoring="accuracy",
            iid=True,
            verbose=0,
            n_jobs=self.n_jobs,
        )  # verbose = 2 for helpful info during the search

        gs.fit(self.X_train, self.y_train)

        gs_accuracy = gs.score(self.X_test, self.y_test)

        expected_accuracy = 0.61
        self.assertAlmostEqual(gs_accuracy, expected_accuracy, places=2)
