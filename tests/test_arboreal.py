from nose_focus import focus  # enable with @focus
import unittest

from core.arboreal_tree import ArborealTree
from core.dataset import Metadata, Dataset

from loguru import logger

logger.disable("core")  # Toggle to enable/disable logging in core module


class TestArborealTree(unittest.TestCase):
    def setUp(self):
        self.default_vals = dict(
            bootstrap_criterion_fraction_threshold=0.8,
            maximum_bootstrap_branching_factor=10,
            feature_subset_fraction=None,
            min_samples_split=2,
            random_seed=None,
            disable_logging_parallel=False,
        )
        self.nondefault_vals = dict(
            bootstrap_criterion_fraction_threshold=0.9,
            maximum_bootstrap_branching_factor=11,
            feature_subset_fraction=0.5,
            min_samples_split=5,
            random_seed=42,
            disable_logging_parallel=False,
        )

    def test_arboreal_constructor(self):
        # test that the constructor receives input args and sets them
        at = ArborealTree()
        # assert at has the right number of attributes set:
        user_attributes = [
            a for a in vars(at) if not (a.startswith("_") or a.endswith("_"))
        ]
        self.assertEqual(len(user_attributes), len(self.default_vals))
        # now check equality of default values and set values:
        for arg_name, default_val in self.default_vals.items():
            self.assertEqual(getattr(at, arg_name), default_val)

    def test_arboreal_set_params(self):
        # test that the set_params function sets parameters
        at = ArborealTree()
        at.set_params(**self.nondefault_vals)
        # assert at has the right number of attributes set:
        user_attributes = [
            a for a in vars(at) if not (a.startswith("_") or a.endswith("_"))
        ]
        self.assertEqual(len(user_attributes), len(self.nondefault_vals))
        # now check equality of nondefault_vals and set values:
        for arg_name, nondefault_val in self.nondefault_vals.items():
            self.assertEqual(getattr(at, arg_name), nondefault_val)

    def test_arboreal_constructor_and_set_params_are_equal(self):
        at_1 = ArborealTree(**self.default_vals)
        at_2 = ArborealTree()
        at_2.set_params(**self.default_vals)
        self.assertEqual(at_1.get_params(), at_2.get_params())

        at_3 = ArborealTree(**self.nondefault_vals)
        at_4 = ArborealTree()
        at_4.set_params(**self.nondefault_vals)
        self.assertEqual(at_3.get_params(), at_4.get_params())

    def test_arboreal_get_params(self):
        at = ArborealTree()
        got_params = at.get_params()
        self.assertEqual(got_params, self.default_vals)

    def test_arboreal_set_then_get_params(self):
        at = ArborealTree()
        at.set_params(**self.nondefault_vals)
        got_params = at.get_params()
        self.assertEqual(got_params, self.nondefault_vals)
