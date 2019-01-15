from nose_focus import focus  # enable with @focus
import unittest

from core.valset import (
    _create_val_set_set_set_numerical,
    _create_val_set_set_set_categorical,
)

from loguru import logger

# logger.disable("core")  # Toggle to enable/disable logging in core module


@focus
class TestValset(unittest.TestCase):
    def setUp(self):
        pass

    def test_create_val_set_set_set_numerical_with_no_missing_values(self):
        numerical_set = set([1, 2])
        actual = _create_val_set_set_set_numerical(numerical_set)
        expected = frozenset(
            {
                frozenset({(float("-inf"), 1), (1, float("inf"))}),
                frozenset({(float("-inf"), 2), (2, float("inf"))}),
            }
        )
        self.assertEqual(expected, actual)

    def test_create_val_set_set_set_numerical_with_missing_values(self):
        numerical_set = set([1, 2, None, float("nan")])
        actual = _create_val_set_set_set_numerical(numerical_set)
        expected = frozenset(
            {
                frozenset(
                    {frozenset({None}), (float("-inf"), 1), (1, float("inf"))}
                ),
                frozenset(
                    {(float("-inf"), 2), frozenset({None}), (2, float("inf"))}
                ),
            }
        )
        self.assertEqual(expected, actual)

    def test_create_val_set_set_set_categorical_with_no_missing_values(self):
        categorical_set = set(["a", "b", "c", "d"])
        actual = _create_val_set_set_set_categorical(categorical_set)
        expected = frozenset(
            {
                frozenset({frozenset({"a"}), frozenset({"c", "d", "b"})}),
                frozenset({frozenset({"d"}), frozenset({"c", "b", "a"})}),
                frozenset({frozenset({"b"}), frozenset({"c", "d", "a"})}),
                frozenset({frozenset({"c"}), frozenset({"d", "b", "a"})}),
            }
        )
        self.assertEqual(expected, actual)

    def test_create_val_set_set_set_categorical_with_missing_values(self):
        categorical_set = set(["a", "b", "c", "d", None, float("nan")])
        actual = _create_val_set_set_set_categorical(categorical_set)
        expected = frozenset(
            {
                frozenset(
                    {frozenset({"b", "a", None, "d"}), frozenset({"c"})}
                ),
                frozenset(
                    {frozenset({"b", "c", None, "d"}), frozenset({"a"})}
                ),
                frozenset(
                    {frozenset({"b", "a", "c", "d"}), frozenset({None})}
                ),
                frozenset(
                    {frozenset({"b", "a", "c", None}), frozenset({"d"})}
                ),
                frozenset(
                    {frozenset({"b"}), frozenset({"a", "c", None, "d"})}
                ),
            }
        )
        self.assertEqual(expected, actual)
