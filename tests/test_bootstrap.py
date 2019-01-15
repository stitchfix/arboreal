from collections import Counter
import random
import unittest

from core.bootstrap import bootstrap


class TestBootstrap(unittest.TestCase):
    def test_bootstrap(self):
        random.seed(42)
        iterable = [1, 2, 3, 4, 5]
        b = bootstrap(iterable)
        expected = [1, 1, 2, 2, 3]
        self.assertEqual(len(b), len(expected))
        self.assertEqual(Counter(b), Counter(expected))
