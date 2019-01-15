from nose_focus import focus  # enable with @focus
import unittest


class TestSanity(unittest.TestCase):
    def test_sanity(self):
        self.assertEqual(1 + 1, 2)
