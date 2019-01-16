import unittest
from main import add, subtract
"""
Unit tests for the calculator library
"""


class TestCalculator:
    def test_addition(self):
        assert 4 == add(2, 2)

    def test_subtraction(self):
        assert 2 == subtract(4, 2)


class TestFeatures(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
