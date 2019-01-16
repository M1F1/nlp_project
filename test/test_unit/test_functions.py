import unittest

"""
Unit test for the calculator library
"""
main = __import__("main")


class TestCalculator(unittest.TestCase):
    def test_addition(self):
        assert 4 == main.add(2, 2)

    def test_subtraction(self):
        assert 2 == main.subtract(4, 2)

    def test_multiplication(self):
        assert 100 == main.multiply(10, 10)


if __name__ == "__main__":
    unittest.main()
