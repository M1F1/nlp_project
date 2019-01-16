import numpy as np


def add(first_term, second_term):
    correct_types = np.ndarray
    if isinstance(first_term, correct_types) and isinstance(second_term, correct_types):
        result = np.zeros_like(first_term)
        result = first_term + second_term
        return result
    else:
        raise ValueError


def subtract(first_term, second_term):
    return first_term - second_term


def multiply(first_term, second_term):
    return first_term * second_term


if __name__ == "__main__":
    pass
