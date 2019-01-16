import unittest
import numpy as np
import pandas as pd

"""
Unit test for the calculator library
"""
from app import preprocessing
from app import main


class TestPreProcessingFunctions(unittest.TestCase):
    test_data = {"col1": [1, 2, 2], "col2": [3, 4, 10]}
    df = pd.DataFrame(data=test_data)
    test_sentence = "Ala ma Kota i SŁUCHA SLIPKNOTA."

    def test_filtering_return_error_if_args_invalid_type(self):
        self.assertRaises(
            ValueError, preprocessing.filter_df_rows, np.array([0]), "three", "2"
        )

    def test_filtering(self):
        result_df = preprocessing.filter_df_rows(
            self.__class__.df, column_name="col1", not_wanted_value=1
        )

        target_data = {"col1": [2, 2], "col2": [4, 10]}
        target_df = pd.DataFrame(data=target_data)
        self.assertEqual(result_df, target_df)

    def test_tokenize_return_error_if_args_invalid_type(self):
        self.assertRaises(ValueError, preprocessing.tokenize, 2)

    def test_tokenize(self):
        result_sentence = preprocessing.tokenize(self.__class__.test_sentence)
        target_sentence = ["Ala", "ma", "Kota", "i", "SŁUCHA", "SLIPKNOTA."]
        self.assertEqual(result_sentence, target_sentence)

    def test_to_lower_case_return_error_if_args_invalid_type(self):
        self.assertRaises(ValueError, preprocessing.to_lower_case, 2)

    def test_to_lower_case(self):
        result_sentence = preprocessing.to_lower_case(self.__class__.test_sentence)
        target_sentence = ["ala ma kota i słucha slipknota."]
        self.assertEqual(result_sentence, target_sentence)

    def test_create_vocab_list_return_error_if_args_invalid_type(self):
        self.assertRaises(
            ValueError,
            preprocessing.create_vocab,
            self.__class__.df,
            columns_names=["col1", "col2"],
        )

    def test_create_vocab_list(self):

        vocab_test_data = {
            "col1": [["ala", "ma", "kota"], ["kot", "ma", "ale"]],
            "col2": [["kota", "ma", "ale"], ["ala", "ma", "ale"]],
        }
        vocab_test_df = pd.DataFrame(data=vocab_test_data)
        result_vocab_list = preprocessing.create_vocab_list(
            vocab_test_df, columns_names=["col1", "col2"]
        )
        target_vocab_list = ["ala", "ma", "kota", "kot", "ale"]
        self.assertEqual(result_vocab_list, target_vocab_list)

    def test_valid_input_to_create_vocab_dict(self):
        self.assertRaises(
            ValueError, preprocessing.create_vocab_dict, 1, ['ala']
        )

    def test_create_vocab_dict(self):
        embedding_list = ["ala", "ma",
                          "kota", "i",
                          "slucha", "spliknota",
                          '<unk>', '<bos>', '<eos>']
        vocab_list = ["ala", "ma",
                      "kota", "spliknota",
                      "a", "kot", "nie",
                      '<bos>', '<eos>']
        target_vocab_dict = {'ala': 0, 'ma': 1,
                             'kota': 2, 'spliknota': 5,
                             'a': len(embedding_list) - 3,
                             'kot': len(embedding_list) - 3,
                             'nie': len(embedding_list) - 3,
                             '<bos>': len(embedding_list) - 2,
                             '<eos>': len(embedding_list) - 1
                             }
        result_vocab_dict = preprocessing.create_vocab_dict(
            embeddings_words_list=embedding_list, corpus_words_list=vocab_list
        )
        self.assertEqual(result_vocab_dict, target_vocab_dict)
    # TODO: test_function_creating_embeddings_numpy_array + add 3 random generated vectors for eos bos and unk
    # create emmbedidngs matrix base on base on vocab_dict
    # TODO: test_function_counting_all_elements_in_vocab + <unk> words


class TestCalculatorFunctions(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(main.add(np.array([2]), np.array([2])), np.array([4]))

    def test_calculator_returns_error_message_if_both_args_not_numbers(self):
        self.assertRaises(ValueError, main.add, "ad", "three")

    def test_calculator_returns_error_message_if_first_arg_not_numbers(self):
        self.assertRaises(ValueError, main.add, "ad", 3)

    def test_calculator_returns_error_message_if_second_arg_not_numbers(self):
        self.assertRaises(ValueError, main.add, 3, "three")

    def test_subtraction(self):
        self.assertEqual(main.subtract(2, 2), 0)

    def test_multiplication(self):
        self.assertEqual(main.multiply(2, 2), 4)


if __name__ == "__main__":
    unittest.main()
