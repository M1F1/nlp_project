import unittest
import numpy as np
import pandas as pd

"""
Unit test for the calculator library
"""
from app import preprocessing


class TestPreProcessing(unittest.TestCase):
    test_data = {"col1": [1, 2, 2], "col2": [3, 4, 10]}
    df = pd.DataFrame(data=test_data)
    test_sentence = "Ala ma Kota i SŁUCHA SLIPKNOTA."

    def test_filter_df_rows_return_error_if_args_invalid_type(self):
        self.assertRaises(
            ValueError, preprocessing.filter_df_rows, np.array([0]), "three", "2"
        )

    test_filter_df_rows_return_error_if_args_invalid_type.layer = 1

    def test_filter_df_rows(self):
        result_df = preprocessing.filter_df_rows(
            self.__class__.df, column_name="col1", not_wanted_value=1
        )

        target_data = {"col1": [2, 2], "col2": [4, 10]}
        target_df = pd.DataFrame(data=target_data)
        # import pdb; pdb.set_trace()
        pd.testing.assert_frame_equal(result_df, target_df)

    test_filter_df_rows.layer = 1

    def test_tokenize_return_error_if_args_invalid_type(self):
        self.assertRaises(ValueError, preprocessing.tokenize, 2)

    test_tokenize_return_error_if_args_invalid_type.layer = 1

    def test_tokenize(self):
        result_sentence = preprocessing.tokenize(self.__class__.test_sentence)
        target_sentence = ["Ala", "ma", "Kota", "i", "SŁUCHA", "SLIPKNOTA", "."]
        self.assertEqual(result_sentence, target_sentence)

    test_tokenize.layer = 1

    def test_to_lower_case_return_error_if_args_invalid_type(self):
        self.assertRaises(ValueError, preprocessing.to_lower_case, 2)

    test_to_lower_case_return_error_if_args_invalid_type.layer = 1

    def test_to_lower_case(self):
        result_sentence = preprocessing.to_lower_case(self.__class__.test_sentence)
        target_sentence = "ala ma kota i słucha slipknota."
        self.assertEqual(result_sentence, target_sentence)

    test_to_lower_case.layer = 1

    def test_create_vocab_set_return_error_if_args_invalid_type(self):
        self.assertRaises(
            ValueError,
            preprocessing.create_vocab_set,
            self.__class__.df,
            columns_names=1,
        )

    test_create_vocab_set_return_error_if_args_invalid_type.layer = 2

    def test_create_vocab_set(self):

        vocab_test_data = {
            "col1": [["ala", "ma", "kota"], ["kot", "ma", "ale"]],
            "col2": [["kota", "ma", "ale"], ["ala", "ma", "ale"]],
        }
        vocab_test_df = pd.DataFrame(data=vocab_test_data)
        result_vocab_set = preprocessing.create_vocab_set(
            vocab_test_df, columns_names=["col1", "col2"]
        )
        target_vocab_set = {"ala", "ma", "kota", "kot", "ale"}
        self.assertEqual(result_vocab_set, target_vocab_set)

    test_create_vocab_set.layer = 2

    def test_valid_input_to_create_embeddings_and_vocab_intersection(self):
        self.assertRaises(
            ValueError, preprocessing.create_embeddings_vocab_intersection, 1, ["ala"]
        )

    test_valid_input_to_create_embeddings_and_vocab_intersection.layer = 2

    def test_create_embeddings_and_vocab_intersection(self):
        vocab_list = ["ala", "ma", "kota", "spliknota", "a", "kot", "nie"]
        embeddings_list = ["ala", "ma", "kota", "i", "slucha", "spliknota"]
        target_embeddings_idx = [0, 1, 2, 5]
        result_embeddings_idx = preprocessing.create_embeddings_vocab_intersection(
            embeddings_list=embeddings_list, vocab_list=vocab_list
        )
        self.assertEqual(result_embeddings_idx, target_embeddings_idx)

    test_create_embeddings_and_vocab_intersection.layer = 2

    def test_valid_input_to_count_unk_words(self):
        self.assertRaises(ValueError, preprocessing.count_unk_words, 1, [2, 10])

    test_valid_input_to_count_unk_words.layer = 2

    def test_count_unk_words(self):
        embeddings_idx = [0, 1, 2, 5]
        vocab_list = ["ala", "ma", "kota", "spliknota", "a", "kot", "nie"]
        target_num = 3
        result_num = preprocessing.count_unk_words(
            embeddings_idx=embeddings_idx, vocab_list=vocab_list
        )
        self.assertEqual(target_num, result_num)

    test_count_unk_words.layer = 2

    def test_valid_input_to_extend_embeddings_matrix(self):
        self.assertRaises(ValueError, preprocessing.extend_embeddings_matrix, [1, 2])

    test_valid_input_to_extend_embeddings_matrix.layer = 3

    def test_extend_embeddings_matrix(self):
        init_embeddings_matrix = np.array([[1, 2, 3], [2, 3, 4], [2, 4, 5]])
        target_extended_embeddings_matrix = np.zeros(
            (init_embeddings_matrix.shape[0] + 4, init_embeddings_matrix.shape[1])
        )
        result_extended_embeddings_matrix = preprocessing.extend_embeddings_matrix(
            init_embeddings_matrix=init_embeddings_matrix
        )
        self.assertEqual(
            result_extended_embeddings_matrix.shape,
            target_extended_embeddings_matrix.shape,
        )
        target_non_zero_rows = np.array([])
        np.testing.assert_array_equal(
            np.where(~result_extended_embeddings_matrix.any(axis=1))[0],
            target_non_zero_rows,
        )

    test_extend_embeddings_matrix.layer = 3

    def test_valid_input_to_extend_embeddings_list(self):
        self.assertRaises(ValueError, preprocessing.extend_embeddings_list, 1)

    test_valid_input_to_extend_embeddings_list.layer = 3

    def test_extend_embeddings_list(self):
        embeddings_list = ["ala", "ma", "kota", "spliknota", "a", "kot", "nie"]
        target_embeddings_list = [
            "<pad>",
            "ala",
            "ma",
            "kota",
            "spliknota",
            "a",
            "kot",
            "nie",
            "<bos>",
            "<eos>",
            "<unk>",
        ]
        result_vocab_list = preprocessing.extend_embeddings_list(
            embeddings_list=embeddings_list
        )
        self.assertEqual(result_vocab_list, target_embeddings_list)

    test_extend_embeddings_list.layer = 3

    def test_valid_input_to_extend_vocab_list(self):
        self.assertRaises(ValueError, preprocessing.extend_vocab_list, 1)

    test_valid_input_to_extend_vocab_list.layer = 3

    def test_extend_vocab_list(self):
        vocab_list = ["ala", "ma", "kota", "spliknota", "a", "kot", "nie"]
        target_vocab_list = [
            "<pad>",
            "ala",
            "ma",
            "kota",
            "spliknota",
            "a",
            "kot",
            "nie",
            "<bos>",
            "<eos>",
        ]
        result_vocab_list = preprocessing.extend_vocab_list(vocab_list=vocab_list)
        self.assertEqual(result_vocab_list, target_vocab_list)

    test_extend_vocab_list.layer = 3

    def test_valid_input_to_create_vocab_dict(self):
        self.assertRaises(ValueError, preprocessing.create_vocab_dict, 1, ["ala"])

    test_valid_input_to_create_vocab_dict.layer = 4

    def test_create_vocab_dict(self):
        embedding_list = [
            "<pad>",
            "ala",
            "ma",
            "kota",
            "spliknota",
            "<bos>",
            "<eos>",
            "<unk>",
        ]
        vocab_list = [
            "<pad>",
            "ala",
            "ma",
            "kota",
            "spliknota",
            "a",
            "kot",
            "nie",
            "<bos>",
            "<eos>",
        ]
        target_vocab_dict = {
            "<pad>": 0,
            "ala": 1,
            "ma": 2,
            "kota": 3,
            "spliknota": 4,
            "a": len(embedding_list) - 1,
            "kot": len(embedding_list) - 1,
            "nie": len(embedding_list) - 1,
            "<bos>": len(embedding_list) - 3,
            "<eos>": len(embedding_list) - 2,
        }
        result_vocab_dict = preprocessing.create_vocab_dict(
            embeddings_list=embedding_list, vocab_list=vocab_list
        )
        self.assertEqual(result_vocab_dict, target_vocab_dict)

    test_create_vocab_dict.layer = 4

    def test_valid_input_to_add_beginning_and_ending_word_to_sentence(self):
        self.assertRaises(
            ValueError, preprocessing.add_beginning_and_ending_word_to_sentence, 1
        )

    test_valid_input_to_add_beginning_and_ending_word_to_sentence.layer = 4

    def test_add_beginning_and_ending_word_to_sentence(self):
        tokenize_sentence = ["ala", "ma", "kota", "i", "slucha", "slipknota"]
        target_tokenize_sentence = [
            "<bos>",
            "ala",
            "ma",
            "kota",
            "i",
            "slucha",
            "slipknota",
            "<eos>",
        ]
        result_sentence = preprocessing.add_beginning_and_ending_word_to_sentence(
            tokenize_sentence=tokenize_sentence
        )
        self.assertEqual(result_sentence, target_tokenize_sentence)

    test_add_beginning_and_ending_word_to_sentence.layer = 4

    def test_define_batches_amount(self):
        data = {
            "label": ["neutral", "entailment", "contradiction"],
            "premises": [["ala", "ma", "kota"], ["kot", "ma", "ale"], ["ala"]],
            "hypothesises": [["kota", "ma"], ["ala", "ma", "ale"], ["not", "ala"]],
        }
        df = pd.DataFrame(data=data)
        rows_idx = np.arange(df.shape[0])
        batch_size = 2
        target_batches_count = 2
        result_batches_count = preprocessing.define_batches_amount(rows_idx, batch_size)
        self.assertEqual(result_batches_count, target_batches_count)

    test_define_batches_amount.layer = 5

    def test_shuffle_data_idx(self):
        data = {
            "label": ["neutral", "entailment", "contradiction"],
            "premises": [["ala", "ma", "kota"], ["kot", "ma", "ale"], ["ala"]],
            "hypothesises": [["kota", "ma"], ["ala", "ma", "ale"], ["not", "ala"]],
        }
        df = pd.DataFrame(data=data)
        rows_idx = np.arange(df.shape[0])
        seed = 43
        target_idx = np.array([1, 2, 0])
        result_idx = preprocessing.shuffle_data_idx(rows_idx, seed)
        np.testing.assert_array_equal(result_idx, target_idx)

    test_shuffle_data_idx.layer = 5

    def test_generate_batch_idx_from_data_idx(self):
        data = {
            "label": ["neutral", "entailment", "contradiction"],
            "premises": [["ala", "ma", "kota"], ["kot", "ma", "ale"], ["ala"]],
            "hypothesises": [["kota", "ma"], ["ala", "ma", "ale"], ["not", "ala"]],
        }
        df = pd.DataFrame(data=data)
        rows_idx = np.arange(df.shape[0])
        target_idx_batch_1 = np.array([0, 1])
        target_idx_batch_2 = np.array([2])

        batch_size = 2
        batch_number_1 = 0
        batch_number_2 = 1

        result_idx_batch_1 = preprocessing.generate_batch_idx_from_data_idx(
            rows_idx, batch_size, batch_number_1
        )
        result_idx_batch_2 = preprocessing.generate_batch_idx_from_data_idx(
            rows_idx, batch_size, batch_number_2
        )
        np.testing.assert_array_equal(result_idx_batch_1, target_idx_batch_1)
        np.testing.assert_array_equal(result_idx_batch_2, target_idx_batch_2)

    test_generate_batch_idx_from_data_idx.layer = 5

    def test_get_labels_and_batch_lists_representation(self):
        data = {
            "label": ["neutral", "entailment"],
            "premises": [["ala", "ma", "kota"], ["kot", "ma", "ale"]],
            "hypothesises": [["kota", "ma"], ["ala", "ma", "ale"]],
        }
        df = pd.DataFrame(data=data)
        target_labels, target_premises_batch, target_hypothesises_batch = (
            ["neutral"],
            [["ala", "ma", "kota"]],
            [["kota", "ma"]],
        )

        result_labels, result_premises_batch, result_hypothesises_batch = preprocessing.get_labels_and_batch_lists_representation(
            df=df, selected_rows_idx_list=[0]
        )
        self.assertEqual(result_premises_batch, target_premises_batch)
        self.assertEqual(result_labels, result_labels)
        self.assertEqual(result_hypothesises_batch, target_hypothesises_batch)

    test_get_labels_and_batch_lists_representation.layer = 5

    def test_create_batch_word_to_idx_representation(self):
        list_of_words_lists = [
            ["<bos>", "ala", "bal", "kur", "<eos>"],
            ["<bos>", "kot", "kot", "<eos>"],
            ["<bos>", "ala", "bal", "kur", "ala", "<eos>"],
        ]
        vocab_dict = {
            "<pad>": 0,
            "ala": 1,
            "bal": 2,
            "kur": 3,
            "kot": 4,
            "<bos>": 5,
            "<eos>": 6,
            "<unk>": 7
        }
        # embeddings_matrix = np.array([[1, 2],
        #                              [2, 3],
        #                              [2, 3],
        #                              [2, 4],
        #                              [2, 3],
        #                              [2, 2],
        #                              [3, 3]])

        target_batch = [[5, 1, 2, 3, 6], [5, 4, 4, 6], [5, 1, 2, 3, 1, 6]]
        result_batch = preprocessing.create_word_to_idx_representation(
            list_of_words_lists, vocab_dict
        )

        np.testing.assert_array_equal(result_batch, target_batch)

    test_create_batch_word_to_idx_representation.layer = 5

    def test_create_batch_matrix_representation(self):
        word_to_idx_sequences = [[5, 1, 2, 3, 6], [5, 4, 4, 6], [5, 1, 2, 3, 1, 6]]
        vocab_dict = {
            "<pad>": 0,
            "ala": 1,
            "bal": 2,
            "kur": 3,
            "kot": 4,
            "<bos>": 5,
            "<eos>": 6,
        }

        target_batch = np.array(
            [[5, 1, 2, 3, 6, 0], [5, 4, 4, 6, 0, 0], [5, 1, 2, 3, 1, 6]]
        )
        result_batch = preprocessing.create_batch_matrix_representation(
            word_to_idx_sequences, vocab_dict
        )

        np.testing.assert_array_equal(result_batch, target_batch)

    test_create_batch_matrix_representation.layer = 5

    def test_labels2idx(self):
        labels = ['neutral', 'entailment', 'contradiction', 'neutral']
        target_idx = np.array([0, 2, 1, 0])
        result_idx = preprocessing.labels2idx(labels)
        np.testing.assert_array_equal(result_idx, target_idx)


if __name__ == "__main__":
    unittest.main()
