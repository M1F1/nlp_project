import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')


def filter_df_rows(df: pd.DataFrame, column_name: str, not_wanted_value):
    arg_1_correct_types = pd.DataFrame
    arg_2_correct_types = str
    arg_3_correct_types = (str, int)
    if (
        isinstance(df, arg_1_correct_types)
        and isinstance(column_name, arg_2_correct_types)
        and isinstance(not_wanted_value, arg_3_correct_types)
    ):
        return df[df[column_name] != not_wanted_value].reset_index(
            drop=True
        )
    else:
        raise ValueError


def tokenize(sentence: str):
    arg_1_correct_types = str
    if isinstance(sentence, arg_1_correct_types):
        tokenize_sentence = nltk.word_tokenize(sentence)
        return tokenize_sentence
    else:
        raise ValueError


def to_lower_case(sentence: str):
    arg_1_correct_types = str
    if isinstance(sentence, arg_1_correct_types):
        return sentence.lower()
    else:
        raise ValueError


def create_vocab_set(df: pd.DataFrame, columns_names: [str]):
    arg_1_correct_types, arg_2_correct_types = pd.DataFrame, list
    if isinstance(df, arg_1_correct_types) and isinstance(
        columns_names, arg_2_correct_types
    ):
        words = []
        for column in columns_names:
            words += df[column].sum()
        return set(words)
    else:
        raise ValueError


def create_embeddings_vocab_intersection(
    embeddings_list: [str], vocab_list: [str]
):
    arg_1_correct_types, arg_2_correct_types = list, list
    if isinstance(embeddings_list, arg_1_correct_types) and isinstance(
        vocab_list, arg_2_correct_types
    ):
        idx_list = []
        for word in vocab_list:
            try:
                idx_list.append(embeddings_list.index(word))
            except ValueError:
                pass
        return idx_list
    else:
        raise ValueError


def count_unk_words(embeddings_idx: [int], vocab_list: [str]):
    arg_1_correct_types, arg_2_correct_types = list, list
    if isinstance(embeddings_idx, arg_1_correct_types) and isinstance(
        vocab_list, arg_2_correct_types
    ):
        unk_words_num = len(vocab_list) - len(embeddings_idx)
        return unk_words_num
    else:
        raise ValueError


def extend_embeddings_matrix(init_embeddings_matrix: np.ndarray):

    arg_1_correct_types = np.ndarray
    if isinstance(init_embeddings_matrix, arg_1_correct_types):
        number_of_random_initialize_vectors = 4
        extended_embeddings_matrix = np.zeros(
            (
                init_embeddings_matrix.shape[0]
                + number_of_random_initialize_vectors,
                init_embeddings_matrix.shape[1],
            )
        )

        new_vectors = np.random.rand(
            number_of_random_initialize_vectors,
            init_embeddings_matrix.shape[1],
        )
        extended_embeddings_matrix[0] = new_vectors[0]
        extended_embeddings_matrix[-3:] = new_vectors[1:]
        extended_embeddings_matrix[1:-3] = init_embeddings_matrix
        return extended_embeddings_matrix
    else:
        raise ValueError


def extend_embeddings_list(embeddings_list):
    arg_1_correct_types = list
    if isinstance(embeddings_list, arg_1_correct_types):
        new_vocab_list = []
        new_vocab_list = ["<pad>"] + embeddings_list
        new_vocab_list += ["<bos>", "<eos>", "<unk>"]
        return new_vocab_list
    else:
        raise ValueError


def extend_vocab_list(vocab_list):
    arg_1_correct_types = list
    if isinstance(vocab_list, arg_1_correct_types):
        new_vocab_list = []
        new_vocab_list = ["<pad>"] + vocab_list
        new_vocab_list += ["<bos>", "<eos>"]
        return new_vocab_list
    else:
        raise ValueError


def create_vocab_dict(embeddings_list, vocab_list):
    arg_1_correct_types, arg_2_correct_types = list, list
    if isinstance(embeddings_list, arg_1_correct_types) and isinstance(
        vocab_list, arg_2_correct_types
    ):
        vocab_dict = {}
        for word in vocab_list:
            try:
                vocab_dict[word] = embeddings_list.index(word)
            except ValueError:
                vocab_dict[word] = embeddings_list.index("<unk>")
        return vocab_dict
    else:
        raise ValueError


def add_beginning_and_ending_word_to_sentence(
    beginning_word, ending_word, tokenize_sentence
):
    arg_1_correct_types, arg_2_correct_types, arg_3_correct_types = (
        str,
        str,
        list,
    )
    if (
        isinstance(beginning_word, arg_1_correct_types)
        and isinstance(ending_word, arg_2_correct_types)
        and isinstance(tokenize_sentence, arg_3_correct_types)
    ):
        new_tokenize_sentence = []
        new_tokenize_sentence = ["<bos>"] + tokenize_sentence
        new_tokenize_sentence += ["<eos>"]
        return new_tokenize_sentence
    else:
        raise ValueError
