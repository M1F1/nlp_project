import numpy as np
import pandas as pd
import nltk
import tqdm
import os
import pickle

nltk.download("punkt")


def filter_df_rows(df: pd.DataFrame, column_name: str, not_wanted_value):
    arg_1_correct_types = pd.DataFrame
    arg_2_correct_types = str
    arg_3_correct_types = (str, int)
    if (
        isinstance(df, arg_1_correct_types)
        and isinstance(column_name, arg_2_correct_types)
        and isinstance(not_wanted_value, arg_3_correct_types)
    ):
        return df[df[column_name] != not_wanted_value].reset_index(drop=True)
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
        words = set()
        for i in tqdm.tnrange(df.shape[0]):
            for column in columns_names:
                words.update(df.iloc[i][column])
        return words
    else:
        raise ValueError


def create_embeddings_vocab_intersection(embeddings_list: [str], vocab_list: [str]):
    arg_1_correct_types, arg_2_correct_types = list, list
    if isinstance(embeddings_list, arg_1_correct_types) and isinstance(
        vocab_list, arg_2_correct_types
    ):
        idx_list = []
        for i in tqdm.tnrange(len(vocab_list)):
            if vocab_list[i] in embeddings_list:
                idx_list.append(embeddings_list.index(vocab_list[i]))
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
                init_embeddings_matrix.shape[0] + number_of_random_initialize_vectors,
                init_embeddings_matrix.shape[1],
            )
        )

        new_vectors = np.random.rand(
            number_of_random_initialize_vectors, init_embeddings_matrix.shape[1]
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
        for i in tqdm.tnrange(len(vocab_list)):
            try:
                vocab_dict[vocab_list[i]] = embeddings_list.index(vocab_list[i])
            except ValueError:
                vocab_dict[vocab_list[i]] = embeddings_list.index("<unk>")
        return vocab_dict
    else:
        raise ValueError


def add_beginning_and_ending_word_to_sentence(tokenize_sentence):
    arg_1_correct_types = list
    if isinstance(tokenize_sentence, arg_1_correct_types):
        new_tokenize_sentence = []
        new_tokenize_sentence = ["<bos>"] + tokenize_sentence
        new_tokenize_sentence += ["<eos>"]
        return new_tokenize_sentence
    else:
        raise ValueError


# TODO: write test
def save_obj(obj, name, dir_path):
    filename = name + ".pkl"
    with open(os.path.join(dir_path, filename), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# TODO: write_test
def load_obj(filename, dir_path):
    with open(os.path.join(dir_path, filename), "rb") as f:
        return pickle.load(f)


# TODO: prepere data:
# write batch creation_function which compute the longest sentence in batch and
# padd others sentences
# add bos and eos to every sentence
# create numpy matrix of idx for premises and hipothesis with padd


def define_batches_amount(rows_idx: np.ndarray, batch_size: int):
    batches_amount = (rows_idx.shape[0] // batch_size) + 1
    return batches_amount


def shuffle_data_idx(rows_idx: np.ndarray, seed: int):
    np.random.seed(seed)
    np.random.shuffle(rows_idx)
    return rows_idx


def generate_batch_idx_from_data_idx(
    rows_idx: np.ndarray, batch_size: int, batch_number: int
):
    first_element = batch_number * batch_size
    last_element = (batch_number + 1) * batch_size
    if last_element <= rows_idx.size:
        result = rows_idx[first_element:last_element]
        return result
    else:
        result = rows_idx[first_element: rows_idx.size]
        return result


def get_labels_and_batch_lists_representation(df, selected_rows_idx_list):
    selected_rows = df.iloc[selected_rows_idx_list, :]
    labels = selected_rows.iloc[:, 0].values.tolist()
    premises = selected_rows.iloc[:, 1].values.tolist()
    hypothesis = selected_rows.iloc[:, 2].values.tolist()
    return labels, premises, hypothesis


def create_word_to_idx_representation(list_of_words_list, vocab_list):
    return [[vocab_list[word] for word in l] for l in list_of_words_list]


def create_batch_matrix_representation(list_of_words_lists, vocab_dict):
    words_lists_lengths = [len(l) for l in list_of_words_lists]
    max_length = max(words_lists_lengths)
    batch_array = np.ones((len(list_of_words_lists), max_length)) * vocab_dict["<pad>"]

    for i, list_len in enumerate(words_lists_lengths):
        sequence = list_of_words_lists[i]
        batch_array[i, :list_len] = sequence[:list_len]

    return batch_array
