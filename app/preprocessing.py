import numpy as np
import pandas as pd
import nltk
import tqdm
from tqdm import tqdm
import os
import pickle
import argparse

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
        print('Creating vocab_set:')
        for i in tqdm(range(df.shape[0])):
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
        print('\nCreating vocab intercestion with embeddings:')
        for i in tqdm(range(len(vocab_list))):
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
        print('\nCreating vocab dict: ')
        for i in tqdm(range(len(vocab_list))):
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


def create_word_to_idx_representation(list_of_words_list, vocab_dict):
    return [[vocab_dict.get(word, vocab_dict['<unk>']) for word in l] for l in list_of_words_list]


def create_batch_matrix_representation(list_of_idx_words_lists, vocab_dict):
    words_lists_lengths = [len(l) for l in list_of_idx_words_lists]
    max_length = max(words_lists_lengths)
    batch_array = np.ones((len(list_of_idx_words_lists), max_length)) * vocab_dict["<pad>"]

    for i, list_len in enumerate(words_lists_lengths):
        sequence = list_of_idx_words_lists[i]
        batch_array[i, :list_len] = sequence[:list_len]

    return batch_array.astype(np.int)


def labels2idx(labels):
    labels_dict = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
    labels_idx = [labels_dict[label] for label in labels]
    return np.array(labels_idx)

def preprocessing_pipe(data_path, data_set, data_filename):
    import tqdm
    particular_data_path = os.path.join(data_path, data_set, data_filename)
    import csv
    tqdm.tqdm.pandas()
    with open(particular_data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        lines = []
        for line in reader:
            lines.append(line)

    columns = lines.pop(0)
    df = pd.DataFrame(data=lines, columns=columns)
    data = df[['gold_label', 'sentence1', 'sentence2']].copy()
    if data_set != 'train':
        print(data.loc[data['gold_label'] == '-'].iloc[:10])
        data = filter_df_rows(data, 'gold_label', '-')

    print(F'{data_set} data processing:')
    print(F'\nProcessing premises')
    data['sentence1'] = data['sentence1'].progress_map(lambda x: add_beginning_and_ending_word_to_sentence(
        tokenize(to_lower_case(x))))
    print(F'\nProcessing hypothesises')
    data['sentence2'] = data['sentence2'].progress_map(lambda x: add_beginning_and_ending_word_to_sentence(
        tokenize(to_lower_case(x))))
    saving_path = os.path.join(data_path, data_set, data_set + '_processed_data.pickle')
    print(F'\nSaving processed data to: {saving_path}')
    data.to_pickle(os.path.join(data_path, data_set, data_set + '_processed_data.pickle'))
    print('Data saved.')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=False,
                        help="The input data dir path.")

    parser.add_argument("--dataset_part",
                        default=None,
                        type=str,
                        required=True,
                        help="The input datas_set part. Posibilities: trian, dev, test")
    parser.add_argument("--dataset_part_filename",
                        default=None,
                        type=str,
                        required=True,
                        help="The input dataset part filename.")
    args = parser.parse_args()
    preprocessing_pipe(args.data_path, args.dataset_part, args.dataset_part_filename)


if __name__ == '__main__':
    main()
    # dev_data_filename = 'multinli_1.0_dev_matched.txt'
    # test_data_filename = 'multinli_1.0_dev_mismatched.txt'
    # train_data_filename = 'multinli_1.0_train.txt'

