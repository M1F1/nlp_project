import pandas as pd
import numpy as np
import os
from app import preprocessing as prep
import pickle
from tqdm import tqdm
import time
import sys


def load_obj(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.load(f)


class Batch:
    def __init__(self, premises: np.ndarray, hypothesises: np.ndarray, labels: np.ndarray):
        self.premises = premises
        self.hypothesises = hypothesises
        self.labels = labels

    def __str__(self):
        return F'premises: {self.premises}\n' + \
               F'hypothesises: {self.hypothesises}\n' + \
               F'labels: {self.labels}'


class DataLoader:

    def __init__(self,
                 train_path,
                 test_path,
                 dev_path,
                 vocab_path,
                 embeddings_path,
                 batch_size):

        self.train_data = pd.read_pickle(train_path)
        self.test_data = pd.read_pickle(test_path)
        self.dev_data = pd.read_pickle(dev_path)
        self.vocab_dict = load_obj(vocab_path)
        self.labels_dict = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
        self.embeddings = np.load(embeddings_path)
        self.batch_size = batch_size
        self.vocab_dict['<unk>'] = self.embeddings.shape[0] - 1

    def _prepare_batches(self, data, data_name):
        print(F'\nPreparing {data_name} batches:\n', end='')
        rows_idx = np.arange(data.shape[0])
        batches_nbr = prep.define_batches_amount(rows_idx, self.batch_size)
        batches = []
        time.sleep(1)
        for i in tqdm(range(batches_nbr)):
            selected_idx = prep.generate_batch_idx_from_data_idx(rows_idx, self.batch_size, i)
            labels, premises, hypothesises = prep.get_labels_and_batch_lists_representation(data, selected_idx)
            premises = prep.create_word_to_idx_representation(premises, self.vocab_dict)
            hypothesises = prep.create_word_to_idx_representation(hypothesises, self.vocab_dict)
            premises = prep.create_batch_matrix_representation(premises, self.vocab_dict)
            hypothesises = prep.create_batch_matrix_representation(hypothesises, self.vocab_dict)
            labels = prep.labels2idx(labels)
            batch = Batch(premises, hypothesises, labels)
            batches.append(batch)

        time.sleep(1)
        return batches

    def get_batches(self):
        train_batches = self._prepare_batches(self.train_data, 'train')
        dev_batches = self._prepare_batches(self.dev_data, 'dev')
        test_batches = self._prepare_batches(self.test_data, 'test')
        return train_batches, dev_batches, test_batches


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root_dir, 'data')
    train_path = os.path.join(data_path, 'train', 'train_processed_data.pickle')
    dev_path = os.path.join(data_path, 'dev', 'dev_processed_data.pickle')
    test_path = os.path.join(data_path, 'test', 'test_processed_data.pickle')
    embeddings_path = os.path.join(data_path, 'vocab', 'final_embeddings_matrix.npy')
    vocab_path = os.path.join(data_path, 'vocab', 'vocab_dict.pkl')
    data_loader = DataLoader(train_path=train_path,
                             test_path=test_path,
                             dev_path=dev_path,
                             vocab_path=vocab_path,
                             embeddings_path=embeddings_path,
                             batch_size=32)
    tb, db, testb = data_loader.get_batches()
