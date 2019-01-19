import pandas as pd
import numpy as np
from app import preprocessing
import os
import csv
root_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(root_dir, 'data')
word_vectors_path = os.path.join(data_path, 'word_vectors.txt')

data = pd.read_pickle(os.path.join(data_path, 'train', 'train_processed_data.pickle'))
# vocab_set = preprocessing.create_vocab_set(data, ['sentence1', 'sentence2'])
# vocab_list = list(vocab_set)
print('load vocab list')
vocab_list = pd.read_pickle(os.path.join(data_path, 'vocab_list.pickle'))
vocab_list = list(vocab_list[0])
print('Done')

print('Load GLoVE words vectors')
with open(word_vectors_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=" ", quotechar=None)
    lines = []
    for line in reader:
        lines.append(line)

embeddings = pd.DataFrame(lines)
print('Done')
embeddings_series = embeddings.iloc[:, 0]
embeddings_list = list(embeddings_series.values.tolist())

# intersection_words_idx = preprocessing.\
#     create_embeddings_vocab_intersection(embeddings_list=embeddings_list, vocab_list=vocab_list)
print('load inter_df')
inter_df = pd.read_pickle(os.path.join(data_path, 'intersection_idx.pickle'))
intersection_words_idx = list(inter_df[0].values.tolist())
print(F'Number of <unk> words in vocabulary: {preprocessing.count_unk_words(intersection_words_idx, vocab_list)}')
inter_df_embeddings = embeddings.iloc[intersection_words_idx]
final_embeddings_list = inter_df_embeddings.iloc[:, 0].values.tolist()
embeddings_vectors_values = inter_df_embeddings.iloc[:, 1:]
embeddings_matrix = embeddings_vectors_values.values
extended_embeddings_matrix = preprocessing.extend_embeddings_matrix(embeddings_matrix)
np.save(os.path.join(data_path, 'vocab', 'final_embeddings_matrix'), extended_embeddings_matrix)
extended_embeddings_list = preprocessing.extend_embeddings_list(final_embeddings_list)
extended_vocab_list = preprocessing.extend_vocab_list(vocab_list)
vocab_dict = preprocessing.create_vocab_dict(extended_embeddings_list, extended_vocab_list)
print('saving vocab_dict')
preprocessing.save_obj(vocab_dict, 'vocab_dict', os.path.join(data_path, 'vocab'))
print('done')
