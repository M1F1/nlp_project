import numpy as np
import pandas as pd
import nltk


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


def create_vocab_set(df: pd.DataFrame, columns_names=[str]):
    arg_1_correct_types = pd.DataFrame
    arg_2_correct_types = list
    if isinstance(df, arg_1_correct_types) and\
       isinstance(columns_names, arg_2_correct_types):
        words = []
        for column in columns_names:
            # import pdb; pdb.set_trace()
            words += df[column].sum()
        return set(words)
    else:
        raise ValueError
