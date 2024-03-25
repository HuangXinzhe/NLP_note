# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# basic settings for DL_4_NER Project
BASE_DIR = "06_dl_ner/NERSystem"
CORPUS_PATH = f"{BASE_DIR}/train.txt"  # 语料库路径

KERAS_MODEL_SAVE_PATH = f'{BASE_DIR}/Bi-LSTM-4-NER.h5'  # 模型保存路径

WORD_DICTIONARY_PATH = f'{BASE_DIR}/word_dictionary.pk'
InVERSE_WORD_DICTIONARY_PATH = f'{BASE_DIR}/inverse_word_dictionary.pk'
LABEL_DICTIONARY_PATH = f'{BASE_DIR}/label_dictionary.pk'
OUTPUT_DICTIONARY_PATH = f'{BASE_DIR}/output_dictionary.pk'


CONSTANTS = [
             KERAS_MODEL_SAVE_PATH,
             InVERSE_WORD_DICTIONARY_PATH,
             WORD_DICTIONARY_PATH,
             LABEL_DICTIONARY_PATH,
             OUTPUT_DICTIONARY_PATH
             ]

# load data from corpus to from pandas DataFrame
def load_data():
    with open(CORPUS_PATH, 'r') as f:
        text_data = [text.strip() for text in f.readlines()]
    text_data = [text_data[k].split('\t') for k in range(0, len(text_data))]
    index = range(0, len(text_data), 3)

    # Transforming data to matrix format for neural network
    input_data = list()
    for i in range(1, len(index) - 1):
        rows = text_data[index[i-1]:index[i]]
        sentence_no = np.array([i]*len(rows[0]), dtype=str)
        rows.append(sentence_no)
        rows = np.array(rows).T
        input_data.append(rows)

    input_data = pd.DataFrame(np.concatenate([item for item in input_data]),\
                               columns=['word', 'pos', 'tag', 'sent_no'])

    return input_data