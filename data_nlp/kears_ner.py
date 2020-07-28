# -*- coding: utf-8 -*-
"""
@time   : 2020/07/18 10:45
@author : 姚明伟
"""
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras_contrib.layers import CRF
import pickle
from keras.layers import Embedding, Bidirectional, LSTM


class Data_set:
    # 定义Dataset类，封装一些数据读入和预处理方法。
    def __init__(self, data_path, labels):
        with open(data_path, "rb") as f:
            self.data = f.read().decode("utf-8")
        self.process_data = self.process_data()
        self.labels = labels

    def process_data(self):
        train_data = self.data.split("\n\n")
        train_data = [token.split("\n") for token in train_data]
        train_data = [[j.split() for j in i] for i in train_data]
        train_data.pop()
        return train_data

    def save_vocab(self, save_path):
        all_char = [char[0] for sen in self.process_data for char in sen]
        chars = set(all_char)
        word2id = {char: id_ + 1 for id_, char in enumerate(chars)}
        word2id["unk"] = 0
        with open(save_path, "wb") as f:
            pickle.dump(word2id, f)
        return word2id

    def generate_data(self, vocab, maxlen):
        char_data_sen = [[token[0] for token in i] for i in self.process_data]
        label_sen = [[token[1] for token in i] for i in self.process_data]
        sen2id = [[vocab.get(char, 0) for char in sen] for sen in char_data_sen]
        label2id = {label: id_ for id_, label in enumerate(self.labels)}
        lab_sen2id = [[label2id.get(lab, 0) for lab in sen] for sen in label_sen]
        sen_pad = pad_sequences(sen2id, maxlen)
        lab_pad = pad_sequences(lab_sen2id, maxlen, value=-1)
        lab_pad = np.expand_dims(lab_pad, 2)
        return sen_pad, lab_pad