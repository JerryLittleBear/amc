# -*- coding: utf-8 -*-
import pickle
import numpy as np


def load_data(filename):
     #read data from data file
     with open(filename, 'rb') as f:
         data = pickle.load(f, encoding='bytes')#python3
     #data = pickle(f)
     return data[b'data'], data[b'labels']
    
class CifarData:
    def __init__(self, filenames, need_shuffle):
       # 参数1：文件夹 参数2：是否需要随机打乱
        all_data = []
        all_labels = []
        
        for filename in filenames:
            # 将所有的数据， 标签分别存放在两个list中
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        
        # 将list组成一个np.array    
        self._data = np.vstack(all_data)
        # 对数据进行归一化， 尺度在[-1, 1]
        self._data = self._data / 127.5 - 1
        # 把list变成np.array
        self._labels = np.hstack(all_labels)
        # 样本数量
        self._num_examples = self._data.shape[0]
        # 是否需打乱
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shffle_data()
            
    def _shffle_data(self):
        # np.random.permutation() 从0到参数， 随机打乱
        p = np.random.permutation(self._num_examples)
        # 保存已经打乱的数据
        self._data = self._data[p]
        self._labels = self._labels[p]
        
    def next_batch(self, batch_size):
        # 开始点 + 数量 = 结束点
        end_indicator = self._indicator + batch_size
        
        if end_indicator > self._num_examples:
            if self._need_shuffle:
              
                self._shffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('have no more examples')
        if end_indicator > self._num_examples:
            raise Exception('batch size is larger than all examples')
           
        batch_data = self._data[self._indicator : end_indicator]
        batch_labels = self._labels[self._indicator : end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels
