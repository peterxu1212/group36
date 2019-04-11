# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:39:20 2019

@author: PeterXu-Desktop
"""

import numpy as np  # linear algebra
import pandas as pd  # data processing

import json

#import csv

training_set = []
testing_set = []

str_input_train_file = "../../../yelp_review_full_csv/yelp_review_full_csv/train.csv"

str_output_train_file = "../../../datasets/Zhang_5_label_yelp_dataset/data/data.json"


train = pd.read_csv(str_input_train_file)

data_labels = train.iloc[:, 0]
data_texts = train.iloc[:, 1]

i_len_train = len(data_texts)

print("i_len_train = ", i_len_train)

for i_index in range(0, i_len_train):
    
    training_item = {}
    
    training_item['review'] = data_texts[i_index]
    training_item['rating'] = int(data_labels[i_index])
    
    training_set.append(training_item)

#data_images = train.iloc[:, 1:]

#data_images = data_images.values

#data_images = np.reshape(data_images, (data_images.shape[0], 64, 64))



with open(str_output_train_file, "w") as wf_training_set:
    json.dump(training_set, wf_training_set)

