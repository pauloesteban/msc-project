# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:55:58 2015

@author: paulochiliguano
"""


import tables
import numpy as np
import cPickle
import sklearn.preprocessing as preprocessing

#Read HDF5 file that contains log-mel spectrograms
filename = '/homes/pchilguano/msc_project/dataset/gtzan/features/\
feats_3sec_9.h5'
with tables.openFile(filename, 'r') as f:
    features = f.root.x.read()
    #filenames = f.root.filenames.read()

#Pre-processing of spectrograms mean=0 and std=1
#initial_shape = features.shape[1:]
n_per_example = np.prod(features.shape[1:-1])
number_of_features = features.shape[-1]
flat_data = features.view()
flat_data.shape = (-1, number_of_features)
scaler = preprocessing.StandardScaler().fit(flat_data)
flat_data = scaler.transform(flat_data)
flat_data.shape = (features.shape[0], -1)
#flat_targets = filenames.repeat(n_per_example)
#genre = np.asarray([line.strip().split('\t')[1] for line in open(filename,'r').readlines()])

#Read labels from ground truth
filename = '/homes/pchilguano/msc_project/dataset/gtzan/lists/ground_truth.txt'
with open(filename, 'r') as f:
    tag_set = set()
    for line in f:
        tag = line.strip().split('\t')[1]
        tag_set.add(tag)

#Assign label to a discrete number
tag_dict = dict([(item, index) for index, item in enumerate(sorted(tag_set))])
with open(filename, 'r') as f:
    target = np.asarray([], dtype='int32')
    mp3_dict = {}
    for line in f:
        tag = line.strip().split('\t')[1]
        target = np.append(target, tag_dict[tag])

train_input, valid_input, test_input = np.array_split(
    flat_data,
    [flat_data.shape[0]*1/2,
    flat_data.shape[0]*3/4]
)
train_target, valid_target, test_target = np.array_split(
    target,
    [target.shape[0]*1/2,
    target.shape[0]*3/4]
)

f = file('/homes/pchilguano/msc_project/dataset/gtzan/features/\
gtzan_3sec_9.pkl', 'wb')
cPickle.dump(
    (
        (train_input, train_target),
        (valid_input, valid_target),
        (test_input, test_target)
    ),
    f,
    protocol=cPickle.HIGHEST_PROTOCOL
)
f.close()

'''
flat_target = target.repeat(n_per_example)

train_input, valid_input, test_input = np.array_split(flat_data, [flat_data.shape[0]*4/5, flat_data.shape[0]*9/10])
train_target, valid_target, test_target = np.array_split(flat_target, [flat_target.shape[0]*4/5, flat_target.shape[0]*9/10])

f = file('/homes/pchilguano/deep_learning/gtzan_logistic.pkl', 'wb')
cPickle.dump(((train_input, train_target), (valid_input, valid_target), (test_input, test_target)), f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
'''
