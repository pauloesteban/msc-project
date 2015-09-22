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
filename = '/homes/pchilguano/msc_project/dataset/7digital/features/\
feats.h5'
with tables.openFile(filename, 'r') as f:
    features = f.root.x.read()
    #filenames = f.root.filenames.read()

#Pre-processing of spectrograms mean=0 and std=1
n_per_example = np.prod(features.shape[1:-1])
number_of_features = features.shape[-1]
flat_data = features.view()
flat_data.shape = (-1, number_of_features)
scaler = preprocessing.StandardScaler().fit(flat_data)
flat_data = scaler.transform(flat_data)
flat_data.shape = (features.shape[0], -1)

f = file('/homes/pchilguano/msc_project/dataset/7digital/features/\
feats.pkl', 'wb')
cPickle.dump(flat_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
