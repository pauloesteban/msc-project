# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:37:43 2015

@author: Paulo
"""


import pandas as pd
import csv
import itertools
import numpy as np
import wmf

# Read songID of downloaded audio clips
filename = '/homes/pchilguano/dataset/small_dataset.txt'
with open(filename, 'rb') as f:
    song_set = set()
    for line in f:
        songID = line.strip().split(',')[0]
        song_set.add(songID)
    #available = list(csv.reader(f))
    #chain1 = list(itertools.chain(*available))
    
# Sparse user-item matrix
result = pd.DataFrame()
filename = '/homes/pchilguano/dataset/train_triplets_wo_mismatches.csv'
for chunk in pd.read_csv(
        filename,
        low_memory=False,
        delim_whitespace=False, 
        chunksize=10000,
        names=['user', 'song', 'plays'],
        header=None):
    chunk = chunk[chunk.song.isin(song_set)]
    result = result.append(chunk, ignore_index=True)
    #result = result.append(chunk.pivot(index='user', columns='song', values='plays'))
    #print (result.shape)


cnames = result.set_index('user').T.to_dict().keys()
final = {}
for a in cnames:
    final[a] ={result.set_index('user').T.to_dict()[a]['song']: result.set_index('user').T.to_dict()[a]['plays']}


dict((k, v.dropna().to_dict()) for k, v in pd.compat.iteritems(result))

sresult = result.to_sparse()
sresult.to_pickle('/homes/pchilguano/dataset/taste_profile_sparse.pkl')

# Weight Matrix Factorization
B = np.load("test_matrix.pkl")
S = wmf.log_surplus_confidence_matrix(B, alpha=2.0, epsilon=1e-6)
U, V = wmf.factorize(S, num_factors=40, lambda_reg=1e-5, num_iterations=2, init_std=0.01, verbose=True, dtype='float32', recompute_factors=wmf.recompute_factors_bias)