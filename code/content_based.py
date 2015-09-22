# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:58:19 2015

@author: paulochiliguano
"""

import cPickle as pickle
from math import sqrt
import numpy as np
import pandas as pd
import time

# Item-vector dictionary
f = file('/Users/paulochiliguano/Documents/msc-project/dataset/\
genre_classification/genre_prob.pkl', 'rb')
song_library = pickle.load(f)
f.close()

# Normalisation
#test = []
#for k, v in song_library.iteritems():
#    test.append(v)
#test = np.array(test)
#test_median = np.median(test, axis=0)
#test_abs = abs(test - test_median)
#test_asd = test_abs.sum(axis=0) / test.shape[0]
#for k, v in song_library.iteritems():
#    modified_standard_score = (np.array(v) - test_median) / test_asd
#    song_library[k] = modified_standard_score.tolist()

# Load training and test data
f = file('/Users/paulochiliguano/Documents/msc-project/dataset/\
cross_validation.pkl', 'rb')
users_train, users_test = pickle.load(f)
f.close()

# Adjusted Cosine Similarity
def adj_cos_sim(vector_i, vector_j):
    avrg_w_i = (float(sum(vector_i)) / len(vector_i))
    avrg_w_j = (float(sum(vector_j)) / len(vector_j))
    num = sum(map(
        lambda w_i, w_j: (w_i - avrg_w_i) * (w_j - avrg_w_j),
        vector_i,
        vector_j)
    )
    dem1 = sum(map(lambda w_i: (w_i - avrg_w_i) ** 2, vector_i))
    dem2 = sum(map(lambda w_j: (w_j - avrg_w_j) ** 2, vector_j))
    return num / (sqrt(dem1) * sqrt(dem2))

def build_model_cb(train_data, k=30):
    a = []
    for user, info in train_data.iteritems():
        a.extend([i for i in info])
    songIDs = list(set(a))       
    #other_songs = song_library.keys()
    
    similarity_matrix = {}
    for song in songIDs:
        similarities = []
        for other in songIDs:
            if other != song:
                sim = adj_cos_sim(song_library[song], song_library[other])
                similarities.append((sim, other))
        similarities.sort(reverse=True)
        similarity_matrix[song] = similarities[0:k]
    
    return similarity_matrix
        #similarity_rows[song] = {t[1]: t[0] for t in similarities}

def top_n(sim_matrix, user, song_rating, rating_threshold=2, N=10): 
    candidate = pd.DataFrame()
    entries = song_rating.keys()
    for song, rating in song_rating.iteritems():
        if rating > rating_threshold:
            sim = sim_matrix[song]
            list_a = [k for v, k in sim]
            raw = [v for v, k in sim]
            sim_norm = [float(i)/max(raw) for i in raw]
            the_dict = dict(zip(list_a, sim_norm))
            for key in entries:
                if key in the_dict:
                    del the_dict[key]
            candidate_aux = pd.DataFrame(
                the_dict.items(),
                columns=['song', 'similarity']
            )
            candidate = candidate.append(candidate_aux, ignore_index=True)
            #tuples = [(k,v) for k,v in the_dict.iteritems()]
            #candidate.extend(tuples)
    topN = candidate.groupby('song')['similarity'].sum()
    topN.sort(1, ascending=False)
    
    return list(topN.head(N).keys())

def evaluate_cb(topN, test_data, rating_threshold=2):    
    
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    for user, song_rating in test_data.iteritems():
        entries = topN[user]
        for song, rating in song_rating.iteritems():
            if song in entries:
                if rating > rating_threshold:
                    tp += 1
                elif rating <= rating_threshold:
                    fp += 1   
            else:
                if rating > rating_threshold:
                    fn += 1
                elif rating <= rating_threshold:
                    tn += 1
    #print tp, fp, fn, tn
    if tp != 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        F1 = 0
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    return precision, recall, F1, accuracy

p = np.array([])
f = np.array([])
r = np.array([])
a = np.array([])

for i in range(len(users_train)):
 
    start_time = time.time()
    sim_matrix = build_model_cb(users_train[i])
    
    topN = {}
    for user, song_rating in users_train[i].iteritems():
        topN[user] = top_n(sim_matrix, user, song_rating, rating_threshold=2, N=20)
    elapsed_time = time.time() - start_time
    print 'Training execution time: %.3f seconds' % elapsed_time
        
    pi, ri, fi, ai = evaluate_cb(topN, users_test[i])
    
    p = np.append(p, pi)
    r = np.append(r, ri)
    f = np.append(f, fi)
    a = np.append(a, ai)
    
print "Precision = %f3 ± %f3" % (p.mean(), p.std())
print "Recall = %f3 ± %f3" % (r.mean(), r.std())
print "F1 = %f3 ± %f3" % (f.mean(), f.std())
print "Accuracy = %f3 ± %f3" % (a.mean(), a.std())

#        set_C = {t[0]: t[1] for t in candidate}
#        for song in set_C:
#            sim = sim_matrix[song]
#            the_dict = {t[1]: t[0] for t in sim}
#            for key in entries:
#                if key in the_dict:
#                    the_dict[key]
