# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 01:41:18 2015

@author: paulochiliguano
"""

import pandas as pd
import cPickle as pickle
#import random
import numpy as np

# Normalise user's play count
def normalise_user_play_count(user_df, cv_value = 0.5, max_score=5, similar_score=3):
    
    '''cv: Coefficient of variation, normalised measure of dispersion of a
       probability distribution'''
    cv = user_df.plays.std() / user_df.plays.mean()

    if cv <= 0.5:
        '''Homogenous listening habits'''
        user_df.plays = similar_score
        print "Homogeneous"
        #for songID, play_count in user[userID].items():
            #user[userID][songID] = 3
    else:
        '''Complementary cumulative distribution'''
        user_df = user_df.sort(columns='plays', ascending=False)
        user_df['ccs'] = 1 - user_df.plays.cumsum() / float(user_df.plays.sum())
        
        user_df.loc[user_df.ccs >= 0.8, 'plays'] = 5
        user_df.loc[(user_df.ccs < 0.8) & (user_df.ccs >= 0.6), 'plays'] = 4
        user_df.loc[(user_df.ccs < 0.6) & (user_df.ccs >= 0.4), 'plays'] = 3
        user_df.loc[(user_df.ccs < 0.4) & (user_df.ccs >= 0.2), 'plays'] = 2
        user_df.loc[user_df.ccs < 0.2, 'plays'] = 1
        
        user_df = user_df.drop('ccs', 1)
        #song_play_count_q = pd.cut(
            #user_df["plays"],
            #max_score,
            #labels=False
        #) + 1
        #user_df.plays = song_play_count_q
        #user[userID] = song_play_count.set_index('songID')['play_count'].to_dict()
    return user_df
    #for userID in user:
        #song_play_count = pd.DataFrame(
            #user[userID].items(),
            #columns=["songID", "play_count"]
        #)

# User-item data frame
users_df = pd.read_pickle('/Users/paulochiliguano/Documents/msc-project/\
dataset/CF_dataset.pkl')

# Normalise users' rating
# users_norm_df = pd.DataFrame()
#for k, v in users_df.groupby("user"):
    #users_norm_df = users_norm_df.append(normalise_user_play_count(v))

# SongIDs of downloaded audio clips
filename = '/Users/paulochiliguano/Documents/msc-project/dataset/7digital/\
CF_dataset_7digital.txt'
with open(filename, 'rb') as f:
    available_clips = [line.strip().split('\t')[0] for line in f]

# Ground truth with available tracks
#users_ground_truth_df = users_norm_df[users_norm_df.song.isin(available_clips)]
users_df = users_df[users_df.song.isin(available_clips)]

# Users with more than 50 ratings
#users_df = users_df.groupby('user').filter(lambda x: len(x) >= 50)

# Normalise users' rating
users_norm_df = pd.DataFrame()
for k, v in users_df.groupby("user"):
    norm = normalise_user_play_count(v)
    users_norm_df = users_norm_df.append(norm)
#    counts = norm['plays'].value_counts()
#    if counts[counts.index == 5].values > 0:
#        users_norm_df = users_norm_df.append(norm)
    
#for k, v in users_norm_df.groupby('user'):
#    counts = v['plays'].value_counts()
#    df = v.loc[v['plays'].isin(counts[counts >= 5].index), :]
#    print df

trial = 10
users_train = []
users_test = []
#highest_rating = [4, 5]
#lowest_rating = [1, 2, 3]
for i in range(trial):
    test_df = pd.DataFrame()
    train_df = pd.DataFrame()
    for k, v in users_norm_df.groupby("user"):
#        likes = v.loc[v['plays'].isin(highest_rating)]
#        dislikes = v.loc[v['plays'].isin(lowest_rating)]
#        test_like_index = np.random.choice(
#            likes.index,
#            1,
#            replace=False
#        )
#        test_dislike_index = np.random.choice(
#            dislikes.index,
#            1,
#            replace=False
#        )
#        test_index = np.append(test_like_index, test_dislike_index)
#        test_index = test_like_index
        test_index = np.random.choice(
            v.index,
            int(len(v.index) / 5),
            replace=False
        )

        test_df = test_df.append(v.loc[test_index])
        train_df = train_df.append(v.loc[~v.index.isin(test_index)])
    
    users_train.append([])
    users_train[i] = {}
    for k, v in train_df.groupby("user"):
        users_train[i][k] = {
            x: y["plays"].values[0] for x, y in v.groupby("song")
        }
    
    users_test.append([])
    users_test[i] = {}
    for k, v in test_df.groupby("user"):
        users_test[i][k] = {
            x: y["plays"].values[0] for x, y in v.groupby("song")
        }

# Save training and test sets
f = file('/Users/paulochiliguano/Documents/msc-project/dataset/\
cross_validation.pkl', 'wb')
pickle.dump((users_train, users_test), f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()
