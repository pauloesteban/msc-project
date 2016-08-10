#!/usr/bin/env python

import time
import pandas as pd
#import sys
import os
#import cPickle as pickle

os.chdir('/Users/paulochiliguano/Documents/QMUL/msc-project/dataset/taste_profile')

# Read songIDs from Million Song Dataset songID-trackID mismatches
def read_songid_mismatches(filename):
    #start_time = time.time()
    with open(filename, 'r+') as f:
        songIdMismatches = set()
        for line in f:
            songIdMismatches.add(line[8:26])
    return songIdMismatches

# Delete rows with songIDs mismatches from Taste Profile Subset
def delete_mismatch_triplets(tripletsfile, mismatchesfile):
    print("Reading songID mismatches...")
    mismatches = read_songid_mismatches(mismatchesfile)
    print("There are %d songId-trackId mismatches" % len(mismatches))
    print("Reading Taste Profile subset...")
    for chunk in pd.read_table(
            tripletsfile,
            header=None,
            names=['userId', 'songId', 'numPlays'],
            chunksize=10*len(mismatches),
            ):
        chunk = chunk[~chunk.songId.isin(mismatches)]
        #chunk.to_csv(filename_out, mode='a', header=False, index=False)
        chunk.to_hdf('../train_triplets_clean.h5','triplets', mode='a', format='table', append=True, complevel=9, complib='zlib')
#elapsed_time = time.time() - start_time




# Select most active users
start_time = time.time()
played_songs = 1000
print("Reading (filtered) Taste Profile subset...")
df = pd.read_csv(
    filename_out,
    delim_whitespace=False,
    header=None,
    names=['user','song','plays'])
print("Selecting most active users (> %d ratings)..." % played_songs)
df_active = df.groupby('user').filter(lambda x: len(x) > played_songs)

print("Saving user-item matrix as dataframe...")
df_active.to_pickle('../dataset/CF_dataset.pkl')

'''f = file('/Users/paulochiliguano/Documents/msc-project/dataset/\
CF_dataset.pkl', 'wb')
pickle.dump(df_active, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()'''

# Select most frequent songs
frequent_songs = 1500
print("Selecting %d frequent songs..." % frequent_songs)
counts = df_active['song'].value_counts().head(frequent_songs)
#df_active = df_active.loc[df_active['song'].isin(counts.index), :]
print("Saving Echonest songID list...")
filename = '../dataset/CF_dataset_songID.txt'
with open(filename, 'wb') as f:
    for item in counts.index.tolist():
        f.write("%s\n" % item)
elapsed_time = time.time() - start_time
print("Execution time: %.3f seconds" % elapsed_time)

'''
#important
#df['user'].value_counts().head(50)

ddf = df.drop_duplicates(subset = 'song')
ddf.to_csv('/homes/pchilguano/dataset/train_triplets_songID.csv',columns=['song'], header=False, index=False)



with open('/homes/pchilguano/dataset/sid_mismatches_songID.txt', 'rb') as input1, open('/homes/pchilguano/dataset/train_triplets_songID.csv', 'rb') as input2, open('/homes/pchilguano/dataset/echonest_songID.txt', 'wb') as myfile:
    l1 = list(csv.reader(input1))
    chain1 = list(itertools.chain(*l1))
    l2 = list(csv.reader(input2))
    chain2 = list(itertools.chain(*l2))
    l3 = set(chain2) - set(chain1)
    wr = csv.writer(myfile, delimiter=',')
    for item in l3:
        wr.writerow([item])

# Save Taste Profile dataset without SongID mismatches
mdf = df[df.song.isin(l3)]
mdf.to_csv('/homes/pchilguano/dataset/train_triplets_wo_mismatches.csv', header=False, index=False)

result = pd.DataFrame()
for chunk in pd.read_csv('/homes/pchilguano/dataset/train_triplets_wo_mismatches.csv', low_memory = False, delim_whitespace=False, chunksize=10000, names=['user','song','plays'], header=None):
    chunk = chunk[chunk.song.isin(l3)]    
    result = result.append(chunk.pivot(index='user', columns='song', values='plays')    
    , ignore_index=True)
    print (result.shape)
'''

#if __name__ == '__main__':
    #if len(sys.argv) < 2:
        #print("Not enough arguments %s" % sys.argv[0])
        #sys.exit()
    #taste_profile_path = os.path.abspath(sys.argv[1])
    #os.chdir(taste_profile_path)
delete_mismatch_triplets('train_triplets.txt','sid_mismatches.txt')
a=pd.read_hdf('../train_triplets_clean.h5', 'triplets')