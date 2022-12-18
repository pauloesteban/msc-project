#!/usr/bin/env python


"""
This is the starting point

@author: paulochiliguano
"""

import os
import requests
import tempfile
import zipfile
import time
import pandas as pd


def _read_song_id_mismatches():
    """ Read song ID mismatches
    """
    url = "http://millionsongdataset.com/sites/default/files/tasteprofile/sid_mismatches.txt"
    response = requests.get(url)
    song_id_mismatches = set()

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(response.content)
        fp.seek(0)

        for line in fp:
            song_id_mismatches.add(line[8:26])
    
    return song_id_mismatches


def _request_taste_profile_subset():
    """ Request the Taste Profile Subset from MSD
    """
    url = "http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip"
    response = requests.get(url)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(response.content)
        fp.seek(0)

        with zipfile.ZipFile(fp) as myzip:
            with myzip.open('train_triplets.txt') as myfile:
                pass


# Unzip Taste Profile subset
def unzip_tasteprofile(zippedfile):
    print("Unzipping Taste Profile subset...")
    uncompressedFilename = os.path.splitext(zippedfile)[0]
    with zipfile.ZipFile(zippedfile) as myzip:
        myzip.extract(uncompressedFilename)
    return uncompressedFilename

# Read songIDs from Million Song Dataset songID-trackID mismatches
def read_songid_mismatches(filename):
    print("Reading songID mismatches...")
    with open(filename, 'r+') as f:
        songIdMismatches = set()
        for line in f:
            songIdMismatches.add(line[8:26])
    return songIdMismatches

def read_available_songid(filename):
    print("Reading available songIDs...")
    with open(filename, 'r+') as f:
        songIdAvailable = set()
        for line in f:
            songIdAvailable.add(line[0:18])
    return songIdAvailable   

def delete_triplets(zippedfile='train_triplets.txt.zip',
                    mismatchesfile='sid_mismatches.txt'):
    """
    Delete triplets with songIDs mismatches and unavailable audio clips from
    7Digital (UK)

    This is applied on Taste Profile subset.

    :type zippedfile: string
    :param zippedfile: filename of the downloaded subset

    :type mismatchesfile: string
    :param mismatchesfile: filename of the downloaded list of mismatches

    """
    tripletsfile = unzip_tasteprofile(zippedfile)
    mismatches = read_songid_mismatches(mismatchesfile)
    print("There are %d songId-trackId mismatches." % len(mismatches))
    availableClips = read_available_songid('7digital/CF_dataset_7digital.txt')
    print("There are %d audio clips available." % len(availableClips))
    cleanfile = os.path.splitext(tripletsfile)[0] + '.h5'
    print("Deleting triplets with mismatches and unavailable songs...")
    for chunk in pd.read_table(
            tripletsfile,
            header=None,
            names=['userId', 'songId', 'numPlays'],
            chunksize=100*len(mismatches),
            ):
        chunk = chunk[~chunk.songId.isin(mismatches)]
        chunk = chunk[chunk.songId.isin(availableClips)]
        #chunk.to_csv(filename_out, mode='a', header=False, index=False)
        chunk.to_hdf(
                cleanfile,
                'triplets',
                mode='a',
                format='table',
                append=True,
                complevel=9,
                complib='zlib',
                fletcher32=True
                )
    # Delete the large text file!
    os.remove(tripletsfile)
    print("Triplets without mismatches saved in %s" % cleanfile)

if __name__ == '__main__':
    #if len(sys.argv) < 1:
        #print("Not enough arguments %s" % sys.argv[0])
        #sys.exit()
    dataset_path = os.path.join(os.path.split(os.getcwd())[0],'dataset')
    os.chdir(dataset_path)
    start_time = time.time()
    delete_triplets()
    elapsed_time = time.time() - start_time
    print("Execution time: %.2f minutes" % (elapsed_time/60))

#a=pd.read_hdf('../train_triplets_clean.h5', 'triplets')

#played_songs = 1000
#df = pd.read_csv(
    #filename_out,
    #delim_whitespace=False,
    #header=None,
    #names=['user','song','plays'])
#df_active = df.groupby('user').filter(lambda x: len(x) > played_songs)
#df_active.to_pickle('../dataset/CF_dataset.pkl')

#f = file('/Users/paulochiliguano/Documents/msc-project/dataset/\
#CF_dataset.pkl', 'wb')
#pickle.dump(df_active, f, protocol=pickle.HIGHEST_PROTOCOL)
#f.close()

# Select most frequent songs
#frequent_songs = 1500
#print("Selecting %d frequent songs..." % frequent_songs)
#counts = df_active['song'].value_counts().head(frequent_songs)
#df_active = df_active.loc[df_active['song'].isin(counts.index), :]
#print("Saving Echonest songID list...")
#filename = '../dataset/CF_dataset_songID.txt'
#with open(filename, 'wb') as f:
    #for item in counts.index.tolist():
       #f.write("%s\n" % item)

#important
#df['user'].value_counts().head(50)

#ddf = df.drop_duplicates(subset = 'song')
#ddf.to_csv('/homes/pchilguano/dataset/train_triplets_songID.csv',
           #columns=['song'],
           #header=False,
           #index=False)