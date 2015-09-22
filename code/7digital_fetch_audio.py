# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:41:44 2015

@author: paulochiliguano
"""


import csv
import time
from pyechonest import song, config #http://echonest.github.io/pyechonest/
import oauth2 as oauth #https://github.com/jasonrubenstein/python_oauth2
import urllib2
import os

# 7digital keys
consumer_key = '7ds28qendsk9'
consumer_secret = 'm5nsktn3hu6x45cy'
consumer = oauth.Consumer(consumer_key, consumer_secret)

# EchoNest key
config.ECHO_NEST_API_KEY="LINDFDUTQZQ781IE8"

# Retrieve audio clips
mp3_folder = '/Users/paulochiliguano/Documents/msc-project/dataset/7digital/\
audio'
filename_echonest = '/Users/paulochiliguano/Documents/msc-project/dataset/\
CF_dataset_songID.txt'
filename_7digital = '/Users/paulochiliguano/Documents/msc-project/dataset/\
7digital/CF_dataset_7digital.txt'
with open(filename_echonest, 'rb') as f, open(filename_7digital, 'wb') as out:
    writer = csv.writer(out, delimiter='\t')	
    '''for i in xrange(1218):
        f.readline()'''
    next = f.readline()
    while next != "":
        try:
            s = song.Song(next)
            #s = song.Song('SOPEXHZ12873FD2AC7')
        #except:        
        except IndexError:
            time.sleep(3)
            print "%s not available" % next[:-1]
            next = f.readline()
        else:
            time.sleep(3)
            try:
                ss_tracks = s.get_tracks('7digital-UK')
            except:
                time.sleep(3)
                print "%s not in UK catalog" % next[:-1]
                next = f.readline()
            else:
                #print(len(ss_tracks))
                if len(ss_tracks) != 0:
                    ss_track = ss_tracks[0]
                    preview_url = ss_track.get('preview_url')	
                    track_id = ss_track.get('id')
                    
                    req = oauth.Request(
                        method="GET",
                        url=preview_url,
                        is_form_encoded=True
                    )
                    req['oauth_timestamp'] = oauth.Request.make_timestamp()
                    req['oauth_nonce'] = oauth.Request.make_nonce()
                    req['country'] = "GB"
                    sig_method = oauth.SignatureMethod_HMAC_SHA1()
                    req.sign_request(sig_method, consumer, token=None)
                    
                    try:
                        response = urllib2.urlopen(req.to_url())
                    except:
                        #time.sleep(16)
                        print "No available preview for %s" % next[:-1]
                        #writer.writerow([next[:-2], 'NA', s.artist_name.encode("utf-8"), s.title.encode("utf-8")])
                    else:                                                
                        print([
                            next[:-1],
                            track_id,
                            s.artist_name,
                            s.title,
                            preview_url
                        ])
                        writer.writerow([
                            next[:-1],
                            track_id,
                            s.artist_name.encode("utf-8"),
                            s.title.encode("utf-8"),
                            preview_url
                        ])
                        mp3_file = os.path.join(mp3_folder, next[:-1]+'.mp3')
                        with open(mp3_file, 'wb') as songfile:
                            songfile.write(response.read())
                    time.sleep(16)
                next = f.readline()	
        