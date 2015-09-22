# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:42:09 2015

@author: paulochiliguano
"""


from math import log, sqrt
import numpy as np
import pandas as pd
import cPickle as pickle
import time

# Item-vector dictionary
f = file('/Users/paulochiliguano/Documents/msc-project/dataset/\
genre_classification/genre_prob.pkl', 'rb')
song_library = pickle.load(f)
f.close()

# Load training and test data
f = file('/Users/paulochiliguano/Documents/msc-project/dataset/\
cross_validation.pkl', 'rb')
users_train, users_test = pickle.load(f)
f.close()

# Cosine Similarity
def cosine_similarity(vector1, vector2):
    dot_product = sum(map(lambda x, y: x * y, vector1, vector2))
    length_x = sqrt(sum(map(lambda x: x ** 2, vector1)))
    length_y = sqrt(sum(map(lambda y: y ** 2, vector2)))
    return dot_product / (length_x * length_y)

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

# Fitness function for EDA
def Fitness(profile_u, user_subset):
    fitness_value = 0
    for songID, score in user_subset.iteritems():
        #print cosine_similarity(profile_u, song_library[songID])
        sim = cosine_similarity(profile_u, song_library[songID])
        if sim <= 0:
            fitness_value += -708
            #math.log(sys.float_info.min)
        else:
            fitness_value += log(score * sim)
        #fitness_value += log(score * manhattan(profile, song_library[songID]))
        #fitness_value += score * cosine_similarity(profile, song_library[songID])
    return fitness_value

def users_likes_subset(users, rating_threshold=2):
    # Subset of most-liked items
    users_subset = {}
    for userID, songs in users.iteritems():
        scores_above_threshold = {
            songID: score for songID, score in songs.iteritems() if score > rating_threshold
        }
        users_subset[userID]= scores_above_threshold
        
    #for songID, score in songs.iteritems():
        #print score >0
        #if score > 0:
            #print {userID: {songID: score}}

    #{k: v for k, v in users.iteritems() for i,j in v.iteritems() if j > 0}
        
    return users_subset

def eda_train(users_subset, max_gen=250):
    # TRAINING
    num_features = len(song_library.values()[0])
    # Given parameters for EDA
    population_size = len(users_subset)
    fraction_of_population = int(round(0.5 * population_size))
    
    # Ku set
    weights = list(np.linspace(0.1, 0.9))
    tags = [
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock'
    ]
    for i, j in enumerate(tags):
        tags[i] = i
    list_a = np.tile(weights, num_features)
    list_b = np.repeat(tags, len(weights))
    Ku = zip(list_b, list_a)
    Ku_np = np.array(Ku, dtype=('int, float'))
    
    # Generate initial population
    np.random.seed(12345)
    profile_u = {}
    profile_aux = {}
    for userID in users_subset:
        a = np.random.choice(
            Ku_np,
            num_features,
        ).tolist()
        #a = sorted(a, key=lambda student: student[1], reverse=True)
        b = {t[0]: t[1] for t in a}
        feature_v = list(np.zeros(num_features))
        for k, v in b.iteritems():
            feature_v[k] = v
        profile_u[userID] = feature_v
        profile_aux[userID] = [(k, v) for k, v in b.iteritems()]

    fitnesses = []
    
    generation = 0
    while generation < max_gen:
        # Compute fitness values
        users_fitness = {}
        for userID in profile_u:
            users_fitness[userID] = Fitness(
                profile_u[userID],
                users_subset[userID]
            )
        users_fitness_df = pd.DataFrame(
            users_fitness.items(),
            columns=["userID", "fitness"]
        )
        fitnesses.append(users_fitness_df.fitness.values.tolist())
    
        # Selection of best individuals based on fitness values
        #best_individuals = {}
        users_fitness_df = users_fitness_df.sort(columns='fitness')
        M_sel = users_fitness_df.tail(fraction_of_population)
        M_sel_dict = M_sel.set_index('userID')['fitness'].to_dict()
        #for userID in M_sel_dict:
            #best_individuals[userID] = profile_u[userID]
        
        Xs = []
        for userID in M_sel_dict:
            Xs.extend(profile_aux[userID])
        
        # Update probability model
        p = []
        for i in Ku:
            p.append(float(Xs.count(i)) / fraction_of_population)
        
        # Sample new population
        profile_u = {}
        profile_aux = {}
        for userID in users_subset:
            a = np.random.choice(
                Ku_np,
                num_features,
                p
            ).tolist()
            #a = sorted(a, key=lambda student: student[1], reverse=True)
            b = {t[0]: t[1] for t in a}
            feature_v = list(np.zeros(num_features))
            for k, v in b.iteritems():
                feature_v[k] = v
            profile_u[userID] = feature_v
            profile_aux[userID] = [(k, v) for k, v in b.iteritems()]

        generation += 1
    
    return profile_u, p, np.array(fitnesses)

# Similarity matrix
def cb_similarity(profileID, profile_data, test_data, N):
    
    a = []
    for user, info in test_data.iteritems():
        a.extend([i for i in info])
    songIDs = list(set(a))    
    
    ''' Content-based: Similarity matrix '''
    similarity = []
    #keys_a = train_data[profileID].keys()
    for songID in songIDs:
        sim = adj_cos_sim(profile_data, song_library[songID])
        similarity.append((sim, songID))
    ''' Top-N recommendation '''
    similarity.sort(reverse=True)
    if len(similarity) > N:
        similarity = similarity[0:N]
        
    #sim_matrix[userID] = {t[1]: t[0] for t in similarity}
    return {t[1]: t[0] for t in similarity}

def evaluate_eda(
    profiles,
    test_data,
    N=10,
    rating_threshold=2,
    EDA_treshold=0.5):    
    
    ''' Evaluation '''
    
    sim_matrix = {}
    for userID, features in profiles.iteritems():
        sim_matrix[userID] = cb_similarity(userID, features, test_data, N)
    
    # Content-Based: Evaluation
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    
    for user, song_rating in test_data.iteritems():
        entries = sim_matrix[user]
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
    
#    for userID, songID_sim in sim_matrix.iteritems():
#        for songID, sim_value in songID_sim.iteritems():
#            score = test_data[userID][songID]
#            if score > rating_threshold and sim_value >= EDA_treshold:
#                tp += 1
#            elif score <= rating_threshold and sim_value >= EDA_treshold:
#                fp += 1
#            elif score > rating_threshold and sim_value < EDA_treshold:
#                fn += 1
#            elif score <= rating_threshold and sim_value < EDA_treshold:
#                tn += 1
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
    profile_u, prob, fffitness = eda_train(users_likes_subset(users_train[i]))
    elapsed_time = time.time() - start_time
    print 'Training execution time: %.3f seconds' % elapsed_time
    
    pi, ri, fi, ai = evaluate_eda(profile_u, users_test[i], N=20)
    p = np.append(p, pi)
    r = np.append(r, ri)
    f = np.append(f, fi)
    a = np.append(a, ai)

print "Precision = %f3 ± %f3" % (p.mean(), p.std())
print "Recall = %f3 ± %f3" % (r.mean(), r.std())
print "F1 = %f3 ± %f3" % (f.mean(), f.std())
print "Accuracy = %f3 ± %f3" % (a.mean(), a.std())
