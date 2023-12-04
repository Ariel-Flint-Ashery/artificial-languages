# -*- coding: utf-8 -*-
"""
LEGACY CODE!!

Created on Sun Nov  5 15:20:24 2023

@author: ariel

This is a playground for testing the effect of the size of the learning bottleneck on agents who learn from a sequence produced by a single ancestor, speaking a single language. The learner observes the entire sequence.
"""
#%%
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import numpy as np
from scipy.stats import beta
import math
from math import log, log2
from scipy.special import logsumexp
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, permutations, combinations
from functools import lru_cache
import multiprocessing
import time
#%%
def normalize_probs(probs):
    total = sum(probs) #calculates the summed probabilities
    normedprobs = []
    for p in probs:
        normedprobs.append(p / total) 
    return normedprobs

def normalize_logprobs(logprobs):
    logtotal = logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs = []
    for logp in logprobs:
        normedlogs.append(logp - logtotal) #normalise - subtracting in the log domain equivalent to divising in the normal domain
    return normedlogs

def log_roulette_wheel(normedlogs):
    r=log(random.random()) #generate a random number in [0,1), then convert to log
    accumulator = normedlogs[0]
    for i in range(len(normedlogs)):
        if r < accumulator:
            return i
        accumulator = logsumexp([accumulator, normedlogs[i + 1]])
def characterise_language(data):
    # text is a list of (signal,meaning) pairs.
    #check for degenerate language
    if len(set([pair[1] for pair in data]))==1:
        return 0
    
    degenerate_list = list(sum(data, ()))
    if len(degenerate_list) != len(set(degenerate_list)):
        return 2 #partially degenerate
    
    #compositional or holistic?

    for i in range(len(data)-1):
        pair = data[i]
        sig_match = [x for j in range(len(pair[0])) for x in data[i+1:] if x[0][j]==pair[0][j]]
        mean_match = [x for j in range(len(pair[1])) for x in sig_match if x[1][j]==pair[1][j]]
        if len(mean_match)==0:
            return 1 #holistic
                
    return 3 #compositional

def generate_language(vocabulary_size, word_size):
    alphabet = list(map(chr, range(97, 123))) #default english alphabet
    vocabulary = alphabet[:vocabulary_size]
    signals =  ["".join(item) for item in
     list(product(vocabulary, repeat=word_size))]
    #note: there are as many meanings as signals to account for holistic languages
    meaning_chr_set = [] #?
    for pos in range(word_size):
        meaning_chr_set.append(list(np.array(range(vocabulary_size)) + pos*vocabulary_size))
    meanings = [''.join(str(y) for y in x) for x in product(*meaning_chr_set)]

    #generate all possible meaning, signal pairs
    all_pairs = list(product(meanings, signals))
    
    #separate into meaning groupings
    groupings = []
    for meaning in meanings:
        meaning_group = []
        for pair in all_pairs:
            if meaning in pair:
                meaning_group.append(pair)
        groupings.append(meaning_group)

    #create all possible language forms (== number of meanings)
    possible_languages = [list(item) for item in list(product(*groupings))]

    return possible_languages
    
    
#%% INITIALISE
meanings = ['02', '03', '12', '13']
signals = ['aa', 'ab', 'ba', 'bb']
possible_languages = [[('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'aa'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ab'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'ba'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'aa'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ab'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'ba'), ('12', 'bb'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'aa'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ab'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'ba'), ('13', 'bb')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'aa')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ab')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'ba')], [('02', 'bb'), ('03', 'bb'), ('12', 'bb'), ('13', 'bb')]]
language_types = [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0]

#%%

language_type = [characterise_language(data) for data in possible_languages]
for index, (first, second) in enumerate(zip(language_types, language_type)):
    if first != second:
        print(index, second, first, possible_languages[index])
        
#%%
hspace, step = np.linspace(0,1, len(possible_languages), endpoint=False, retstep = True)
hspace+=step*0.5

def get_uniform_logprior():
    prior = [0]*len(possible_languages)
    for t in range(4):
        type_prob = 0.25/language_type.count(t)
        indices=np.where(np.array(language_type) == t)[0]
        for index in indices:
            prior[index]=type_prob
    return np.log(prior).tolist()

def get_logprior_type_dist(prior):
    prior_type = []
    for t in range(4):
        indices = np.where(np.array(language_type) == t)[0]
        prior_type.append(logsumexp([prior[index] for index in indices]))
    return prior_type
        
def bias_prior(prior, bias):
    for t in range(4):
        if bias[t] <= 0:
            continue
        indices = np.where(np.array(language_type) == t)[0]
        for index in indices:
            prior[index]=logsumexp([prior[index],log(bias[t])])
    return normalize_logprobs(prior)

@lru_cache
def code_length(signals):
    code_length = 0
    for s in signals:
        code_length -= log2(signals.count(s)/len(signals))
    return code_length

def non_normed_prior(language):
    signals = ''.join([s for _, s in language])
    return 2 ** -code_length(signals)

def get_compressible_prior():
    priors = [non_normed_prior(language) for language in possible_languages]
    priors = normalize_probs(priors)
    priors = [log(prior) for prior in priors]
    return priors

def get_beta_logprior(alpha):
    logprior = []
    for h in hspace:
        logprior.append(beta.logpdf(h, alpha, alpha)) 
    return normalize_logprobs(logprior) 

def loglikelihoods(data, posterior):
    #likelihood of generating a sequence of (m,s) pairs for each language.

    in_language = log(1 - error_probability)
    out_of_language = log(error_probability / (len(signals) - 1))
    #meaning_prob = log(1/len(meanings)) #probability of generating a given meaning. is this necessary?
    # loglikelihoods = []
    # for d in data:
    #     logprob = []
    #     for language in possible_languages:
    #         if d in language:
    #             # a=[s for _, s in language].count(d[1])
    #             # express_term = log((1/a)**expressivity)
    #             logprob.append(in_language)#+express_term) #+meaning_prob)
    #         else: 
    #             logprob.append(out_of_language) #+meaning_prob)
    #     loglikelihoods.append(logprob)
    # sequence_likelihood = [sum(i) for i in zip(*loglikelihoods)] #do I need to normalize here?
    sequence_likelihood = np.zeros(len(possible_languages))
    for d in data:
        for i in range(len(possible_languages)):
            if d in possible_languages[i]:
                sequence_likelihood[i]+=in_language
            else:
                sequence_likelihood[i]+=out_of_language
    return sequence_likelihood.tolist()

def update_posterior(data, posterior, prior):
    sequence_likelihood = loglikelihoods(data, posterior)
    new_posterior = normalize_logprobs([sum(i) for i in zip(*[sequence_likelihood, prior])]) #here, we would swap 'prior' with a Dirichlet process prior
    return new_posterior
 
def sample(posterior, MAP = False):
    if MAP==False:
        selected_index = log_roulette_wheel(posterior)
    else:
        probs = np.exp(posterior)
        max_signal_prob = max(probs)
        selected_index = random.choice([i for i, v in enumerate(probs.tolist()) if v == max_signal_prob])
    return possible_languages[selected_index]

def produce(posterior, bottleneck,  expressivity=0, MAP = False, initial_language=False):
    # randomly choose meaning to express
    intended_meanings = random.choices(meanings, k=bottleneck)

    # select speaker language

    if initial_language==True:
        # set initial language to a random holistic language
        indices = np.where(np.array(language_type)==1)[0]
        language = random.choice([possible_languages[index] for index in indices])
    
    if expressivity==0:
        language = sample(posterior, MAP=MAP)

    if expressivity != 0:
        # weight each language by how easy it is to express a given meaning
        new_posterior = posterior.copy()

        meaning_dict = {m: intended_meanings.count(m) for m in set(intended_meanings)}

        for meaning in meaning_dict.keys():
            for signal in signals:
                for i in range(len(possible_languages)):
                    if (meaning, signal) in possible_languages[i]:
                        a=list(sum(possible_languages[i], ())).count(signal) # add ambiguity term
                        express_term = log((1/a)**expressivity)
                        new_posterior[i]+=express_term * meaning_dict[meaning]
        language = sample(normalize_logprobs(new_posterior), MAP = MAP)

    # generate data

    data=[]
    for meaning in intended_meanings:
        for m, s in language:
            if m == meaning:
                signal = s # find the signal that is mapped to the meaning 
    
        if random.random() < error_probability: # add the occasional mistake
            other_signals = []
            for other_signal in signals:
                if other_signal != signal:
                    other_signals.append(other_signal) # make a list of all the "wrong" signals
            data.append((meaning, random.choice(other_signals))) # pick one of them
        
        data.append((meaning, signal))
    
    return data, language

def iterate(prior, bottleneck, generations, expressivity=0, MAP = False):
    # initialise posterior
    # posterior = np.zeros(len(possible_languages))
    # posterior[indices] = 1/len(indices)
    posterior = prior.copy()
    
    # initalise data collection
    data = produce(posterior, bottleneck=bottleneck, expressivity=expressivity, MAP = MAP, initial_language=True)#[produce(initial_language) for i in range(bottleneck)]
    language_accumulator = []
    posterior_accumulator = []
    data_accumulator = []

    # iterate across generations
    for generation in range(generations):
        posterior = update_posterior(data, posterior, prior)
        data, language = produce(posterior, bottleneck=bottleneck, expressivity=expressivity, MAP = MAP) #[produce(language) for i in range(bottleneck)]
        language_accumulator.append(language)
        posterior_accumulator.append(posterior)
        data_accumulator.append(data)

    return language_accumulator, posterior_accumulator, data_accumulator

def iterate_stats(language, posterior):
    iterated_lang_type = [characterise_language(lang) for lang in language]
    #avg_posterior = np.mean(np.array(posterior), axis=0)
    posterior_type_evolution = []
    for t in range(4):
        indices = np.where(np.array(language_type) == t)[0]
        posterior_type_evolution.append([logsumexp([post[index] for index in indices]) for post in posterior])
        #posterior_type_evolution.append([post[index] for index in indices for post in posterior])

    return iterated_lang_type, posterior_type_evolution #avg_posterior

def iterate_population(prior, n_pop, bottleneck, generations, expressivity=0, MAP = False):
    posterior_accumulator = []
    for agent in range(n_pop):
        lang_type_agent, posterior_agent, _ = iterate(prior, bottleneck=bottleneck, generations=generations, expressivity=expressivity, MAP=MAP)
        _, posterior_type_evolution = iterate_stats(lang_type_agent, posterior_agent)
        posterior_accumulator.append(posterior_type_evolution)
    return posterior_accumulator

def analyse_population_type(posterior_accumulator):
    proportions =[np.mean(np.exp([agent[t] for agent in posterior_accumulator]), axis=0) for t in range(4)]
    return proportions
#%% CONSTANTS
error_probability = 0.05
#bottleneck = 20
bottlerange = np.arange(5,300,25)
generations = 100
n_pop = 20
expressivity = 0 #>=0 The higher it is, the more expressive the langauge
MAP = False
colors = ['r', 'g', 'blue', 'orange']
labels = ['Degenerate', 'Holistic', 'Partially Degenerate', 'Compositional']
#%% PREPARE BETA PRIOR
prior = get_beta_logprior(1)
plt.plot(np.exp(prior))

prior_type = get_logprior_type_dist(prior)
plt.bar(range(4), np.exp(prior_type), color = colors, tick_label = labels)
#%% PREPARE COMPRESSIBLE PRIOR
prior = get_compressible_prior()
plt.plot(np.exp(prior))
plt.show()

prior_type = get_logprior_type_dist(prior)
plt.bar(range(4), np.exp(prior_type), color = colors, tick_label = labels)
plt.show()
#%% PREPARE UNIFORM PRIOR
prior = get_uniform_logprior()
plt.plot(np.exp(prior))
plt.show()

prior_type = get_logprior_type_dist(prior)
plt.bar(range(4), np.exp(prior_type), color = colors, tick_label = labels)
plt.show()
#%% PREPARE MANUAL BIASED PRIOR --make sure to initalise prior first
prior=bias_prior(prior, [0, 0.1, 0, 0.5])
plt.plot(np.exp(prior))
plt.show()

prior_type = get_logprior_type_dist(prior)
plt.bar(range(4), np.exp(prior_type), color = colors, tick_label = labels)
plt.show()

#%% BEGIN SIMULATIONS ---POPULATION OF AGENTS---
if np.sqrt(len(bottlerange))-round(np.sqrt(len(bottlerange))) < 0:
    nrow = math.ceil(np.sqrt(len(bottlerange)))
    ncol = nrow
    #if np.sqrt(pop_count)-round(np.sqrt(pop_count)) >= 0:
else:
    nrow = math.floor(np.sqrt(len(bottlerange)))
    ncol = math.ceil(np.sqrt(len(bottlerange)))
    
fig, axs = plt.subplots(nrow, ncol)
axs = axs.flatten()
proportion_accumulator = []
for ax, bottle in tqdm(zip(axs,bottlerange)):
    population_sim = iterate_population(prior, n_pop = n_pop, bottleneck = bottle, generations = generations, expressivity = expressivity, MAP=MAP)
    proportions = analyse_population_type(population_sim)
    proportion_accumulator.append([proportions[t][-1] for t in range(4)])
    for t in range(4):
        ax.plot(range(generations), proportions[t], color = colors[t], label = labels[t])
        ax.set_xlabel('Generations')
        ax.set_ylabel('Proportion')
        ax.set_title('b=%s' % (bottle))
if len(axs)>len(bottlerange):
    fig.delaxes(axs[-1])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles=handles,ncol=len(labels),loc="lower center", bbox_to_anchor=(0.5,-0.07), fontsize = 10)#, facecolor = background, edgecolor = background)
plt.tight_layout()
plt.show()

#%% DISPLAY EVOLUTION
for t in range(4):
    plt.plot(bottlerange, [prop[t] for prop in proportion_accumulator], color = colors[t], label = labels[t])
plt.legend()
plt.xlabel('Bottleneck Size')
plt.ylabel('Final Proportion (%s gens.)' % (generations))
plt.show()
#%%
fig, axs = plt.subplots(nrow, ncol)
axs = axs.flatten()
for i in range(len(bottlerange)):
        axs[i].bar(range(4), proportion_accumulator[i], color=colors)
        axs[i].set_ylabel('Proportion')
        axs[i].set_xticklabels([])
        axs[i].set_title('b=%s' % (bottlerange[i]))
if len(axs)>len(bottlerange):
    fig.delaxes(axs[-1])
    
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles=handles,ncol=len(labels),loc="lower center", bbox_to_anchor=(0.5,-0.07), fontsize = 10)#, facecolor = background, edgecolor = background)
plt.tight_layout()
plt.show()
# %% PARALLEL SIMULATION
def simulation(b):
    lang_type_agent, posterior_agent, _ = iterate(prior, bottleneck=b, generations=generations, expressivity=expressivity, MAP=MAP)
    _, posterior_type_evolution = iterate_stats(lang_type_agent, posterior_agent)
    return posterior_type_evolution

#%% parallelise
#start = time.perf_counter()
if __name__ == "__main__":
    print("""
          -----------------------------
          
              STARTING MULTIPROCESS
          
          -----------------------------
          """)
    pool = multiprocessing.Pool(4) #multiprocessing.cpu_count() - 1) #uses all available processors
    sim = pool.map(simulation, [(b) for b in [5] for _ in range(2)])
    pool.close()
    pool.join()

# %%
import yaml
from munch import munchify
#%%
with open("bottleneck_playground/config.yml", "r") as f:
    config = yaml.safe_load(f)
# %%
mymunch = munchify(config)
# %%
