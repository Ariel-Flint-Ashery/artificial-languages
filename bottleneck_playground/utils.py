#%%
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import numpy as np
from math import log
from scipy.special import logsumexp
import random
from itertools import product
from string import ascii_lowercase as alphabet
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

def roulette_wheel(normedprobs):
    r=random.random() #generate a random number between 0 and 1
    accumulator = normedprobs[0]
    for i in range(len(normedprobs)):
        if r < accumulator:
            return i
        accumulator = accumulator + normedprobs[i + 1]
        
def log_roulette_wheel(normedlogs):
    r=log(random.random()) #generate a random number in [0,1), then convert to log
    accumulator = normedlogs[0]
    for i in range(len(normedlogs)):
        if r < accumulator:
            return i
        accumulator = logsumexp([accumulator, normedlogs[i + 1]])
        
def characterise_language(data):
    # text is a list of (meaning, signal) pairs.
    #check for degenerate language
    if len(set([pair[1] for pair in data]))==1:
        return 0
    
    degenerate_list = list(sum(data, ()))
    if len(degenerate_list) != len(set(degenerate_list)):
        return 2 #partially degenerate
    
    #compositional or holistic?

    for i in range(len(data)-1):
        pair = data[i]
        mean_match = [x for j in range(len(pair[0])) for x in data[i+1:] if x[0][j]==pair[0][j]]
        sig_match = [x for j in range(len(pair[1])) for x in mean_match if x[1][j]==pair[1][j]]
        if len(sig_match)==0:
            return 1 #holistic
                
    return 3 #compositional

def generate_language(vocabulary_size, word_size):
    #alphabet = list(map(chr, range(97, 123))) #default english alphabet
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

    return meanings, signals, possible_languages

def count_ambiguous_languages(possible_languages, signals):
    ambiguous_languages = []
    count = 0
    for language in possible_languages:
        if len(set([s for _,s in language])) < len(signals):
            count+=1
            ambiguous_languages.append(language)

    return count

def get_init_language(language, language_type, possible_languages):
    probs = np.zeros(len(language_type)).tolist()
    if type(language) == int:
        indices = np.where(np.array(language_type)==language)[0]
        for index in indices:
            probs[index]=log(1/len(indices))
        return probs
            #return possible_languages[random.choice(indices)]#random.choice([possible_languages[index] for index in indices])
    
    for t in range(4):
        indices = np.where(np.array(language_type) == t)[0]
        for index in indices:
            probs[index]=log(language[t]/len(indices))
    
    return probs #normalize_logprobs(probs)
# %%
