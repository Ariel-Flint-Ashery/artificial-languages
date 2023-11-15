"""
This is a module containing functions for the iterated learning model involving an agent with the following assumptions:
1. Speaker produces data using one language
2. Posterior is a distribution over all languages
3. Speaker may consider expressivity, but listener does not take this into account
4. Speaker may make mistakes, and listener takes this into account
5. Speaker chooses language based on entire data sequence ine one pass, rather than sequentially (i.e. choosing different language for each meaning)


"""

#%%
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(dir_path)
import numpy as np
from scipy.stats import beta
from utils import normalize_logprobs, log_roulette_wheel
from math import log, log2
from scipy.special import logsumexp
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product, permutations, combinations
from functools import lru_cache
import multiprocessing
import time
import yaml
from munch import munchify
#%% INITALISE CONFIG
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%% READ CONSTANTS FROM CONFIG
possible_languages = config.language.possible_languages
language_type = config.language.language_type
error_probability = config.constants.error_probability
signals = config.language.signals
meanings = config.language.meanings
# expressivity = config.constants.expressivity
# MAP = config.constants.MAP
# bottlerange = config.constants.bottlerange
#%%
def loglikelihoods(data):
    #likelihood of generating a sequence of (m,s) pairs for each language.
    in_language = log(1 - error_probability)
    out_of_language = log(error_probability / (len(signals) - 1))
    # sequence_likelihood = np.zeros(len(possible_languages))
    # for d in data:
    #     for i in range(len(possible_languages)):
    #         if d in possible_languages[i]:
    #             sequence_likelihood[i]+=in_language
    #         else:
    #             sequence_likelihood[i]+=out_of_language
    loglikelihoods = []
    for d in data:
        logprob = []
        for language in possible_languages:
            if d in language:
                logprob.append(in_language)
            else: 
                logprob.append(out_of_language)
        loglikelihoods.append(logprob)
    sequence_likelihood = [sum(i) for i in zip(*loglikelihoods)] #do I need to normalize here?
    return sequence_likelihood

def update_posterior(data, prior):
    sequence_likelihood = loglikelihoods(data)
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
                    if [meaning, signal] in possible_languages[i]:
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
            data.append([meaning, random.choice(other_signals)]) # pick one of them
        
        data.append([meaning, signal])
    
    return data, language

def iterate(prior, bottleneck, generations, expressivity=0, MAP = False):
    # initialise posterior
    # posterior = np.zeros(len(possible_languages))
    # posterior[indices] = 1/len(indices)
    posterior = prior.copy()
    
    # initalise data collection
    data, language = produce(posterior, bottleneck=bottleneck, expressivity=expressivity, MAP = MAP, initial_language=True)#[produce(initial_language) for i in range(bottleneck)]
    language_accumulator = [language]
    posterior_accumulator = [posterior]
    data_accumulator = [data]

    # iterate across generations
    for generation in range(generations):
        posterior = update_posterior(data, prior)
        data, language = produce(posterior, bottleneck=bottleneck, expressivity=expressivity, MAP = MAP) #[produce(language) for i in range(bottleneck)]
        language_accumulator.append(language)
        posterior_accumulator.append(posterior)
        data_accumulator.append(data)

    return language_accumulator, posterior_accumulator, data_accumulator
# %%
