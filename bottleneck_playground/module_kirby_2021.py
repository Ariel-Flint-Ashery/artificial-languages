"""
This script contains a replication of Kirby & Tamariz (2021).
1. Speaker produces data from multiple languages.
2. Speaker data aims to minimize ambiguity
3. Posterior is a distribution over all languages
4. Speaker may consider expressivity, but listener does not take this into account
5. Speaker may make mistakes, and listener takes this into account
6. Speaker chooses best signal across all languages for each meaning i.e. mixed population
7. Listener assumes data is produced from a single language - irrational listener
8. Listener updates posterior after every utterance
"""
#%%
import numpy as np
import random
from utils import normalize_logprobs, normalize_probs, log_roulette_wheel, roulette_wheel
from math import log, ceil, exp
from scipy.special import logsumexp
import random
from itertools import product, permutations, combinations
import yaml
from munch import munchify
#%% INITALISE CONFIG
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%% READ CONSTANTS FROM CONFIG
possible_languages = config.language.possible_languages
language_type = config.language.language_type
signals = config.language.signals
meanings = config.language.meanings
epsilon = config.constants.epsilon
generations = config.constants.generations
expressivity = config.constants.expressivity
MAP = config.constants.MAP
signaller_type = config.constants.signaller_type
pop_size = config.constants.pop_size
training_rounds = config.constants.training_rounds

if signaller_type == 'oldest':
    signaller_age = pop_size -1

if signaller_type == 'learner':
    signaller_age = ceil(pop_size/2) - 1

#%%
def update_posterior(data, posterior): 
    in_language = log(1 - epsilon)
    out_of_language = log(epsilon / (len(signals) - 1))
    new_posterior = posterior.copy()
    for word in data:
        for i in range(len(posterior)):
            if word in possible_languages[i]:
                new_posterior[i] += in_language
            else:
                new_posterior[i] += out_of_language
    return normalize_logprobs(new_posterior)

def sample(probs, MAP = True):
    if MAP == True:
        index = random.choice(np.where(np.array(probs)==max(probs))[0])
    else:
        index = log_roulette_wheel(probs)
    return index

def get_signal_dict(posterior, expressivity=0):
    signal_dict = {m: {} for m in meanings}
    for m in signal_dict.keys():
        signal_probs = []
        for signal in signals:
            probs = []
            for i in range(len(posterior)):
                language = possible_languages[i]
                if [m, signal] in language:
                    if expressivity == 0:
                        probs.append(posterior[i])
                    else:
                        a = list(sum(possible_languages[i], [])).count(signal)
                        probs.append(log((1 / a) ** expressivity) + posterior[i])
            signal_probs.append(logsumexp(probs))
        signal_dict[m] = signal_probs
    return signal_dict

def new_population(prior):
    population = []
    for i in range(pop_size):
        population.append(prior)
    return population    

def population_communication(population, bottleneck, expressivity=0, MAP = True):

    #choose signaller
    if signaller_type == 'random':
        signaller_index = random.choice(list(range(pop_size)))
    else:
        signaller_index = signaller_age

    #remove signaller from potential learner pool. We want the learner to be a different group member.
    learners = list(range(pop_size))
    learners.remove(signaller_index)

    #obtain signal probabilities for each meaning for signaller
    signal_dict = get_signal_dict(population[signaller_index], expressivity)
    
    meanings_to_produce = random.choices(list(signal_dict.keys()), k=bottleneck)
    
    #communicate
    for meaning in meanings_to_produce:
        learner_index = random.choice(learners)
        signal = signals[sample(signal_dict[meaning], MAP = MAP)]
        if random.random() < epsilon:
            other_signals = [s for s in signals if s != signal]
            signal = random.choice(other_signals)
        population[learner_index] = update_posterior([[meaning, signal]], population[learner_index])

def language_stats(posteriors):
    stats = [0., 0., 0., 0.] # degenerate, holistic, other, combinatorial
    for p in posteriors:
        probs = np.exp(p)  / len(posteriors)
        for i in range(len(p)):
            stats[language_type[i]] += probs[i]
    return stats

def iterate(prior, bottleneck):
    indices = np.where(np.array(language_type)==1)[0]
    seed_languages = [possible_languages[index] for index in indices]
    seed_language = seed_languages[0] #random.choice(seed_languages)
    results = []
    population = new_population(prior)
    for i in range(pop_size):
        #seed_language = random.choice(seed_languages)
        for j in range(training_rounds): # This just trains the initial population on the seed language
            word = random.choice(seed_language)
            population[i] = update_posterior([word], population[i])
    results.append(language_stats([population[-1]]))

    for i in range(generations):
        population_communication(population, bottleneck, expressivity=expressivity, MAP=MAP)
        results.append(language_stats([population[-1]])) # We measure the stats just on the oldest learner
        population = [prior] + population[:-1] # remove the oldest and add a newborn learner
                       
    return results
# %%
