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
initial_training_rounds = config.constants.initial_training_rounds

if signaller_type == 'oldest':
    signaller_age = pop_size -1

if signaller_type == 'learner':
    signaller_age = ceil(pop_size/2) - 1

#%%
def update_posterior(posterior, meaning, signal):
    in_language = log(1 - epsilon)
    out_of_language = log(epsilon / (len(signals) - 1))
    new_posterior = []
    for i in range(len(posterior)):
        if [meaning, signal] in possible_languages[i]:
            new_posterior.append(posterior[i] + in_language)
        else:
            new_posterior.append(posterior[i] + out_of_language)
    return normalize_logprobs(new_posterior)


def sample(posterior): #sample language from posterior
    return possible_languages[log_roulette_wheel(posterior)]

def signalling(posterior, meaning):
    signal_probs = []
    for signal in signals:
        probs = []
        for i in range(len(posterior)):
            language = possible_languages[i]
            if [meaning, signal] in language:
                a = list(sum(possible_languages[i], [])).count(signal) #use square brackets here?
                probs.append(log((1 / a) ** expressivity) + posterior[i])
        signal_probs.append(logsumexp(probs))

    if MAP == True:
        #max_signal_prob = max(signal_probs)
        #signal_index = random.choice([i for i, v in enumerate(signal_probs) if v == max_signal_prob])
        signal_index = random.choice(np.where(np.array(signal_probs)==max(signal_probs))[0])
    else:
        #signal_index = roulette_wheel(normalize_probs(signal_probs))
        signal_index = np.random.choice(range(len(signal_probs)), p = normalize_probs(signal_probs))
    return signals[signal_index]

def new_population(prior):
    population = []
    for i in range(pop_size):
        population.append(prior)
    return population    

def population_communication(population):
    learner_index = random.randrange(len(population))
    if signaller_type == 'random':
        signaller_index = random.choice(range(pop_size))
    else:
        signaller_index = signaller_age - 1
    
    meaning = random.choice(meanings)
    signal = signalling(population[signaller_index], meaning, expressivity=expressivity, MAP=MAP)
    if random.random() < epsilon:
        other_signals = [s for s in signals if s != signal]
        signal = random.choice(other_signals)
    population[learner_index] = update_posterior(population[learner_index], meaning, signal)

def language_stats(posteriors):
    stats = [0., 0., 0., 0.] # degenerate, holistic, other, combinatorial
    for p in posteriors:
        for i in range(len(p)):
            stats[language_type[i]] += exp(p[i]) / len(posteriors)
    return stats

def iterate(prior, bottleneck):
    indices = np.where(np.array(language_type)==1)[0]
    seed_languages = [possible_languages[index] for index in indices]
    results = []
    population = new_population(prior)
    for i in range(pop_size):
        seed_language = random.choice(seed_languages)
        for j in range(initial_training_rounds): # This just trains the initial population on the seed language
            meaning, signal = random.choice(seed_language)
            population[i] = update_posterior(population[i], meaning, signal)
    results.append(language_stats(population))

    for i in range(generations):
        for j in range(bottleneck):
            population_communication(population, expressivity=expressivity, MAP=MAP)
        results.append(language_stats([population[-1]])) # We measure the stats just on the oldest learner
        population = new_population(1) + population[:-1] # remove the oldest and add a newborn learner
                       
    return results