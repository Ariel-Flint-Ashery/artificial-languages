"""
In this model, we assume that population are infinitely large and organised into discrete generations.
In each generation, the new learners learn from data produced by the previous generation.
Languages are chosen on word-by-word basis i.e. learners learn from multiple teachers
"""
#%%
import numpy as np
from utils import normalize_logprobs, normalize_probs, get_init_language
from math import log, exp
from scipy.special import logsumexp
import random
import yaml
from munch import munchify
#%% INITALISE CONFIG
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%% READ CONSTANTS FROM CONFIG
possible_languages = config.language.possible_languages
language_type = config.language.language_type
epsilon = config.constants.epsilon
signals = config.language.signals
meanings = config.language.meanings
generations = config.constants.generations
expressivity = config.constants.expressivity
MAP = config.constants.MAP
initial_language = config.constants.initial_language
training_rounds = config.constants.training_rounds
#%%
def loglikelihoods(word, posterior):
    #likelihood of generating a sequence of (m,s) pairs for each language.
    in_language = log(1 - epsilon)
    out_of_language = log(epsilon / (len(signals) - 1))
    new_posterior = []
    for i in range(len(posterior)):
        if word in possible_languages[i]:
            new_posterior.append(posterior[i] + in_language)
        else:
            new_posterior.append(posterior[i] + out_of_language)
    return new_posterior

def update_posterior(data, posterior):
    for word in data:
        posterior = loglikelihoods(word, posterior)
    #sequence_likelihood = loglikelihoods(data)
    #new_posterior = normalize_logprobs([sum(i) for i in zip(*[sequence_likelihood, prior])]) #here, we would swap 'prior' with a Dirichlet process prior
    return normalize_logprobs(posterior)

def signalling(posterior, meaning, expressivity=0, MAP=False):
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
        signal_probs = normalize_probs(signal_probs)
        signal_index = random.choice(np.where(np.array(signal_probs)==max(signal_probs))[0])
    else:
        #signal_index = roulette_wheel(normalize_probs(signal_probs))
        signal_index = np.random.choice(range(len(signal_probs)), p = normalize_probs(signal_probs))
    return signals[signal_index]

def produce(posterior):
    # generate data
    meaning = random.choice(meanings)
    signal = signalling(posterior, meaning, expressivity=expressivity, MAP=MAP)
    if random.random() < epsilon:
        other_signals = [s for s in signals if s != signal]
        signal = random.choice(other_signals)
    
    return [meaning, signal]
    
def do_continual_learning(posterior, population, bottleneck):
    data = [produce(population) for b in range(bottleneck)]
    posterior = update_posterior(data, posterior)
    
    return posterior

def language_stats(posteriors):
    stats = [0., 0., 0., 0.] # degenerate, holistic, other, combinatorial
    for p in posteriors:
        for i in range(len(p)):
            stats[language_type[i]] += exp(p[i]) / len(posteriors)
    return stats

def iterate(prior, bottleneck):
    posterior = do_continual_learning(prior, population= get_init_language(language=initial_language, language_type=language_type, possible_languages=possible_languages), bottleneck=bottleneck)
    posterior = do_continual_learning(prior, posterior, bottleneck)
    # iterate across generations
    results = [language_stats([posterior])]
    for generation in range(generations-1):
        # if baby_talk == True:
        #     posterior = do_continual_learning(prior, population= get_init_language(language=initial_language, language_type=language_type, possible_languages=possible_languages), bottleneck=training_rounds)
        posterior = do_continual_learning(prior, posterior, bottleneck)
        
        results.append(language_stats([posterior]))

    return results #language_accumulator, posterior_accumulator, data_accumulator
#%%