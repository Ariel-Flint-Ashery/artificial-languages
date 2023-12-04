"""
In this model, we assume that population are long, thin chains organised into discrete generations.
In each generation, the new learners learn from data produced by the previous generation.
Languages are chosen on word-by-word basis i.e. learners learn from multiple teachers
"""
#%%
import numpy as np
from utils import normalize_logprobs, normalize_probs, get_init_language, log_roulette_wheel
from math import log, exp
from scipy.special import logsumexp
from itertools import cycle
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
continual_learning = config.constants.continual_learning
initial_language = config.constants.initial_language
pop_size = config.constants.pop_size
degree = config.constants.degree
training_rounds = config.constants.training_rounds
#%%
# def loglikelihoods(word, posterior):
#     #likelihood of generating a sequence of (m,s) pairs for each language.
    # in_language = log(1 - epsilon)
    # out_of_language = log(epsilon / (len(signals) - 1))
    # new_posterior = []
    # for i in range(len(posterior)):
    #     if word in possible_languages[i]:
    #         new_posterior.append(posterior[i] + in_language)
    #     else:
    #         new_posterior.append(posterior[i] + out_of_language)
#     return new_posterior

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
        #posterior = loglikelihoods(word, posterior)
    return normalize_logprobs(new_posterior)

# def get_signal_dict(posterior, expressivity=0):
#     signal_dict = {m: {} for m in meanings}
#     express_term = []
#     for signal in signals:
#         e=[]
#         for i in range(len(posterior)):
#             a = []
#             list(map(a.extend, possible_languages[i]))
#             e.append(log((1/a.count(signal))**expressivity))
#         express_term.append(e)

#     for m in signal_dict.keys():
#         signal_probs = []
#         for j in range(len(signals)):
#             #probs = [log((1 / list(sum(possible_languages[i], [])).count(signal)) ** expressivity) + posterior[i] for i in range(len(posterior)) if [m, signal] in possible_languages[i]]
#             probs = [express_term[j][i] + posterior[i] for i in range(len(posterior)) if [m, signals[j]] in possible_languages[i]]
#             signal_probs.append(logsumexp(probs))
#         signal_dict[m] = normalize_logprobs(signal_probs)
#     return signal_dict
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
            #probs = [log((1 / list(sum(possible_languages[i], [])).count(signal)) ** expressivity) + posterior[i] for i in range(len(posterior)) if [m, signal] in possible_languages[i]]
            signal_probs.append(logsumexp(probs))
        signal_dict[m] = signal_probs
    return signal_dict

def sample(probs, MAP = False):
    if MAP == True:
        index = random.choice(np.where(np.array(probs)==max(probs))[0])
    else:
        ##signal_index = roulette_wheel(normalize_probs(signal_probs))
        #signal_index = np.random.choice(range(len(signal_probs)), p = normalize_probs(signal_probs))
        index = log_roulette_wheel(probs)
    return index


# def produce(posterior):
#     # generate data
#     meaning = random.choice(meanings)
#     signal = signalling(posterior, meaning, expressivity=expressivity, MAP=MAP)
#     if random.random() < epsilon:
#         other_signals = [s for s in signals if s != signal]
#         signal = random.choice(other_signals)
    
#     return [meaning, signal]
def produce(signal_dict, num_words):
    meanings_to_produce = random.choices(list(signal_dict.keys()), k=num_words)
    data = []
    for meaning in meanings_to_produce:
        signal = signals[sample(signal_dict[meaning], MAP = MAP)]
        if random.random() < epsilon:
            other_signals = [s for s in signals if s != signal]
            signal = random.choice(other_signals)
        data.append([meaning, signal])
    return data
# def do_continual_learning(learner_index, learner_population, speaker_population, bottleneck):
#     speakers = random.sample([i for i in range(len(speaker_population)) if i != learner_index], k=degree)
#     random.shuffle(speakers)
#     speakers = cycle(speakers)
#     #data = [produce(speaker_population[next(speakers)]) for b in range(bottleneck)]
#     posteriors = [speaker_population[next(speakers)] for b in range(bottleneck)]
#     data = list(map(produce, posteriors))
#     learner_population[learner_index] = update_posterior(data, learner_population[learner_index])

def do_population_learning(learner_population, speaker_population, bottleneck):
    speaker_dict = {s: get_signal_dict(speaker_population[s]) for s in range(len(speaker_population))}

    for learner_index in range(len(learner_population)):
        potential_speakers = random.sample([s for s in range(len(speaker_population)) if s != learner_index], k=degree)
        speakers_in_turn = random.choices(potential_speakers, k = bottleneck)
        # random.shuffle(speakers)
        # speakers = cycle(speakers)
        # posteriors = [speaker_population[next(speakers)] for b in range(bottleneck)]
        # data = list(map(produce, posteriors))
        dicts = [speaker_dict[s] for s in set(speakers_in_turn)]
        num_words = [speakers_in_turn.count(s) for s in set(speakers_in_turn)]
        data = []
        list(map(data.extend, list(map(produce, dicts, num_words))))
        learner_population[learner_index] = update_posterior(data, learner_population[learner_index])
        #do_continual_learning(i, learner_population, speaker_population, bottleneck)

def new_population(prior, pop_size):
    return [prior]*pop_size

def language_stats(posteriors):
    stats = [0., 0., 0., 0.] # degenerate, holistic, other, combinatorial
    for p in posteriors:
        p = normalize_probs(list(map(exp, p)))
        for i in range(len(p)):
            stats[language_type[i]] += p[i] / len(posteriors)
    return stats

def iterate(prior, bottleneck):
    learner_population = new_population(prior, pop_size)
    do_population_learning(learner_population, speaker_population = [get_init_language(language=initial_language, language_type=language_type)]*pop_size, bottleneck = training_rounds)
    results = [language_stats(learner_population)]
    speaker_population = learner_population
    learner_population = new_population(prior, pop_size)
    # iterate across generations

    for generation in range(generations-1):
        do_population_learning(learner_population, speaker_population, bottleneck)
        results.append(language_stats(learner_population))
        speaker_population = learner_population
        learner_population = new_population(prior, pop_size)

    return results #language_accumulator, posterior_accumulator, data_accumulator
# %%
