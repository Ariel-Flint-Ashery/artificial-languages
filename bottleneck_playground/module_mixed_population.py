"""
In this model, we assume that population are infinitely large and organised into discrete generations.
In each generation, the new learners learn from data produced by the previous generation.
Languages are chosen on word-by-word basis i.e. learners learn from multilingual teachers
"""
#%%
import numpy as np
from utils import normalize_logprobs, normalize_probs, log_roulette_wheel, get_init_language
from math import log, exp
from scipy.special import logsumexp
from multiprocessing import Process, Queue #, Pipe
import time
import queue # imported for using queue.Empty exception
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
initial_language = config.constants.initial_language
training_rounds = config.constants.training_rounds
#%%
# def loglikelihoods(word, posterior):
#     #likelihood of generating a sequence of (m,s) pairs for each language.
#     in_language = log(1 - epsilon)
#     out_of_language = log(epsilon / (len(signals) - 1))
#     new_posterior = []
#     for i in range(len(posterior)):
#         if word in possible_languages[i]:
#             new_posterior.append(posterior[i] + in_language)
#         else:
#             new_posterior.append(posterior[i] + out_of_language)
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

def get_signal_dict(population, expressivity=0):
    signal_dict = {m: {} for m in meanings}
    for m in signal_dict.keys():
        signal_probs = []
        for signal in signals:
            probs = []
            for i in range(len(population)):
                language = possible_languages[i]
                if [m, signal] in language:
                    if expressivity == 0:
                        probs.append(population[i])
                    else:
                        a = list(sum(possible_languages[i], [])).count(signal)
                        probs.append(log((1 / a) ** expressivity) + population[i])
            #probs = [log((1 / list(sum(possible_languages[i], [])).count(signal)) ** expressivity) + posterior[i] for i in range(len(posterior)) if [m, signal] in possible_languages[i]]
            signal_probs.append(logsumexp(probs))
        signal_dict[m] = signal_probs#normalize_logprobs(signal_probs)
    return signal_dict

def sample(probs, MAP = False):
    if MAP == True:
        index = random.choice(np.where(np.array(probs)==max(probs))[0])
    else:
        ##signal_index = roulette_wheel(normalize_probs(signal_probs))
        #signal_index = np.random.choice(range(len(signal_probs)), p = normalize_probs(signal_probs))
        index = log_roulette_wheel(probs)
    return index

def produce(population, bottleneck, expressivity=0, MAP = False):
    signal_dict = get_signal_dict(population, expressivity)
    meanings_to_produce = random.choices(list(signal_dict.keys()), k=bottleneck)
    data = []
    for meaning in meanings_to_produce:
        signal = signals[sample(signal_dict[meaning], MAP = MAP)]
        if random.random() < epsilon:
            other_signals = [s for s in signals if s != signal]
            signal = random.choice(other_signals)
        data.append([meaning, signal])
    return data, signal_dict

def do_continual_learning(prior, population, bottleneck):
    #data = list(map(produce, [population]*bottleneck))
    #data = get_data(population, bottleneck)
    data, signal_dict_result = produce(population, bottleneck, expressivity = expressivity, MAP = MAP)
    return update_posterior(data, prior), signal_dict_result

def language_stats(posteriors):
    stats = [0., 0., 0., 0.] # degenerate, holistic, other, combinatorial
    for p in posteriors:
        p = np.exp(p)  / len(posteriors) 
        for i in range(len(p)):
            stats[language_type[i]] += p[i]
    return stats

def signal_stats(signal_dict):
    stats = [[0]*len(signals)]*len(meanings)
    for idx, m in enumerate(signal_dict.keys()):
        stats[idx] = np.exp(signal_dict[m])
    return np.array(stats)/np.sum(stats)

def iterate(prior, bottleneck):
    posterior, signal_dict_result = do_continual_learning(prior, population= get_init_language(language=initial_language, language_type=language_type), bottleneck=bottleneck)
    #print(signal_dict_result)
    # iterate across generations
    language_results = [language_stats([posterior])]
    signal_results = [signal_stats(signal_dict_result)]
    posterior_accumulator = [posterior]
    for generation in range(generations):
        posterior, signal_dict_result = do_continual_learning(prior=prior, population=posterior, bottleneck=bottleneck)
        language_results.append(language_stats([posterior]))
        posterior_accumulator.append(posterior)
        signal_results.append(signal_stats(signal_dict_result))

    return [np.array(language_results), np.array(signal_results), np.exp(posterior_accumulator)]
#%%
# def signalling(posterior, meaning, expressivity=0, MAP=False):
#     signal_probs = []
#     for signal in signals:
#         probs = []
#         for i in range(len(posterior)):
#             language = possible_languages[i]
#             if [meaning, signal] in language:
#                 a = list(sum(possible_languages[i], [])).count(signal) #use square brackets here?
#                 probs.append(log((1 / a) ** expressivity) + posterior[i])
#         signal_probs.append(logsumexp(probs))

#     if MAP == True:
#         signal_index = random.choice(np.where(np.array(signal_probs)==max(signal_probs))[0])
#     else:
#         ##signal_index = roulette_wheel(normalize_probs(signal_probs))
#         #signal_index = np.random.choice(range(len(signal_probs)), p = normalize_probs(signal_probs))
#         signal_index = log_roulette_wheel(signal_probs)
#     return signals[signal_index]

# def produce(posterior):
#     # generate data
#     meaning = random.choice(meanings)
#     signal = signalling(posterior, meaning, expressivity=expressivity, MAP=MAP)
#     if random.random() < epsilon:
#         other_signals = [s for s in signals if s != signal]
#         signal = random.choice(other_signals)
    
#     return [meaning, signal]

# def speak(speaker, output_queue):
#     # while True:
#     #     try:
#     #         '''
#     #             try to get task from the queue. get_nowait() function will 
#     #             raise queue.Empty exception if the queue is empty. 
#     #             queue(False) function would do the same task also.
#     #         '''
#     #         speaker = speaker_queue.get_nowait()
#     #     except queue.Empty:

#     #         break
#     #     else:
#     #         '''
#     #             if no exception has been raised, add the task completion 
#     #             message to task_that_are_done queue
#     #         '''
#     #         output_queue.put(produce(speaker))
#     # return True
#     output_queue.put(produce(speaker))
    
# def get_data(population, bottleneck):
#     number_of_processes = 8
#     speaker_queue = cycle([population]*bottleneck)
#     output_queue = Queue()
#     processes = []
#     data = []
#     # for b in range(bottleneck):
#     #     speaker_queue.put(population)

#     for w in range(number_of_processes):
#         #recv_end, send_end = Pipe(False)
#         p = Process(target=speak, args=(next(speaker_queue), output_queue))
#         processes.append(p)
#         p.start()
#     for p in processes:
#         data.append(output_queue.get())
#     # completing process
#     for p in processes:
#         p.join()
#     # for p in processes:
#     #     p.close()
#     return data