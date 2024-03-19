#%%
import numpy as np
from utils import normalize_logprobs, normalize_probs, log_roulette_wheel, get_init_language
from CRP import incremental_CRP, CRP_process
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
signals = config.language.signals
meanings = config.language.meanings
epsilon = config.constants.epsilon
generations = config.constants.generations
expressivity = config.constants.expressivity
MAP = config.constants.MAP
pop_size = config.constants.pop_size
training_rounds = config.constants.training_rounds
incremental_learning = config.constants.incremental_learning
alpha = config.prior_constants.alpha # concentration parameter
#%%
def sample(probs, MAP = False):
    if MAP == True:
        index = random.choice(np.where(np.array(probs)==max(probs))[0])
    else:
        index = log_roulette_wheel(probs)
    return index

def new_population(prior):
    population = []
    for i in range(pop_size):
        population.append(prior)
    return population    

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

def produce(population, bottleneck, expressivity=0, MAP=False):
    # select a random set of meanings
    meanings_to_produce = random.choices(meanings, k=bottleneck)
    # select a random agent for each meaning
    signaller_ids = random.choices(list(range(pop_size)), k = bottleneck)
    signaller_dict = {id: get_signal_dict(population[id], expressivity=expressivity) for id in set(signaller_ids)}
    # select language and word for each agent-meaning pair
    data = []
    for meaning, id in zip(meanings_to_produce, signaller_ids):
        signal = signals[sample(signaller_dict[id][meaning], MAP = MAP)]
        if random.random() < epsilon:
            other_signals = [s for s in signals if s != signal]
            signal = random.choice(other_signals)
        data.append([meaning, signal])
    return data

def run_inference(prior, data):
    new_pop = []
    if incremental_learning == True:
        for i in range(pop_size):
            new_pop.append(incremental_CRP(base=prior, alpha=alpha, data=data, training_rounds=training_rounds))
    else:
        for i in range(pop_size):
            new_pop.append(CRP_process(base=prior, alpha=alpha, data=data, training_rounds=training_rounds))
    
    return np.array(new_pop)

def language_stats(posteriors):
    stats = np.array([0., 0., 0., 0.]) # degenerate, holistic, other, combinatorial
    for p in posteriors:
        p = np.exp(p)  / len(posteriors) 
        for i in range(len(p)):
            stats[language_type[i]] += p[i]
    return stats #np.array(stats)/sum(stats)

def iterate(prior, bottleneck):
    indices = np.where(np.array(language_type)==1)[0]
    seed_languages = [possible_languages[index] for index in indices]
    seed_language = seed_languages[0] #random.choice(seed_languages)
    results = []
    #population = [] #new_population(prior)
    # train the initial population on the seed language
    data_to_learn = list(random.choices(seed_language, k = 2*bottleneck))
    population = run_inference(prior, data_to_learn)
    results = [language_stats(population)]

    for i in range(generations):
        data_to_learn = produce(population, bottleneck, expressivity=expressivity, MAP=MAP)
        population = run_inference(prior, data_to_learn)
        results.append(language_stats(population))
        #population = [prior] + population[:-1] # remove the oldest and add a newborn learner
                       
    return np.array(results)
# %%
