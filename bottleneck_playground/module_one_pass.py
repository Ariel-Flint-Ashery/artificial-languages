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
from utils import normalize_logprobs, normalize_probs, get_init_language, log_roulette_wheel
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
random.seed(config.constants.seed)
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
#%%
def loglikelihoods(word, posterior):
    #likelihood of generating a sequence of (m,s) pairs for each language.
    in_language = log(1 - epsilon)
    out_of_language = log(epsilon / (len(signals) - 1))
    # loglikelihoods = []
    # for d in data:
    #     logprob = []
    #     for language in possible_languages:
    #         if d in language:
    #             logprob.append(in_language)
    #         else: 
    #             logprob.append(out_of_language)
    #     loglikelihoods.append(logprob)
    # sequence_likelihood = [sum(i) for i in zip(*loglikelihoods)] #do I need to normalize here?
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
    
    #gumbel should (in theory) work with unnormalised posterior - gives crossover.
    # try with normalised posterior.
    return normalize_logprobs(posterior)
 
def sample(posterior, MAP = False):
    if MAP==False:
        #unnormalised w/ gumbel: crossover
        #normalised posterior during production:
        #normalised posterior during sampling:
        selected_index = log_roulette_wheel(posterior)#np.random.choice(range(len(possible_languages)), p = np.exp(posterior)) #log_roulette_wheel(posterior)
    else:
        #probs = np.exp(posterior)
        #max_signal_prob = max(probs)
        selected_index = random.choice(np.where(np.array(posterior)==max(posterior))[0])#random.choice([i for i, v in enumerate(probs.tolist()) if v == max_signal_prob])
    return possible_languages[selected_index]

def produce(posterior, bottleneck, language = None, signal_dict = dict()):#initial_language=False, initial_population=False):
    # randomly choose meaning to express
    intended_meanings = random.choices(meanings, k=bottleneck)

    if language == None:

        # select speaker language

        if expressivity==0:
            language = sample(posterior, MAP=MAP)
            signal_dict = get_signal_dict(posterior)
            
        else: #if expressivity != 0:
            # weight each language by how easy it is to express a given meaning
            new_posterior = posterior.copy()

            meaning_dict = {m: intended_meanings.count(m) for m in set(intended_meanings)}

            for meaning in meaning_dict.keys():
                for signal in signals:
                    for i in range(len(possible_languages)):
                        if [meaning, signal] in possible_languages[i]:
                            a=list(sum(possible_languages[i], [])).count(signal) # add ambiguity term
                            express_term = log((1/a)**expressivity)
                            new_posterior[i]+=express_term * meaning_dict[meaning]
            #prob = normalize_logprobs(new_posterior)
            language = sample(normalize_logprobs(new_posterior), MAP = MAP) #normalise?
            signal_dict = get_signal_dict(normalize_logprobs(new_posterior))

    # generate data

    data=[]
    for meaning in intended_meanings:
        for m, s in language:
            if m == meaning:
                signal = s # find the signal that is mapped to the meaning 
    
        if random.random() < epsilon: # add the occasional mistake
            other_signals = []
            for other_signal in signals:
                if other_signal != signal:
                    other_signals.append(other_signal) # make a list of all the "wrong" signals
            signal = random.choice(other_signals) # pick one of them        
        data.append([meaning, signal])
    
    return data, signal_dict #, language

def language_stats(posteriors):
    stats = [0., 0., 0., 0.] # degenerate, holistic, other, combinatorial
    for p in posteriors:
        p = np.exp(p) / len(posteriors) #np.exp(normalize_logprobs(p))
        for i in range(len(p)):
            stats[language_type[i]] += p[i] 
    return stats

def get_signal_dict(population):
    signal_dict = {m: {} for m in meanings}
    for m in signal_dict.keys():
        signal_probs = []
        for signal in signals:
            probs = []
            for i in range(len(population)):
                language = possible_languages[i]
                if [m, signal] in language:
                    probs.append(population[i] + log(1-epsilon))
                else:
                    probs.append(population[i] + log(epsilon))
            #probs = [log((1 / list(sum(possible_languages[i], [])).count(signal)) ** expressivity) + posterior[i] for i in range(len(posterior)) if [m, signal] in possible_languages[i]]
            signal_probs.append(logsumexp(probs))
        signal_dict[m] = normalize_logprobs(signal_probs)
    return signal_dict

def signal_stats(signal_dict):
    values = np.exp(list(signal_dict.values()))
    return np.mean(values, axis = 0)#, np.std(values, axis = 0)/np.sqrt(len(values))

def iterate(prior, bottleneck):
    # initialise posterior
    posterior = prior.copy()
    language_results = []
    signal_results = []
    # initalise data collection
    data, signal_dict = produce(posterior, bottleneck=bottleneck, language= possible_languages[log_roulette_wheel(get_init_language(initial_language, language_type))])#np.random.choice(possible_languages, p=get_init_language(initial_language, language_type)))
    #language_accumulator = [language]
    #posterior_accumulator = [posterior]
    #data_accumulator = [data]
    print(data)
    print(signal_dict)
    # iterate across generations
    for generation in range(generations):
        posterior = update_posterior(data, prior)
        data, signal_dict = produce(posterior, bottleneck=bottleneck) #[produce(language) for i in range(bottleneck)]

        #language_accumulator.append(language)
        #posterior_accumulator.append(posterior)
        #data_accumulator.append(data)
        language_results.append(language_stats([posterior]))
        signal_results.append(signal_stats(signal_dict))

    return language_results, signal_results #language_accumulator, posterior_accumulator, data_accumulator
# %%
