
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(dir_path)
from utils import normalize_logprobs, normalize_probs, characterise_language
import numpy as np
from scipy.stats import beta
from math import log, log2
from scipy.special import logsumexp
from functools import lru_cache
import matplotlib.pyplot as plt
import yaml
from munch import munchify
#%% INITALISE CONFIG
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%
possible_languages = config.language.possible_languages
language_type = config.language.language_type
#%%
def get_beta_logprior():
    alpha = config.prior_constants.alpha
    hspace, step = np.linspace(0,1, len(possible_languages), endpoint=False, retstep = True)
    hspace+=step*0.5
    logprior = []
    for h in hspace:
        logprior.append(beta.logpdf(h, alpha, alpha)) 
    return normalize_logprobs(logprior) 

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
        
def get_biased_prior(prior):
    bias = config.prior_constants.bias
    if len(bias)<len(set(language_type)):
        return TypeError('config file missing bias initialisation')
    
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

def get_prior(prior_type=config.prior_constants.prior_type):
    #language_type = [characterise_language(data) for data in possible_languages]

    if prior_type == 'compressible':
        return get_compressible_prior()
    
    if prior_type == 'uniform':
        return get_uniform_logprior()
    
    if prior_type == 'beta':
        # if alpha == None:
        #     return TypeError('Please specify alpha parameter')
        return get_beta_logprior()
    
    if prior_type == 'biased':
        # if bias == None:
        #     return TypeError('Please specify alpha parameter')
        return get_biased_prior(get_uniform_logprior())


def plot_prior(prior, num_types=len(set(language_type)), colors = config.plotting_params.colors, labels = config.plotting_params.labels):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(np.exp(prior))
    ax1.set_xlabel('Language by index')
    ax1.set_ylabel('Probability')

    #language_type = [characterise_language(data) for data in possible_languages]
    prior_type = get_logprior_type_dist(prior)
    ax2.bar(range(num_types), np.exp(prior_type), color = colors, tick_label = labels)

    plt.tight_layout()
    #plt.savefig(rf'figures/', dpi = 200)#, bbox_inches = 'tight', pad_inches = 0)
    plt.show()