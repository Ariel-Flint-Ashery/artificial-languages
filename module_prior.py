
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(dir_path)
#%%
from utils import normalize_logprobs, normalize_probs, characterise_language
import numpy as np
from scipy.stats import beta
from math import log, log2
from scipy.special import logsumexp
from functools import lru_cache
from string import ascii_uppercase as ALPHABET
import matplotlib.pyplot as plt
import yaml
from munch import munchify
#%% INITALISE CONFIG
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%
possible_languages = config.language.possible_languages
#forms = config.language.forms
language_type = config.language.language_type
meaning_chr = list(set(''.join(config.language.meanings)))
#language_count = len(forms[0])**len(forms)
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
    return list(map(log, prior))

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
def code_length(encoding):
    code_length = 0
    for e in encoding:
        code_length -= log2(encoding.count(e)/len(encoding))
    return code_length

def non_normed_prior(index, language):
    #signals = ''.join([s for _, s in language])
    #compositional languages
    if language_type[index] == 3:
        groups = ALPHABET[:config.language.word_size]
        language_dict = {g: [] for g in groups}
        for m in meaning_chr:
            words_same_meaning = [word for word in language if m in word[0]]
            signals = [word[1] for word in words_same_meaning]
            for i in range(config.language.word_size):
                signal_chr = [sig[i] for sig in signals]
                if all(s==signal_chr[0] for s in signal_chr):
                    language_dict[groups[i]].append(m+signal_chr[0])
        
        encoding = ['S'+''.join(language_dict.keys())]
        for key in language_dict.keys():
            for pair in language_dict[key]:
                encoding.append(key+pair)
        encoding = '.'.join(encoding)

    else:
        language_dict = {signal: [m for m,s in language if s==signal] for signal in set([s for _, s in language])}
        encoding = []
        for signal in language_dict.keys():
            if len(language_dict[signal]) > 1:
                encoding.append('S' + ','.join(language_dict[signal]) + signal)
            else:
                encoding.append('S' + language_dict[signal][-1] + signal)
        encoding = '.'.join(encoding)

    return 2 ** -code_length(encoding)

def get_compressible_prior():
    priors = [non_normed_prior(index, language) for index, language in enumerate(possible_languages)] #list(map(non_normed_prior, possible_languages))
    priors = normalize_probs(priors)
    priors = list(map(log, priors)) #[log(prior) for prior in priors]
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
    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (8,8))
    ax1.plot(np.exp(prior))
    ax1.set_xlabel('Language by index')
    ax1.set_ylabel('Probability')

    #language_type = [characterise_language(data) for data in possible_languages]
    prior_type = get_logprior_type_dist(prior)
    ax2.bar(range(num_types), np.exp(prior_type), color = colors, tick_label = labels)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Language Type')
    #plt.tight_layout()
    #plt.savefig(rf'figures/', dpi = 200)#, bbox_inches = 'tight', pad_inches = 0)
    #print(np.exp(prior_type))
    plt.show()
# %%
