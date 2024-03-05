#%%
from math import log, exp
from scipy.special import logsumexp
from utils import log_roulette_wheel, normalize_logprobs
from collections import defaultdict 
import yaml
from munch import munchify
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%

"""
A crp partition is represented by three lists: a list of counts for clusters, a list of labels for those clusters, and a list of assignments for
objects indicating which table they belong to.
Base gives the probability of each label (i.e. the nth element of base gives the probability of label n).
"""


#data = [] # observed data
possible_languages = config.language.possible_languages # set of all possible languages
expressivity = config.constants.expressivity
signals = config.language.signals
#base = [] # in log form
epsilon = config.constants.epsilon
alpha = config.prior_constants.alpha # concentration parameter
training_rounds = config.constants.training_rounds
def probability_word_in_language(word, language):
    in_language = log(1 - epsilon)
    out_of_language = log(epsilon / (len(signals) - 1))
    if word in language:
        a=list(sum(language, [])).count(word[1]) # add ambiguity term
        express_term = log((1/a)**expressivity)
        return in_language + express_term 
    else:
        return out_of_language

def language_assignment(base, cluster, data):
    logprobs = base.copy()
    for word in data:
        if word in cluster:
            for i, language in enumerate(possible_languages):
                logprobs[i] += probability_word_in_language(word, language)

    # sample a language
    language_index = log_roulette_wheel(logprobs)
    return language_index

def CRP_process(base, alpha, data, training_rounds):
    n = len(data)
    # first round
    #clusters = {0: [data[0]]}
    clusters = defaultdict(list)
    clusters[0] = [data[0]]
    languages = {0: possible_languages[language_assignment(base, clusters[0], data)]}
    #print(list(clusters.keys()))
    # for i in list(clusters.keys()):
    #     print(languages[i])
    #     print(len(clusters[i]))
    for d in range(1,n):
        word = data[d] # choose new word
        # assign to cluster
        prob_new_cluster = log(alpha) + logsumexp([base[i]+probability_word_in_language(word, language) for i, language in enumerate(possible_languages)]) # probability of creating a new cluster
        prob_clusters = [log(len(clusters[i]))+probability_word_in_language(word, languages[i]) for i in list(clusters.keys())] # probability of joining an existing cluster is proportional to its size
        normedlogs = normalize_logprobs(prob_clusters+[prob_new_cluster]) # normalise cluster probability distribution
        cluster_idx = log_roulette_wheel(normedlogs) # choose an existing cluster, or create a new one
        clusters[cluster_idx].append(word) # finally, we can add word to cluster
        languages[cluster_idx] = possible_languages[language_assignment(base, clusters[cluster_idx], data)] # assign language to cluster based on its new and existing members

    # repeat to 'burn in' Gibbs sampling process
    for it in range(training_rounds-1):
            for d in range(n):
                word = data[d]
                prob_new_cluster = log(alpha) + logsumexp([base[i]+probability_word_in_language(word, language) for i, language in enumerate(possible_languages)])
                prob_clusters = [log(len(clusters[i]))+probability_word_in_language(word, languages[i]) for i in clusters.keys()]
                normedlogs = normalize_logprobs(prob_clusters+[prob_new_cluster])
                cluster_idx = log_roulette_wheel(normedlogs)
                clusters[cluster_idx].append(word)
                languages[cluster_idx] = possible_languages[language_assignment(base, clusters[cluster_idx], data)]
    
    # do inference
    distribution = []
    for idx, language in enumerate(possible_languages):
        language_count = sum([len(clusters[c]) for c in list(clusters.keys()) if languages[c] == language])
        probability = (language_count + alpha*exp(base[idx])) / (len(data) + alpha)
        distribution.append(log(probability))
    
    return normalize_logprobs(distribution)

def incremental_CRP(base, alpha, data, training_rounds):
    posterior = base.copy()
    for i in range(len(data)):
        subset = data[:i+1]
        posterior = CRP_process(posterior, alpha, subset, training_rounds)
# %%
