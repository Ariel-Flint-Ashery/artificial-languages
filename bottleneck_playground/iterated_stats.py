#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import pickle
import yaml
from munch import munchify
import matplotlib.pyplot as plt
#%% INITALISE CONFIG
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

def file_id(name, pkl = True, directory = None):
    """
    Returns:
        Returns the file name with all the relevant directories
    """
    if directory == None:
        dir_path = os.path.dirname(os.path.realpath(__file__))

        directory = dir_path
    else:
        directory = directory
    if pkl == True:
        pkl = 'pkl'
    else:
        pkl = pkl
    __file_name = f'{name}'
    _file_name = str(__file_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
    file_name = os.path.join(directory, 'ILM_data_files/single_rational_data', f'{_file_name}.pkl')
    return file_name
#%% READ CONSTANTS FROM CONFIG
possible_languages = config.language.possible_languages
language_type = config.language.language_type
epsilon = config.constants.epsilon
signals = config.language.signals
meanings = config.language.meanings
expressivity = config.constants.expressivity
MAP = config.constants.MAP
bottlerange = config.constants.bottlerange
generations = config.constants.generations
iterations = config.constants.iterations
prior_type = config.prior_constants.prior_type
model = config.model
initial_language = config.constants.initial_language
if type(initial_language) != int:
    initial_language= '-'.join(str(x*100) for x in initial_language)
fname = '_'.join(str(x) for x in (model, prior_type, expressivity, MAP, generations, iterations, initial_language))
#%% LOAD DATAFILE
#fname = ''
try:
    dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
except:
    raise ValueError('NO DATAFILE FOUND')
#%%

# def get_stats(dataframe):
#     # Populations are infinitely large, organised into discrete generations, and each individual learns from a single model at the previous generation
#     # treat each simulation as an infinitely large population represented by the posterior distribution
#     # or weighted posterior distribution too?

#     #average posterior distribution in each generation across all iterations of the simulation
#     for b in dataframe.keys(): #key is bottleneck
#         posterior_type_evolution_accumulator = []
#         for iteration in dataframe[b]['posterior']:
#             posterior_type_evolution = []
#             for t in range(len(set(language_type))):
#                 #find posterior type distribution  in each generation for a given iteration
#                 indices = np.where(np.array(language_type) == t)[0]
#                 posterior_type_evolution.append([logsumexp([gen_post[index] for index in indices]) for gen_post in iteration])
#             posterior_type_evolution_accumulator.append(posterior_type_evolution)
        
#         #find proportion evolution
#         dataframe[b]['average_posterior_evolution'] = [np.mean(np.exp([iteration[t] for iteration in posterior_type_evolution_accumulator]), axis=0) for t in range(len(set(language_type)))]
#         dataframe[b]['final_proportion'] = [dataframe[b]['average_posterior_evolution'][t][-1] for t in range(len(set(language_type)))]

#     return dataframe

def get_stats(dataframe):
    for b in dataframe.keys():
        dataframe[b]['average_proportion_evolution'] = np.mean(dataframe[b]['proportions'], axis=0)
        dataframe[b]['std_proportion_evolution'] = np.std(dataframe[b]['proportions'], axis=0)
    return dataframe

def plot_bottleneck_proportions(dataframe):
    for t in range(4):
        #plt.plot(bottlerange, [dataframe[b]['average_proportion_evolution'][-1][t] for b in dataframe.keys()], color = config.plotting_params.colors[t])#, label = config.plotting_params.labels[t])
        plt.errorbar(bottlerange, [dataframe[b]['average_proportion_evolution'][-1][t] for b in dataframe.keys()], yerr = [dataframe[b]['std_proportion_evolution'][-1][t] for b in dataframe.keys()], color = config.plotting_params.colors[t], label = config.plotting_params.labels[t])
    plt.legend()
    plt.xlabel('Bottleneck Size')
    plt.ylabel('Final Proportion (%s gens.)' % (generations))
    plt.show()

# %%
dataframe = get_stats(dataframe)
# %%
plot_bottleneck_proportions(dataframe)
# %%
