#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import pickle
import yaml
from munch import munchify
import matplotlib.pyplot as plt
import math
#%% INITALISE CONFIG
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

def file_id(name, folder, directory = None):
    """
    Returns:
        Returns the file name with all the relevant directories
    """
    if directory == None:
        dir_path = os.path.dirname(os.path.realpath(__file__))

        directory = dir_path
    else:
        directory = directory
    __file_name = f'{name}'
    _file_name = str(__file_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
    __folder_name = f'{folder}'
    _folder_name = str(__folder_name).replace(' ', '-').replace(',', '').replace('[', '-').replace(']','-').replace('.','-')
    file_name = os.path.join(directory, 'ILM_data_files', f'{_folder_name}_{_file_name}.npy')
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
    initial_language= '-'.join(str(int(x*100)) for x in initial_language)
foldername = '_'.join(str(x) for x in (model, prior_type, expressivity, MAP, generations, iterations, initial_language))
#%% LOAD DATAFILE
#fname = ''
# try:
#     dataframe = pickle.load(open(f'{file_id(fname)}', 'rb'))
# except:
#     raise ValueError('NO DATAFILE FOUND')
dataframe = {b: {} for b in bottlerange}
with open(f"{file_id(name='language', folder=foldername)}", 'rb') as f:
    for b in bottlerange:
        dataframe[b]['language'] = np.array([np.load(f) for i in range(iterations)])

# with open(f"{file_id(name='signals', folder=foldername)}", 'rb') as f:
#     for b in bottlerange:
#         dataframe[b]['signals'] = np.array([np.load(f) for i in range(36)])

# with open(f"{file_id(name='posterior', folder=foldername)}", 'rb') as f:
#     for b in bottlerange:
#         dataframe[b]['posterior'] = np.array([np.load(f) for i in range(iterations)])

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
        # dataframe[b]['average_proportion_evolution'] = np.mean(dataframe[b]['language'], axis=0)
        # dataframe[b]['error_proportion_evolution'] = np.std(dataframe[b]['language'], axis=0)/np.sqrt(iterations)
        dataframe[b]['average_proportion_evolution'] = np.mean(np.mean(dataframe[b]['language'], axis=1), axis=0)
        dataframe[b]['error_proportion_evolution'] = np.std(np.mean(dataframe[b]['language'], axis=1), axis=0)/np.sqrt(iterations)
        #dataframe[b]['average_signal_evolution'] = np.mean(dataframe[b]['signals'], axis=0)
        #dataframe[b]['error_signal_evolution'] = np.std(dataframe[b]['signals'], axis=0)/np.sqrt(iterations)
    return dataframe

def plot_proportions(dataframe):
    for t in range(4):
        #plt.plot(bottlerange, [dataframe[b]['average_proportion_evolution'][-1][t] for b in dataframe.keys()], color = config.plotting_params.colors[t])#, label = config.plotting_params.labels[t])
        # plt.errorbar(bottlerange, [dataframe[b]['average_proportion_evolution'][-1][t] for b in dataframe.keys()],
        #                 yerr = [dataframe[b]['error_proportion_evolution'][-1][t] for b in dataframe.keys()],
        #                 color = config.plotting_params.colors[t], label = config.plotting_params.labels[t],
        #                 fmt = '.',
        #                 )
        plt.errorbar(range(len(dataframe.keys())), [dataframe[b]['average_proportion_evolution'][t] for b in dataframe.keys()],
                yerr = [dataframe[b]['error_proportion_evolution'][t] for b in dataframe.keys()],
                color = config.plotting_params.colors[t], label = config.plotting_params.labels[t],
                fmt = '.',
                )
    # plt.legend()
    plt.xlabel('Bottleneck Size')
    plt.ylabel('Final Proportion (%s gens.)' % (generations))
    plt.show()

def plot_proportion_evolution(dataframe, repeat=10):
    bottles = sorted(set(list(dataframe.keys())[20:60]))#+[b for b in dataframe.keys() if b % repeat==0]))
    number_of_plots = len(bottles)
    if np.sqrt(number_of_plots)-round(np.sqrt(number_of_plots)) < 0:
        nrow = math.ceil(np.sqrt(number_of_plots))
        ncol = nrow
        #if np.sqrt(pop_count)-round(np.sqrt(pop_count)) >= 0:
    else:
        nrow = math.floor(np.sqrt(number_of_plots))
        ncol = math.ceil(np.sqrt(number_of_plots))
    fig, axs = plt.subplots(nrows = nrow, ncols = ncol, figsize = (20,20))
    axs = axs.flatten()
    for b,ax in zip(bottles, axs):
        means = np.mean(dataframe[b]['language'], axis = 0)
        stds = np.std(dataframe[b]['language'], axis = 0)/np.sqrt(iterations)
        for t in range(4):
            mean = [m[t] for m in means]
            err = [e[t] for e in stds]
            ax.errorbar(range(generations), 
                            mean,
                            yerr = err,
                            color = config.plotting_params.colors[t],
                            #label = meanings[t],
                            fmt = '',
                            alpha = 0.2,
                            )
            ax.plot(range(generations),
                        mean,
                        color = config.plotting_params.colors[t],
                        label = meanings[t],
                        marker = '.',
                     )
        ax.set_title(f'b={b}')
    if len(axs)>len(bottles):
        fig.delaxes(axs[-1])
    axs[0].set_xlabel('Generations')
    axs[0].set_ylabel('Posterior')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles,ncol=len(labels),loc="lower center", bbox_to_anchor=(0.5,-0.07), fontsize = 10)
    plt.tight_layout()
    plt.show()

# %%
dataframe = get_stats(dataframe)
# %%
plot_proportions(dataframe)
#plot_signals(dataframe)
#%%
plot_proportion_evolution(dataframe, repeat = 20)
# %%
with open(f"{file_id(name='signals', folder=foldername)}", 'rb') as f:
    for b in bottlerange:
        dataframe[b]['signals'] = np.array([np.load(f) for i in range(iterations)])

#%%
def plot_signals(dataframe):
    bottles = sorted(set(list(dataframe.keys())[20:]))#+[b for b in dataframe.keys() if b % repeat==0]))
    number_of_plots = len(bottles)
    if np.sqrt(number_of_plots)-round(np.sqrt(number_of_plots)) < 0:
        nrow = math.ceil(np.sqrt(number_of_plots))
        ncol = nrow
        #if np.sqrt(pop_count)-round(np.sqrt(pop_count)) >= 0:
    else:
        nrow = math.floor(np.sqrt(number_of_plots))
        ncol = math.ceil(np.sqrt(number_of_plots))
    fig, axs = plt.subplots(nrows = nrow, ncols = ncol, figsize = (20,20))
    axs = axs.flatten()
    #vmax = np.max([np.log(np.mean(dataframe[b]['signals'][10], axis=0)) for b in bottles])
    #vmin = np.min([np.log(np.mean(dataframe[b]['signals'][10], axis=0)) for b in bottles])
    for b,ax in zip(bottles, axs):
        means = np.log(dataframe[b]['signals'][10][-1]) #np.log(np.mean(dataframe[b]['signals'][10], axis=0))

        ax.imshow(means, cmap = 'Reds', vmin = np.min(means), vmax = np.max(means))
    plt.legend()
    plt.xlabel('Bottleneck Size')
    plt.ylabel('Final Signal Proportion (%s gens.)' % (generations))
    plt.show()
# %%
plot_signals(dataframe)
 # %%
