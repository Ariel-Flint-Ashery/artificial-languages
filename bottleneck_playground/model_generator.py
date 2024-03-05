#%%
import os
import numpy as np
#import module_one_pass as rm
from module_prior import get_prior, plot_prior
#from multiprocessing import Lock, Process, Queue, current_process
import time
import yaml
import pickle
from munch import munchify
from tqdm import tqdm
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
    file_name = os.path.join(directory, 'ILM_data_files', f'{_file_name}.pkl')
    return file_name

#%% READ CONSTANTS FROM CONFIG
generations = config.constants.generations
expressivity = config.constants.expressivity
MAP = config.constants.MAP
bottlerange = config.constants.bottlerange
iterations = config.constants.iterations
prior_type = config.prior_constants.prior_type
initial_language = config.constants.initial_language
model = config.model
initial_language_name = initial_language
if type(initial_language) != int:
    initial_language_name= '-'.join(str(int(x*100)) for x in initial_language)
print(f"""
      ===============================================
      BEGIN SIMULATIONS WITH THE FOLLOWING PARAMETERS
      ===============================================
      -----------------------------------------------
      Model: {model}
      MAP: {MAP}
      bottleneck range: {bottlerange[0]} - {bottlerange[-1]}, {len(bottlerange)} steps
      expressivity: {expressivity}
      iterations: {iterations}
      generations: {generations}
      prior type: {prior_type}
      initial language composition: {initial_language_name}
      -----------------------------------------------
      """) 
fname = '_'.join(str(x) for x in (model, prior_type, expressivity, MAP, generations, iterations, initial_language_name))
#%% GET MODEL
if model == 'one_pass':
    import module_one_pass as ilm
if model == 'kirby_2021':
    import module_kirby_2021 as ilm
if model == 'mixed_population':
    import module_mixed_population as ilm
if model == 'mixed_networked_population':
    import module_mixed_networked_population as ilm
if model == 'CRP':
    import module_burkett as ilm
#%% INITALISE PRIOR
prior = get_prior()
plot_prior(prior)
#%% DEFINE GENERATION FUNCTION
def simulation(b):
    result = ilm.iterate(prior, bottleneck=b)
    return result 
# def parallel_simulation(b_values):
#     num_processes = multiprocessing.cpu_count()-4  # Get the number of available CPU cores
#     pool = multiprocessing.Pool(processes=num_processes)

#     results = pool.map(simulation, b_values)
#     pool.close()
#     pool.join()

#     return results
#%%
start = time.perf_counter()
simulation_results = []
for b in tqdm([(b) for b in bottlerange for _ in range(iterations)]):
    simulation_results.append(simulation(b))
print('Time elapsed: %s'% (time.perf_counter()-start))

# start = time.perf_counter()
# if __name__ == "__main__":
#     # print("""
#     #       -----------------------------
          
#     #           STARTING MULTIPROCESS
          
#     #       -----------------------------
#     #       """)
#     with Pool(processes=4) as pool: #cpu_count()-1
#         sim = pool.map(simulation, [(b) for b in bottlerange for _ in range(iterations)])

# Create a list of b values repeated for the specified number of iterations
# b_values = [b for b in bottlerange for _ in range(iterations)]

# # Run the simulation in parallel
# simulation_results = parallel_simulation(b_values)

# print('Time elapsed: %s'% (time.perf_counter()-start))
# CONFIGURE DATAFRAME
dataframe = {b: {} for b in bottlerange}
left=0
for b in bottlerange:
    #raw = sim[left:left+iterations]
    #dataframe[b]['language_type'] = [lang for lang, _ in raw]
    #dataframe[b]['posterior'] = [post for _, post in raw]
    dataframe[b]['language'] = simulation_results[left:left+iterations]#[0]
    #dataframe[b]['signals'] = simulation_results[left:left+iterations][1]
    #dataframe[b]['signals_std'] = simulation_results[left:left+iterations][1][1]
    left+=iterations

# Save file
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(dataframe, f)
f.close()


# %%
