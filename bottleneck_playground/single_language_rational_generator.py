#%%
import os
import numpy as np
import module_single_language_rational as rm
from module_prior import get_prior, plot_prior
from multiprocessing import Pool, cpu_count
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
    file_name = os.path.join(directory, 'ILM_data_files/single_rational_data', f'{_file_name}.pkl')
    return file_name

#%% READ CONSTANTS FROM CONFIG
generations = config.constants.generations
expressivity = config.constants.expressivity
MAP = config.constants.MAP
bottlerange = config.constants.bottlerange
iterations = config.constants.iterations
prior_type = config.prior_constants.prior_type
print("""
      ===============================================
      BEGIN SIMULATIONS WITH THE FOLLOWING CONSTANTS
      ===============================================
      -----------------------------------------------
      MAP: %s
      bottleneck range: %s - %s, %s steps
      expressivity: %s
      iterations: %s
      generations: %s
      prior_type: %s
      -----------------------------------------------
      """ % (MAP, bottlerange[0], bottlerange[-1], len(bottlerange), expressivity, iterations, generations, prior_type))
fname = '%s_%s_%s_%s_%s' % (prior_type, expressivity, MAP, generations, iterations)
#%% INITALISE PRIOR
prior = get_prior()
plot_prior(prior)
#%% DEFINE GENERATION FUNCTION
def simulation(b):
    #language_agent, posterior_agent, _ = rm.iterate(prior, bottleneck=b)
    result = rm.iterate(prior, bottleneck=b)
    return result #language_agent, posterior_agent
#%%
start = time.perf_counter()
sim = []
for b in tqdm([(b) for b in bottlerange for _ in range(iterations)]):
    sim.append(simulation(b))
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

# print('Time elapsed: %s'% (time.perf_counter()-start))
# CONFIGURE DATAFRAME
dataframe = {b: {} for b in bottlerange}
left=0
for b in bottlerange:
    #raw = sim[left:left+iterations]
    #dataframe[b]['language_type'] = [lang for lang, _ in raw]
    #dataframe[b]['posterior'] = [post for _, post in raw]
    dataframe[b]['proportions'] = sim[left:left+iterations]
    left+=iterations

# Save file
f = open(f'{file_id(fname)}', 'wb')
pickle.dump(dataframe, f)
f.close()


# %%
