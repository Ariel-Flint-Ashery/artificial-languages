"""
NOTE: MOST OF THE PARAMETERS CAN BE EDITED DIRECTLY IN THE CONFIG.YAML FILE. THIS SCRIPT SHOULD BE ONLY BE USED TO GENERATE NEW LANGUAGES.
"""
#%%
# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(dir_path)
import yaml
from utils import characterise_language, generate_language
from munch import munchify
import numpy as np
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
# def set_variable(var_name, var):
#     with open('config.yaml') as f:
#         doc = yaml.load(f)

#     doc[var_name] = var

#     with open('config.yaml', 'w') as f:
#         yaml.dump(doc, f)

def set_variable(var_name, var, var_group=None):
    file_name = "config.yaml"
    with open(file_name, 'r') as f:
        doc = yaml.safe_load(f)
    if var_group==None:
        try:
            doc[var_name] = var
        except:
            return TypeError('incorrect input')
    else:
        try:
            doc[var_group][var_name]=var
        except:
            return TypeError('incorrect input')
    with open(file_name, 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

#%%
meanings, signals, possible_languages = generate_language(vocabulary_size=config.language.vocabulary_size, word_size = config.language.word_size)
language_type = [characterise_language(data) for data in possible_languages]

#update config
set_variable('meanings', meanings, 'language')
set_variable('signals', signals, 'language')
set_variable('language_type', language_type, 'language')
set_variable('possible_languages', possible_languages, 'language')
# %%
bottlerange = np.arange(1,101,1).tolist()
set_variable('bottlerange', bottlerange, 'constants')
# %%
