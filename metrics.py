# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:58:08 2023

@author: ariel
"""

#### TEXT METRICS AND EVALUATION

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from datasets import load_dataset
import torch
import random
from transformers import BartForConditionalGeneration, AutoTokenizer, set_seed
import math
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
import scipy as sp
#%%

vectors = np.random.rand(10,2)
pdist = sp.spatial.distance.pdist(vectors)

#%%

def gen_embeddings(text_list):
    
    embeds = []
    for text in text_list:
        embeds.append(model.encode(text))
        
    embeds = np.array(embeds)
    
    return embeds

def embed_distance(embeddings, metric = 'euclidean'):
    """
    Parameters
    ----------
    embeddings : ndarray
        Array containing all embedding from a given generation.
    metric : STR or FUNCTION, optional
        The distance metric to use. The distance function can be 
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’,
        ‘yule’. The default is 'euclidean'.

    Returns
    -------
    dist : average distance between embeddings

    """
    
    pdist = sp.spatial.distance.pdist(embeddings, metric=metric)
    
    dist = sum(pdist)/len(pdist)
    
    return dist
    
def measure_compositionality(m,s, metric = 'euclidean', corr = 'pearson'):
    
    