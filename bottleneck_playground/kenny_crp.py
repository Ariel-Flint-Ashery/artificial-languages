# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy.special import gamma, gammaln
from math import log,exp

"""
A crp partition is represented by three lists: a list of counts for clusters, a list of labels for those clusters, and a list of assignments for
objects indicating which table they belong to.
Base gives the probability of each label (i.e. the nth element of base gives the probability of label n).
Uses gamma function from numpy, but the numpy types there were causing problems, so cast the result of gamma function to a normal python float.
Using logs is probably a better idea though.
"""
def crp_probability(counts,labels,alpha,base):
    k = len(counts)
    n = np.sum(counts)
    partition_p_numerator = (alpha ** k) * float(gamma(alpha)) * np.prod([float(gamma(count)) for count in counts])
    partition_p_denominator = float(gamma(alpha + n))
    partition_p = partition_p_numerator / partition_p_denominator
    label_p = np.prod([[base[label] for label in labels]])
    return partition_p * label_p


"""
Same calculation, but in the log domain.
Note: assumes base is in log_domain.
"""
def crp_logprobability(counts,labels,alpha,base):
    logalpha = log(alpha)
    k = len(counts)
    n = np.sum(counts)
    partition_p_numerator = (logalpha * k) + gammaln(alpha) + np.sum([float(gammaln(count)) for count in counts])
    partition_p_denominator = gammaln(alpha + n)
    partition_p = partition_p_numerator - partition_p_denominator
    label_p = np.sum([[base[label] for label in labels]])
    return partition_p + label_p

"""
Generates a random crp given alpha, max number of paritions n, and base distribution over labels.
From http://www.psychology.adelaide.edu.au/personalpages/staff/danielnavarro/ccs/technote7_crp.pdf, page 3
code for generating labels added by Kenny

"""
def crp_sample(alpha,n,base):
    assignments = [0 for i in range(n)] # assignments for each of n objects. 
    counts = [0 for i in range(n+1)]; # table counts (n is the max possible K)
    assignments[0] = 0 # assign object 0 to table 0
    counts[0] = 1; # adjust counts
    counts[1] = alpha # "fake" counts for table K+1
    k = 1 # number of unique clusters
    # sequentially assign other objects via CRP
    for i in range(1,n):
        # generate random number, and convert to a "quasi-count"
        u = random.random() # generate uniform random number
        u = u * (i + alpha) # multiply by the CRP normalising constant
        # find the corresponding table
        z = 0 # indexing variable for the table
        while u > counts[z]:
            u = u - counts[z] # subtract off that probability mass
            z = z + 1 #move to the next table
            # record the outcome and adjust
        assignments[i] = z # make the assignment
        if z == k: # if it’s a new table
            counts[z]= 1# assign real count
            counts[z+1] = alpha; # move the "fake"
            k = k+1 #update the number of clusters
        else:
            counts[z] = counts[z] + 1 # increment count
        # truncate the counts matrix for neatness. also, this takes % care of the "fake" count mass in count(K+1)
    counts = counts[:k]
    labels = ["" for i in range(k)]
    for z in range(k):
        labels[z]=random_label(base)
    return counts, assignments, labels

def random_label(distribution):
    r = random.random() # generate uniform random number in [0,1).  Note that distribution assumed to sum to 1
    # find the corresponding table
    i = 0 # indexing variable
    while r > distribution[i]:
        r = r - distribution[i] # subtract off that probability mass
        i = i + 1 #move to the next label
    return i


"""
This is mainly to check that the code doesn't crash, and that the log-domaon version produces the same result as the non-log one
"""
def testcrp(alpha,n,base,reps):
    for _i in range(reps):
        counts,_assignments,labels = crp_sample(alpha,n,base)
        crp_p = crp_probability(counts,labels,alpha,base)
        crp_logp = crp_logprobability(counts,labels,alpha,[log(p) for p in base]) 
        print(crp_p,exp(crp_logp))

