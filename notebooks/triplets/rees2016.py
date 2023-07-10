 #%%
from igraph import *
from bluepy.enums import Cell
from scipy import sparse
import bluepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logging

 #%%

adj_file = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/hippdiss-422/data/ca3_ca1_block_matrix.npz' #bool matrix
all_motifs = ['-C','-B','-A','A','B','C','D','E','F','G','H','I','J','K','L','M']
target = 'cylinder300'