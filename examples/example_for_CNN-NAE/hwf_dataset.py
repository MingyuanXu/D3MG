import numpy as np
from tqdm import tqdm 
import math 
from D3Molgraph.Datastruc import *
from D3Molgraph.Base import * 
from D3Molgraph.Model import * 
import pickle 
import os
with open('plp_gap.pickle','rb') as f:
    Mollist=pickle.load(f)
    print (Mollist[0].__dict__.keys())
