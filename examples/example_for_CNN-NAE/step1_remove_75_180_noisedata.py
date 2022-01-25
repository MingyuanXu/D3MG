import numpy as np
from tqdm import tqdm 
import math 
from D3Molgraph.Datastruc import *
from D3Molgraph.Base import * 
from D3Molgraph.Model import * 
import pickle 
import os
with open('plp.pickle','rb') as f:
    Mollist=pickle.load(f)
tor1_list=[4,7,9,10]
tor2_list=[7,9,10,11]
tor_coord=[]
flag=True
lessmol=None 
moremol=None 
Newmollist=[]
for mol in Mollist:
    coord1=np.array([crd for i,crd in enumerate(mol.d3coord) if i in tor1_list])
    coord2=np.array([crd for i,crd in enumerate(mol.d3coord) if i in tor2_list])
    tor1=cal_torsion_angle(coord1)
    tor2=cal_torsion_angle(coord2)
    if tor2<0:
        tor2=360+tor2
    if tor2<75 or tor2> 140:
        Newmollist.append(mol)
with open('plp_gap.pickle','wb') as f:
    pickle.dump(Newmollist,f)

