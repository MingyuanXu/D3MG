import os
import pickle
import numpy as np
from tqdm import tqdm
import math
from D3Molgraph.Datastruc import *
from D3Molgraph.Base import *
from D3Molgraph.Model import *

os.environ["CUDA_VISIBLE_DEVICES"]='2'

mollist=[]
for i in range(5):
    with open(f'PLP_part{i}_tr.pickle','rb') as f:
            mollist+=pickle.load(f)
pro_MGSet=MGSet(name='PLP',mollist=mollist)
pro_MGSet.prepare()

tor1_list=[4,7,9,10]
tor2_list=[7,9,10,11]
flag=True
'''
with open(f'cluster_info.pickle','rb') as f:
    idlist,dbscan_cluster=pickle.load(f)
clusterdict={}
for i in range(len(idlist)):
    clusterdict[idlist[i]]=dbscan_cluster[i]
#print(cluster[1])
'''
lessmollist=[]
moremollist=[]
'''
for mol in pro_MGSet.molgraphs:
    if clusterdict[mol.name]==0:
        lessmollist.append(mol)
    if clusterdict[mol.name]==5:
        moremollist.append(mol)
'''
tor_coord=[]
tor_coord1=[]
tor_coord2=[]
for mol in pro_MGSet.molgraphs:
    coord1=np.array([crd for i,crd in enumerate(mol.d3coord) if i in tor1_list])
    coord2=np.array([crd for i,crd in enumerate(mol.d3coord) if i in tor2_list])
    tor1=cal_torsion_angle(coord1)
    tor2=cal_torsion_angle(coord2)
    if tor2<0:
        tor2=360+tor2
    tor_coord.append([tor1,tor2])
    if tor2<80:
        tor_coord2.append([tor2,tor1])
    if tor2>150:
        tor_coord1.append([tor2,tor1])
    
    if tor2<70 and tor2 >50 and tor1<80 and flag:
        lessmollist.append(mol)

    if tor2>150 and flag and tor1>40 and tor1 < 80:
        moremollist.append(mol)

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt 
import seaborn as sns 
import random 
moremollist=random.choices(moremollist,k=100)
plt.rc('font',family='Times New Roman',size=15)
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'
tor_coord=np.array(tor_coord)
tor_coord1=np.array(tor_coord1)
tor_coord2=np.array(tor_coord2)
figure=plt.figure(figsize=(12,5))
ax=sns.kdeplot(tor_coord1[:,0],tor_coord1[:,1],cmaps='Blues',shade=True,shade_lowest=False,n_levels=10)
ax=sns.kdeplot(tor_coord2[:,0],tor_coord2[:,1],cmaps='red',shade=True,shade_lowest=False,n_levels=10)
plt.scatter(tor_coord1[:,0],tor_coord1[:,1],color='cyan',s=15,alpha=0.5,edgecolors='cyan')
plt.scatter(tor_coord2[:,0],tor_coord2[:,1],color='orange',s=15,alpha=0.5,edgecolors='orange')
plt.ylim(20,120)
plt.xlim(20,220)
plt.yticks(np.arange(20,121,20))
plt.xticks(np.arange(20,221,20))
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
num=0

#GP.NNsetting.GIERCM_dim=Model.model.GIERCM_encoder.input_dim
GP.NNsetting.GIERCM_dim=48
Model=Graph_DCGM_Model(modelname='PLP')


times=0
flag=True
id1list=[]
id2list=[]
ii=0
jj=0
torcrdlist=[]

for ii,mol1 in enumerate(lessmollist):
    for jj,mol2 in enumerate(moremollist):
        if mol1.connection_weight==mol2.connection_weight and flag and ii not in id1list and jj not in id2list:
            id1list.append(ii)
            id2list.append(jj)
            times+=1
            mol1.build_Zmatrix_on_2D_graph()
            conf_dict=Model.Interpolation(mol1,mol2,20,path=f'Interpolation_0-5/{times}')
            if times>39:
                flag=False
            for i in range(20):
            #   os.system(f'mkdir -p Interpolation/Interpolation_{num}')
                d3coord=conf_dict["interval"][i]
            #   write_xyz(f'Interpolation/Interpolation_{num}/1le1_{i}.xyz',mol1.atoms,d3coord,0)
                coord1=np.array([crd for i,crd in enumerate(d3coord) if i in tor1_list])
                coord2=np.array([crd for i,crd in enumerate(d3coord) if i in tor2_list])
                tor1=cal_torsion_angle(coord1)
                tor2=cal_torsion_angle(coord2)
                if tor2<0:
                    tor2=360+tor2
                if tor1>50 and tor1<75 and tor2>60:
                    torcrdlist.append([tor2,tor1])

torcrdlist=np.array(torcrdlist)
plt.scatter(torcrdlist[:,0],torcrdlist[:,1],color='green',s=15,alpha=0.5,edgecolors='green')
plt.savefig('kde.png',dpi=300)
plt.show() 
