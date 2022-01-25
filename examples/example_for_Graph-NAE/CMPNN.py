from D3Molgraph.Datastruc import * 
from D3Molgraph.Model import * 
from D3Molgraph import * 
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

mollist=[]
for i in range(5):
    with open(f'PLP_part{i}_tr.pickle','rb') as f:
        mollist+=pickle.load(f)
plpset=MGSet('PLP',mollist=mollist)
plpset.prepare()
plpset.split_random(rate=(0.8,0.9,1.0))
GP.NNsetting.GIERCM_dim=plpset.node_feature_max_length
print(GP.NNsetting.GIERCM_dim)
Model=Graph_DCGM_Model(dataset=plpset)
Model.fit(plpset,epochnum=10000,rmsd_freq=200)
Model.save()

