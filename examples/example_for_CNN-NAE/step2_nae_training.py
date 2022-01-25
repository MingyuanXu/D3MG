import numpy as np
from tqdm import tqdm 
import math 
from D3Molgraph.Datastruc import * 
from D3Molgraph.Model import * 
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='0'
with open('plp_gap.pickle','rb') as f:
    Mollist=pickle.load(f)
PLP_MGSet=MGSet(name='plp_dataset',mollist=Mollist)
PLP_MGSet.prepare(ifscale=False)

Trset,Teset=PLP_MGSet.split(0.9)
X_train=[]
X_test=[]
for mol in tqdm(Trset.molgraphs):
    mol.EGCM_and_Rank_on_2D_graph()
    feature_matrix=mol.order_graph_node_coordinate_feature_on_2D_graph()
    X_train.append(feature_matrix)

for mol in tqdm(Teset.molgraphs):
    mol.EGCM_and_Rank_on_2D_graph()
    feature_matrix=mol.order_graph_node_coordinate_feature_on_2D_graph()
    X_test.append(feature_matrix)

X_train=np.array(X_train)
X_test=np.array(X_test)

print (len(Trset.molgraphs),len(Teset.molgraphs))
AEmodel=Convolutional_Noise_Autoencoder(x=X_train,dataname='PLP',lantentdim=256,batchsize=128,noise_percent=0.3)
cutnum=math.ceil(len(X_train)*0.9)

for i in range(8):
    AEmodel.fit(x=X_train[:cutnum],valx=X_train[cutnum:],epochnum=50,lr=0.0001,with_noise=True)
    AEmodel.evaluate_molgraphrmsd(Teset,with_noise=True)
AEmodel.save()
"""
AEmodel=Convolutional_Noise_Autoencoder(modelname='Model_For_PLP')
AEmodel.evaluate_molgraphrmsd(Teset,with_noise=True)
AEmodel.Interpolation()
"""
