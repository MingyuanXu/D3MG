import pickle 
from D3Molgraph import * 
from D3Molgraph.Datastruc import * 
from D3Molgraph.Model import * 
import math 
import os 
from tqdm import tqdm 
groupnum=5
sysname='PLP'
torsiondict={}
for i in tqdm(range(groupnum)):
    with open(f'{sysname}_part{i}_2.pickle','rb') as f:
        mollist=pickle.load(f)
        for m,mol in enumerate(mollist):
                torsiondict[mol.name]=mol.properties['backbone torsion']
                if len(mol.properties['backbone torsion']) !=18:
                    print (len(mol.properties['backbone torsion']),mol.name)
torlist=np.array([np.array(torsiondict[key]) for key in torsiondict.keys()])
print (torlist)
idlist=[key for key in torsiondict.keys()]
period=360
BigM=np.zeros((len(torlist),len(torlist)))
for i in tqdm(range(len(torlist[0]))):
    tori=torlist[:,i]
    distA=pdist(np.reshape(tori,(-1,1)),metric='euclidean')
    distB=squareform(distA)
    M=np.where(distB>0.5*period,period-distB,distB)
    BigM+=np.square(M)
BigM=np.sqrt(BigM)
print (BigM.shape)
with open('torsion_distance.pickle','wb') as f:
    pickle.dump((torlist,idlist,BigM),f)
with open('torsion_distance.pickle','rb') as f:
    torlist,idlist,BigM=pickle.load(f)
    deps=40
    dbscan_cluster=DBSCAN(eps=deps,metric='precomputed').fit_predict(BigM)
    print (Counter(dbscan_cluster))
    while Counter(dbscan_cluster)[-1]>len(torlist)*0.2:
        deps+=10
        dbscan_cluster=DBSCAN(eps=deps,metric='precomputed').fit_predict(BigM)
        print (Counter(dbscan_cluster))
with open('cluster_info.pickle','wb') as f:
    pickle.dump((idlist,dbscan_cluster),f)
clusterdict={}
for i in range(len(idlist)):
    clusterdict[idlist[i]]=dbscan_cluster[i]
Teset=[]
X_train_GIERCM=[]
X_train_rcartesian=[]
X_train_dcartesian=[]
X_train_zmatrix=[]
for i in tqdm(range(groupnum)):
    with open(f'{sysname}_part{i}_2.pickle','rb')  as f:
        Trset=[]
        mollist=pickle.load(f)
        for mol in mollist:
            try:
                mol.properties['cluster id']=clusterdict[mol.name]
                feature_matrix=mol.order_graph_node_coordinate_feature_on_2D_graph()
                rcoord=mol.build_d3coord_on_2D_graph()
                if clusterdict[mol.name]==-1:
                    Teset.append(mol)
                else:
                    Trset.append(mol)
                    X_train_GIERCM.append(feature_matrix)
                    X_train_rcartesian.append(rcoord)
                    X_train_dcartesian.append(mol.d3coord)
                    X_train_zmatrix.append([mol.zbond,mol.zangle,mol.zdihedral])
            except Exception as e:
                print (mol.name,' is failed due to ',e)
    with open(f'{sysname}_part{i}_tr.pickle','wb') as f:
        pickle.dump(Trset,f)
with open(f'{sysname}_te.pickle','wb') as f:
     pickle.dump(Teset,f)
with open(f'{sysname}_GIERCM.pickle','wb')  as f:
    pickle.dump(X_train_GIERCM,f)
with open(f'{sysname}_rcartesian.pickle','wb')  as f:
    pickle.dump(X_train_rcartesian,f)
with open(f'{sysname}_dcartesian.pickle','wb')  as f:
    pickle.dump(X_train_dcartesian,f)
with open(f'{sysname}_zmat.pickle','wb')  as f:
    pickle.dump(X_train_zmatrix,f)
        
