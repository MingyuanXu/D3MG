import pickle 
from D3Molgraph import * 
from D3Molgraph.Datastruc import * 
from D3Molgraph.Model import * 
import math 
import os 
from tqdm import tqdm 
backbonelist=[]

def cal_torsion_descriptor(id):
    newmollist=[]
    with open(f'PLP_part{id}.pickle','rb') as f:
        mollist=pickle.load(f)
        for mol in tqdm(mollist):
            #try:
                mol.properties['backbone torsion']=mol.zdihedral
                print (len(mol.zdihedral))
                newmollist.append(mol)
            #except:
            #    pass
    with open(f'PLP_part{id}_2.pickle','wb') as f:
        pickle.dump(newmollist,f)
    return 

from multiprocessing import Pool,Queue,Manager,Process
from tqdm import tqdm

manager=Manager()
dataQueue=manager.Queue()
p=Pool(5)
resultlist=[]
groupnum=5
for i in range(groupnum):
    result=p.apply_async(cal_torsion_descriptor,(i,))
    resultlist.append(result)
p.close()
for i in tqdm (range(len(resultlist))):
    tmp=resultlist[i].get()
    print (tmp)
p.terminate()
p.join()
dataQueue.put(None)
#receive_process.join()
"""
totalmollist=[]
for i in tqdm(range(groupnum)):
    with open(f'PLP_part{i}.pickle','rb') as f:
        mollist=pickle.load(f)
        totalmollist+=mollist

import random 
random.shuffle(totalmollist)
protein_MGSet=MGSet('PLP_dataset',mollist=totalmollist)
protein_MGSet.prepare(ifscale=False)
fmaxlength=protein_MGSet.node_feature_max_length
name=protein_MGSet.name
protein_MGSet=None
cutnum=math.ceil(len(totalmollist)*0.95)
Trset=MGSet(name+'trset',mollist=totalmollist[:cutnum])
Trset.prepare(ifscale=False,fmaxlength=fmaxlength)
with open('PLP_trset.pickle','wb') as f:
    pickle.dump(Trset,f)
totalmollist=totalmollist[cutnum:]
Teset=MGSet(name+'teset',mollist=totalmollist)
Teset.prepare(ifscale=False,fmaxlength=fmaxlength)
with open('PLP_teset.pickle','wb') as f:
    pickle.dump(Teset,f)

"""
