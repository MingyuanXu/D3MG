import pickle 
from D3Molgraph import * 
from D3Molgraph.Datastruc import * 
from D3Molgraph.Model import * 
import math 
import os 
totalnum=2500
groupnum=math.ceil(totalnum/500)
print (groupnum)
def generate_mollist(ids):
    mollist=[]
    datasetpath='./sdf_dataset'
    for id in range(ids*500+1,(ids+1)*500+1):
        os.system(f'obabel -ipdb {datasetpath}/equil.pdb.{id} -o pdb -O {datasetpath}/{id}.pdb --DelNonPolarH 2> /dev/null')
        #os.system(f'obabel -ipdb {datasetpath}/equil.pdb.{id} -o pdb -O {datasetpath}/{id}.pdb --DelNonPolarH 2> /dev/null')
        try:
            mol=Molgraph(pdbname=f'{datasetpath}/{id}.pdb')
            mol.build_Zmatrix_on_2D_graph()
            if mol:
                 print (mol.n_atoms)
                 mollist.append(mol)
        except Exception as e:
            print (mol.name)

    with open(f'PLP_part{ids}.pickle','wb') as f:
        pickle.dump(mollist,f)
    return 

from multiprocessing import Pool,Queue,Manager,Process
from tqdm import tqdm

manager=Manager()
dataQueue=manager.Queue()
p=Pool(80)
resultlist=[]
for i in range(groupnum):
    result=p.apply_async(generate_mollist,(i,))
    resultlist.append(result)
p.close()
for i in tqdm (range(len(resultlist))):
    tmp=resultlist[i].get()
    print (tmp)
p.terminate()
p.join()
dataQueue.put(None)

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
