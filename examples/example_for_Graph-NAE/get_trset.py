import numpy as np
from tqdm import tqdm 
import math 
from D3Molgraph.Datastruc import * 
from D3Molgraph.Model import * 
import argparse as arg

parser=arg.ArgumentParser(description='Different structure description for structure generation')
#parser.add_argument('--noise_in_train')
parser.add_argument('--sysname')
parser.add_argument('--cuda')
args=parser.parse_args()
#noise_percentage=float(args.noise_percentage)
#noise_in_train=bool(int(args.noise_in_train))
sysname=args.sysname
cuda=args.cuda
groupnum=5
selectnum=50
Trset=[]
for i in range(groupnum):
    with open(f'{sysname}_part{i}_tr.pickle','rb') as f:
        mols=pickle.load(f) 
        print (len(mols))
        newmols=random.sample(mols,10)
        print (len(newmols))
        Trset+=newmols
        print (len(Trset))
with open(f'{sysname}_tr.pickle','wb') as f:
    pickle.dump(Trset,f)

