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

os.environ["CUDA_VISIBLE_DEVICES"]=cuda

with open(f'{sysname}_GIERCM.pickle','rb') as f:
     X_train=pickle.load(f)
maxlen=0
for feature in X_train:
    tmpmax=np.max([len(f) for f in feature]) 
    if tmpmax > maxlen:
        maxlen=tmpmax
with open(f'{sysname}_te.pickle','rb') as f:
    Temollist=pickle.load(f)
for mol in Temollist:
    feature=mol.coordinate_feature_on_2D_graph
    tmpmax=np.max([len(f) for f in feature]) 
    if tmpmax > maxlen:
        maxlen=tmpmax
for i in range(len(X_train)):
    X_train[i]=[np.pad(x,(0,maxlen-len(x)),'constant',constant_values=0) for x in X_train[i]]
X_train=np.array(X_train)

print (maxlen,len(X_train))
Teset=MGSet(f'{sysname}_teset',mollist=Temollist)    
Teset.prepare(ifscale=False,fmaxlength=maxlen)
with open(f'{sysname}_tr.pickle','rb') as f:
    Trmollist=pickle.load(f)
Trset=MGSet(f'{sysname}_trset',mollist=Trmollist)    
Trset.prepare(ifscale=False,fmaxlength=maxlen)

"""
GIE-RCM test
"""
print  ('='*80)
print ('='*35,'GIE-RCM','='*25)
for noise_percentage in [0,0.05,0.1,0.25,0.5]:
    noise_in_train=1
    AEmodel=Convolutional_Noise_Autoencoder(x=X_train,dataname=f'{sysname}_GIERCM_{noise_percentage}_{noise_in_train}',lantentdim=256,batchsize=256,noise_percent=noise_percentage)
    print (AEmodel.inputdim)
    cutnum=math.ceil(len(X_train)*0.9)
    for i in range(10):
        AEmodel.fit(x=X_train[:cutnum],valx=X_train[cutnum:],epochnum=200,lr=0.0001,with_noise=noise_in_train)
        AEmodel.evaluate_molgraphrmsd(Trset,with_noise=False,savedir='Train')
        AEmodel.evaluate_molgraphrmsd(Teset,with_noise=False,savedir='Test')
    AEmodel.save()

"""
noise_in_train=0
AEmodel=Convolutional_Noise_Autoencoder(x=X_train,dataname=f'{sysname}_GIERCM_0_{noise_in_train}',lantentdim=256,batchsize=256,noise_percent=noise_percentage)
cutnum=math.ceil(len(X_train)*0.9)
print ('*'*40+"AutoEncoder"+'*'*40)
for i in range(10):
    AEmodel.fit(x=X_train[:cutnum],valx=X_train[cutnum:],epochnum=200,lr=0.0001,with_noise=noise_in_train)
for noise_percentage in [0,0.05,0.1,0.25,0.5]:
    print ('='*35+'Noise percentage'+'=')
    noise_in_train=1
    AEmodel.evaluate_molgraphrmsd(Trset,with_noise=True,savedir=f'Train_{noise_percentage}',noise_percent=noise_percentage)
    AEmodel.evaluate_molgraphrmsd(Teset,with_noise=True,savedir=f'Test_{noise_percentage}',noise_percent=noise_percentage)
AEmodel.save()
"""
