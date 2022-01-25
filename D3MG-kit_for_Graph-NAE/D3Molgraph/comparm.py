import rdkit 
from rdkit import Chem
MAX_ATOMIC_NUM=100

ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

class NNsetting:
    def __init__(self):
        self.hidden_size=300
        self.bias=False
        self.depth=3
        self.dropout=0.0
        self.activation='tanh' #choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU']
        self.undirected=False
        self.dense_hiddensize=self.hidden_size
        self.dense_layers=2
        self.atom_messages=False
        self.atom_fdim=ATOM_FDIM
        self.bond_fdim=BOND_FDIM
        self.cuda=True 
        self.GIERCM_dim=0
        self.Emodelstruc=[256,256,256]
        self.lantdim=128
        self.batchsize=256
        self.optimizer='Adam'
        self.lrsch='ReduceLR'
        self.Lossw={'E':0.000000,'KL':0.000000,'Re':1}
        self.initlr=0.0001
class GPARAMS:
    def __init__(self):
        self.NNsetting=NNsetting()

GP=GPARAMS()
