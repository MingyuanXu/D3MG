import torch.nn as nn 
#from ...comparm import * 
from numpy.testing._private.utils import jiffies
from rdkit import Chem
import numpy as np
from scipy  import spatial 
from copy import deepcopy 
import math 

def Transtograph(mol_batch):
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.
    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    if type(mol_batch[0])==type('smiles'):
        for smiles in mol_batch:
            if smiles in PARAM.Cmpnn_setting.SMILES_TO_GRAPH:
                mol_graph = PARAM.Cmpnn_setting.SMILES_TO_GRAPH[smiles]
            else:
                mol_graph = MolGraph(molsmiles=smiles)
                if not PARAM.Cmpnn_setting.no_cache:
                    PARAM.Cmpnn_setting.SMILES_TO_GRAPH[smiles] = mol_graph
            mol_graphs.append(mol_graph)
    elif type(mol_batch[0])==type({}):
        for moldict in mol_batch:
            mol_graph=MolGraph(moldict)
            mol_graphs.append(mol_graph)
    return BatchMolGraph(mol_graphs)

class MolGraph:
    def __init__(self,**kwargs):
        molsmiles=kwargs.get("molsmiles",'')
        moldict=kwargs.aget("moldict",{})
        if molsmiles:
            self.id = molsmiles
            self.n_atoms = 0  # number of atoms
            self.n_bonds = 0  # number of bonds
            self.f_atoms = []  # mapping from atom index to atom features
            self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
            self.a2b = []  # mapping from atom index to incoming bond indices
            self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
            self.b2revb = []  # mapping from bond index to the index of the reverse bond
            self.bonds = []
            # Convert smiles to molecule
            mol = Chem.MolFromSmiles(molsmiles)

            # fake the number of "atoms" if we are collapsing substructures
            self.n_atoms = mol.GetNumAtoms()
            # Get atom features
            for i, atom in enumerate(mol.GetAtoms()):
                if PARAM.Cmpnn_setting.one_hot:
                    self.f_atoms.append(atom_features(atom))
                else:
                    self.f_atoms.append(Feature_atom(atom))
            self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue
                    if PARAM.Cmpnn_setting.one_hot:
                        f_bond = bond_features(bond)
                    else:
                        f_bond=Feature_bond(bond)

                    if PARAM.Cmpnn_setting.atom_messages:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    else:
                        self.f_bonds.append(self.f_atoms[a1] + f_bond)
                        self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2
                    self.bonds.append(np.array([a1, a2]))
        elif len(moldict)>0:
            self.id=moldict["id"]
            self.f_atoms=moldict["f_atom"]
            self.f_bonds=moldict["f_bond"]
            self.a2b=moldict["a2b"]
            self.b2a=moldict["b2a"]
            self.b2revb=moldict["b2revb"]
            self.n_bonds=moldict["nbonds"]
            self.n_atoms=moldict["natoms"]
            self.bonds=moldict["bond"]

class ProteinGraph:
    def __init__(self,filename):
        self.id=filename 
        if filename[-4:]=='.pdb':
            self.mol=Chem.MolFromPDBFile(filename,removeHs=False)
        elif filename[-4:]=='.sdf':
            self.mol=Chem.SDMolSupplier(filename,removeHs=False)[0]
        #self.mol=Chem.RemoveHs(self.mol)
        atoms=self.mol.GetAtoms()
        self.atoms=[]
        for i,atom in enumerate(atoms):
            if PARAM.Cmpnn_setting.onehot:
                self.atoms.append(atom_features(atom))
            else:
                self.atoms.append(Feature_atom(atom))

        self.coords=self.mol.GetConformer().GetPositions()
        self.bonds=[]
        self.internal_distance_matrix()
        print (self.inter_dismat.shape)
        for i in range(len(atoms)):
            for j in range(i+1,len(atoms)):
                bond=self.mol.GetBondBetweenAtoms(i,j)
                if bond is None and self.inter_dismat[i][j]>4:
                    continue
                if bond is None and self.inter_dismat[i][j]<=4:
                    if PARAM.Cmpnn_setting.onehot:
                        self.bonds.append([i,j,bond_features(bond)+[self.inter_dismat[i][j]]])
                    else:
                        self.bonds.append([i,j,Feature_bond(bond)+[self.inter_dismat[i][j]]])
                else:
                    if PARAM.Cmpnn_setting.onehot:
                        self.bonds.append([i,j,bond_features(bond)+[self.inter_dismat[i][j]]])
                    else:
                        self.bonds.append([i,j,Feature_bond(bond)+[self.inter_dismat[i][j]]])

    def grep_subgraph(self,centeridx):
        R=self.inter_dismat[centeridx]
        ids=[id for id in np.argsort(R) if R[id] <= PARAM.Coordinate_descriptor_setting.Rc and self.atoms[id][0]!=1]
        ids_table={i:ids[i] for i in range(len(ids))}
        ids_rtable={ids[i]:i for i in range(len(ids))}
        #print (ids_table,ids_rtable)
        subR=[R[id] for id in ids]
        subatoms=[deepcopy(self.atoms[id]) for id in ids]
        subcoords=[deepcopy(self.coords[id]) for id in ids]
        subbonds=[[[] for j in range(len(ids))] for i in range(len(ids))]
        #print (self.bonds)
        for bond in self.bonds:
        
            if bond[0] in ids and bond[1] in ids:
                #print ('--------------------',bond[2:])
                #if PARAM.Cmpnn_setting.atom_messages:
                subbonds[ids_rtable[bond[0]]][ids_rtable[bond[1]]]=bond[2:][0]
                subbonds[ids_rtable[bond[1]]][ids_rtable[bond[0]]]=bond[2:][0]
                #else:
                #    subbonds[ids_rtable[bond[0]]][ids_rtable[bond[1]]]=self.atoms[bond[0]]+bond[2:][0]
                #    subbonds[ids_rtable[bond[1]]][ids_rtable[bond[0]]]=self.atoms[bond[1]]+bond[2:][0]
                    
        #print (subbonds) 
        #subbond=[bond for bond in self.bonds if bond[0] in ids and bond[1] in ids]
        if PARAM.Cmpnn_setting.if3dinfo:
            Sfactor=np.zeros(len(ids))
            for rid,dis in enumerate(subR):
                if rid==0:
                    Sfactor[rid]=1
                elif dis<=PARAM.Coordinate_descriptor_setting.Rcs and rid!=0:
                    Sfactor[rid]=1/dis
                elif PARAM.Coordinate_descriptor_setting.Rcs<=dis<=PARAM.Coordinate_descriptor_setting.Rc:
                    Sfactor[rid]=1/dis*(0.5*math.cos((dis-PARAM.Coordinate_descriptor_setting.Rcs)/(PARAM.Coordinate_descriptor_setting.Rc-PARAM.Coordinate_descriptor_setting.Rcs))+0.5)
            ex,ey,ez=create_coordaxis(subcoords[0],subcoords[1],subcoords[2])
            coord_descriptor=[]
            for i in range(len(ids)):
                vatom=subcoords[i]-subcoords[0]
                x,y,z=np.dot(vatom,ex),np.dot(vatom,ey),np.dot(vatom,ez)
                Ratom=subR[i]
                Satom=Sfactor[i]
                if i!=0:
                    subatoms[i]+=[Satom,x/Ratom*Satom,y/Ratom*Satom,z/Ratom*Satom,Satom*subatoms[i][0]]
                else:
                    subatoms[i]+=[Satom,0,0,0,Satom*subatoms[i][0]]

        suba2b=[]
        subb2a=[]
        subb2revb=[]
        sub_f_bond=[]
        sub_f_atom=[]
        sub_bond_pair=[]
        sub_b2revb=[]
        subnatom=len(ids)
        for i in range(len(ids)):
            sub_f_atom.append(subatoms[i])
            #print (subatoms[i])
        suba2b=[[] for i in range(subnatom)]
        subnbonds=0
        for a1 in range(len(ids)):
            for a2 in range(a1+1,len(ids)):
                if len(subbonds[a1][a2])>0:
                    sub_f_bond.append(subatoms[a1]+subbonds[a1][a2])
                    sub_f_bond.append(subatoms[a2]+subbonds[a2][a1])
                    b1=subnbonds
                    b2=subnbonds+1
                    suba2b[a2].append(b1)
                    subb2a.append(a1)
                    suba2b[a1].append(b2)
                    subb2a.append(a2)
                    subnbonds+=2
                    sub_bond_pair.append(np.array([a1,a2]))
                    sub_b2revb.append(b2)
                    sub_b2revb.append(b1)
        return {"id":self.id+'_atom_'+str(centeridx),"f_atom":sub_f_atom,"f_bond":sub_f_bond,"a2b":suba2b,"b2a":subb2a,"bond":sub_bond_pair,"b2revb":sub_b2revb,"natom":subnatom,"nbonds":subnbonds}
        
    def internal_distance_matrix(self):
        coords1=np.array(self.coords)
        coords2=np.array(self.coords)
        self.inter_dismat=spatial.distance_matrix(coords1,coords2)
        return 


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    A BatchMolGraph contains the attributes of a MolGraph plus:
    - id_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs):
        self.id_batch = [mol_graph.id for mol_graph in mol_graphs]
        self.n_mols = len(self.id_batch)

        self.atom_fdim = ATOM_FDIM + (PARAM.Cmpnn_setting.if3dinfo)*5
        self.bond_fdim = BOND_FDIM + (not PARAM.Cmpnn_setting.atom_messages) * self.atom_fdim

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0,0]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]]) #  if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], 
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
        
        bonds = np.array(bonds).transpose(1,0)
        
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self):
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask
        return self.b2b

    def get_a2a(self):
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a
    
