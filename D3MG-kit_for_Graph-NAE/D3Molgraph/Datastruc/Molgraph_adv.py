from rdkit import Chem
#from pfdgm.base import *
from rdkit.Chem import ChemicalForceFields
import networkx as nx 
import spektral as sp
from spektral.data import Dataset 
import numpy as np
from scipy  import spatial 
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm 
from .periodictable import * 
from sklearn.cluster import DBSCAN 
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import math 
import random 
from ..Base.features import * 
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
from ..comparm import * 
import torch 
from rdkit.Chem.rdForceFieldHelpers import * 
class Molgraph:
    def __init__(self,**kwargs):
        try:
            sdfname=kwargs.get('sdfname',None)
            pdbname=kwargs.get('pdbname',None)
            molobj=kwargs.get('molobj',None)
            if sdfname:
                self.name=sdfname
                self.molobj=Chem.SDMolSupplier(sdfname)[0]
            if  pdbname:
                self.name=pdbname
                self.molobj=Chem.rdmolfiles.MolFromPDBFile(pdbname,removeHs=False)
            if molobj:
                print ('------------')
                name=kwargs.get('name','Mol')
                if name=='Mol':
                    self.name=Chem.rdmolfiles.MolToSmiles(molobj)
                    print (self.name)
                else:
                    self.name=name 
                self.molobj=molobj 
            
            self.atoms=[atom.GetAtomicNum() for atom in self.molobj.GetAtoms()]
            self.properties={}
            self.properties["QED"]=QED.qed(self.molobj)
            ff=AllChem.UFFGetMoleculeForceField(self.molobj)
            self.properties['Energy']=ff.CalcEnergy()
            self.n_atoms=len(self.atoms)
            
            self.d3coord=self.molobj.GetConformer(0).GetPositions()
            self.coordinate_feature_on_2D_graph=None
            self.connection=None
            self.connection_weight=None
            self.G=None
            self.d3dismat=None
            self.d2dismat=None
            self.coordinate_feature_order_map_on_2D_graph=None
            self.coordinate_feature_S_list_on_2D_graph=None
            self.zbond_connectivity=None
            self.zangle_connectivity=None
            self.zdihedral_connectivity=None
            self.f_atoms=[]
            self.f_bonds=[]
            self.a2b=[[] for i in range(self.n_atoms)]
            self.b2a=[]
            self.b2revb=[]
            self.bonds=[]
            self.bondlengths=[]
            self.n_bonds=0
            self.fix_size=kwargs.get('fix_size',True)
            self.generate_2D_graph()
            self.Rc=kwargs.get("Rc",6.5)
            self.Rs=kwargs.get("Rs",5)
            self.generate_graph_node_coordinate_feature_on_2D_graph(Rc=self.Rc,Rcs=self.Rs)
            self.build_Zmatrix_on_2D_graph()
            self.EGCM_and_Rank_on_2D_graph()
            return
        except Exception as e:
            print (f'{self.name} failed due to {e}')
            return None

    def generate_2D_graph(self):
        self.f_atoms=[]
        self.f_bonds=[]
        for i,atom in enumerate(self.molobj.GetAtoms()):
            self.f_atoms.append(atom_features(atom))
        #mp=ChemicalForceFields.UFFGetMoleculeProperties(self.molobj)
        self.connection=[]
        self.connection_weight=[]
        
        for i in range(self.n_atoms):
            for j in range(i+1,self.n_atoms):
                bond=self.molobj.GetBondBetweenAtoms(i,j)
                if bond:
                    f_bond=bond_features(bond)
                    if not GP.NNsetting.atom_messages:
                        self.f_bonds.append(self.f_atoms[i]+f_bond)
                        self.f_bonds.append(self.f_atoms[j]+f_bond)
                    else:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    #standard_bondlength=mp.GetUFFBondStretchParams(self.molobj,i,j)[-1]
                    standard_bondlength=ChemicalForceFields.GetUFFBondStretchParams(self.molobj,i,j)[-1]
                    b1=self.n_bonds
                    b2=b1+1
                    self.a2b[j].append(b1)
                    self.b2a.append(i)
                    self.a2b[i].append(b2)
                    self.b2a.append(j)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds+=2
                    self.bonds.append(np.array([i,j]))
                    self.bondlengths.append((i,j,standard_bondlength))
        G=nx.Graph()
        G.add_edges_from(self.bonds)
        G.add_weighted_edges_from(self.bondlengths)
        connection_length=dict(nx.all_pairs_dijkstra_path_length(G))
        self.d2dismat=np.zeros((self.n_atoms,self.n_atoms))
        for key1 in connection_length.keys():
            for key2 in connection_length[key1].keys():
                self.d2dismat[key1][key2]=connection_length[key1][key2]
        return
    
    def Fix_size(self,param):
        param={1:20,6:15,7:10,8:5,16:2,15:2}
        param_pt={}
        total_length=np.sum([param[key] for key in param.keys()]) 
        fix_connect_table=np.zeros((total_length,total_length))
        for i in range(total_length):
            fix_connect_table[i][i]=1
        keys=np.sort([key for key in param.keys()])
        tnum=0
        for i in range(len(keys)):
            param_pt[keys[i]]=tnum
            tnum+=param_pt[keys[i]]
        numdict={key:0 for key in keys}
        firstchange=np.zeros(len(self.atoms))
        for i in range(self.atoms):
            for key in keys:
                if self.atoms[i]==key:
                    firstchange[i]=numdict[key]+param_pt[key]
                    numdict[key]+=1
        for bond in self.bonds:
            fix_connect_table[firstchange[bond[0]],firstchange[bond[1]]]=1
            fix_connect_table[firstchange[bond[0]],firstchange[bond[1]]]=1
        EGCM,EVEC=np.linalg.eig(fix_connect_table)
        secondchange=np.argsort(EGCM)
        typeori=[]
        for i in range(len(keys)):
            typeori+=[key for key in range(param[key])]
        sortlist=[]
        for key in keys:
            for i in range(len(secondchange)):
                if param_pt[key]<=secondchange[i]<param_pt[key]+param[key]:
                    sortlist.append(secondchange[i])
        print (sortlist)
        origin2s=np.zeros(self.n_atoms)
        for i in range(total_length):
            for j in range(len(firstchange)):
                if firstchange[j]==sortlist[i]:
                    origin2s[j]=i
        # origin2s中保存的是从原始的原子顺序到加入虚原子后的标准顺序的变换关系，origin2s中的第i个原子对应着标准中的origin2s[i]号原子，可以根据这个顺序对GIE-RCM排序。
            
    def build_Zmatrix_on_2D_graph(self):
        self.zbond_connectivity=np.zeros(self.n_atoms,dtype=int)
        self.zangle_connectivity=np.zeros(self.n_atoms,dtype=int)
        self.zdihedral_connectivity=np.zeros(self.n_atoms,dtype=int)
        self.zbond=np.zeros(self.n_atoms)
        self.zangle=np.zeros(self.n_atoms)
        self.zdihedral=np.zeros(self.n_atoms)
        for atom in range(1,self.n_atoms):
            d2disvec=self.d2dismat[atom][:atom]
            distmin=np.array(d2disvec[np.nonzero(d2disvec)]).min()
            nearestindices=np.where(d2disvec==distmin)[0]
            nearestatom=nearestindices[0]
            self.zbond_connectivity[atom]=nearestatom 
            self.zbond[atom]=norm(self.d3coord[atom]-self.d3coord[nearestatom])
            if atom>=2:
                #self.zbond_connectivity[atom]=nearestatom
                atms=[0,0,0]
                atms[0]=atom
                atms[1]=self.zbond_connectivity[atms[0]]
                atms[2]=self.zbond_connectivity[atms[1]]
                if atms[2]==atms[1]:
                    for idx in range(1,len(self.zbond_connectivity[:atom])):
                        if self.zbond_connectivity[idx] in atms and not idx in atms:
                            atms[2]=idx
                            break 
                self.zangle_connectivity[atom]=atms[2]
                self.zangle[atom]=self.calc_angle(atms[0],atms[1],atms[2])
            if atom>=3:
                atms=[0,0,0,0]
                atms[0]=atom
                atms[1]=self.zbond_connectivity[atms[0]]
                atms[2]=self.zangle_connectivity[atms[0]]
                atms[3]=self.zangle_connectivity[atms[1]]
                if atms[3] in atms[:3]:
                    for idx in range(1,len(self.zbond_connectivity[:atom])):
                        if self.zbond_connectivity[idx] in atms and not idx in atms:
                            atms[3]=idx
                            break 
                self.zdihedral[atom]=self.calc_dihedral(atms[0],atms[1],atms[2],atms[3])
                if math.isnan(self.zdihedral[atom]):
                    self.zdihedral[atom]=0.0
                self.zdihedral_connectivity[atom]=atms[3]

    def generate_gzmatstr(self):
        gview_zcoord_str=''
        for i in range(self.n_atoms):
            if i >=3:
                gview_zcoord_str+=f'{Element[self.atoms[i]]} {self.zbond_connectivity[i]+1 :5d} {self.zbond[i] :0.4f} {self.zangle_connectivity[i]+1 :5d} {self.zangle[i] :0.4f} {self.zdihedral_connectivity[i]+1 :5d} {self.zdihedral[i] :0.4f}\n'
            elif i==2:
                gview_zcoord_str+=f'{Element[self.atoms[i]]} {self.zbond_connectivity[i]+1 :5d} {self.zbond[i] :0.4f} {self.zangle_connectivity[i]+1 :5d} {self.zangle[i] :0.4f} \n'
            elif i==1:
                gview_zcoord_str+=f'{Element[self.atoms[i]]} {self.zbond_connectivity[i]+1 :5d} {self.zbond[i] :0.4f} \n'
            elif i==0:
                gview_zcoord_str+=f'{Element[self.atoms[i]]} \n'
        return gview_zcoord_str 

    def build_d3coord_on_2D_graph(self,zb_cnt=None,za_cnt=None,zd_cnt=None,zb=None,za=None,zd=None):
        if zb_cnt is None :
            zb_cnt=self.zbond_connectivity
        if zb is None:
            zb=self.zbond
        if za_cnt is None :
            za_cnt=self.zangle_connectivity
        if za is None :
            za=self.zangle
        if zd is None:
            zd=self.zdihedral
        if zd_cnt is None:
            zd_cnt=self.zdihedral_connectivity 
        self.newd3coords=np.zeros((self.n_atoms,3))
        for i in range(self.n_atoms):
            self.newd3coords[i]=self.calc_positions(i,zb_cnt,za_cnt,zd_cnt,zb,za,zd)

        return self.newd3coords

    def calc_positions(self,i,zb_cnt,za_cnt,zd_cnt,zb,za,zd):
        """Calculate position of another atom based on internal coordinates"""
        position=[0,0,0]
        if i > 1:
            j = zb_cnt[i]
            k = za_cnt[i]
            l = zd_cnt[i]

            # Prevent doubles
            if k == l and i > 0:
                for idx in range(1, len(zb_cnt[:i])):
                    if zb_cnt[idx] in [i, j, k] and not idx in [i, j, k]:
                        l = idx
                        break

            avec = self.newd3coords[j]
            bvec = self.newd3coords[k]

            dst = zb[i]
            ang = za[i] * math.pi / 180.0

            if i == 2:
                # Third atom will be in same plane as first two
                tor = 90.0 * math.pi / 180.0
                cvec = np.array([0, 1, 0])
            else:
                # Fourth + atoms require dihedral (torsional) angle
                tor = zd[i] * math.pi / 180.0
                cvec = self.newd3coords[l]
            v1 = avec - bvec
            v2 = avec - cvec
            n = np.cross(v1, v2)
            nn = np.cross(v1, n)
            n /= norm(n)
            nn /= norm(nn)

            n *= -math.sin(tor)
            nn *= math.cos(tor)

            v3 = n + nn
            v3 /= norm(v3)
            v3 *= dst * math.sin(ang)
            v1 /= norm(v1)
            v1 *= dst * math.cos(ang)
            position = avec + v3 - v1
        elif i == 1:
            # Second atom dst away from origin along Z-axis
            j = zb_cnt[i]
            dst = zb[i]
            position = np.array([self.newd3coords[j][0] + dst, self.newd3coords[j][1], self.newd3coords[j][2]])
        elif i == 0:
            # First atom at the origin
            position = np.array([0, 0, 0])
        return position        

    def generate_3D_dismat(self):
        self.d3dismat=spatial.distance_matrix(self.d3coord,self.d3coord)

    def d3_disance(self,idx,idj):
        return np.sqrt(np.sum((self.d3coord[idx]-self.d3coord[idj])**2))

    def EGCM_and_Rank_on_2D_graph(self):
        self.Coulomb_mat_on_2D_graph=np.zeros((self.n_atoms,self.n_atoms))
        for i in range(self.n_atoms):
            for j in range(i,self.n_atoms):
                if i==j:
                    self.Coulomb_mat_on_2D_graph[i][j]=0.5*self.atoms[i]**2.4
                else:
                    self.Coulomb_mat_on_2D_graph[i][j]=self.atoms[i]*self.atoms[j]/self.d2dismat[i][j]
        self.EGCM_on_2D_graph,EVEC=np.linalg.eig(self.Coulomb_mat_on_2D_graph)
        self.Rank_on_2D_graph=np.argsort(self.EGCM_on_2D_graph)
        self.Rankmap_on_2D_graph={i:x for i,x in enumerate(self.Rank_on_2D_graph)}
        return 

    def EGCM_and_Rank_on_3D_graph(self):
        self.Coulomb_mat_on_3D_space=np.zeros((self.n_atoms,self.n_atoms))
        for i in range(self.n_atoms):
            for j in range(i,self.n_atoms):
                if i==j:
                    self.Coulomb_mat_on_3D_space[i][j]=0.5*self.atoms[i]**2.4
                else:
                    self.Coulomb_mat_on_3D_space[i][j]=self.atoms[i]*self.atoms[j]/self.d3dismat[i][j]
        self.EGCM_on_3D_space,EVEC=np.linalg.eig(self.Coulomb_mat_on_3D_space)
        self.Rank_on_3D_space=np.argsort(self.EGCM_on_3D_space)
        self.Rankmap_on_3D_space={i:x for i,x in enumerate(self.Rank_on_3D_space)}        
        return 
        
    def D3coordinate_feature_on_2D_graph(self,idx,Rc,Rcs):
        R=self.d2dismat[idx]
        S=np.zeros(len(R))
        for rid,dis in enumerate(R):
            if rid==idx:
                S[rid]=1000
            elif dis<=Rcs and rid!=idx:
                S[rid]=1/dis
            elif Rcs<dis<=Rc:
                S[rid]=1/dis*(0.5*np.cos((dis-Rcs)/(Rc-Rcs)*math.pi)+0.5)
        order=[i for i in np.argsort(R) if R[i] <=Rc and i!=idx]
        s1=order[0]
        s2=order[1]
        ex,ey,ez=cal_base_vector(self.d3coord[idx],self.d3coord[s1],self.d3coord[s2])
        coord_feature=[]
        slist=[]
        for aid in order:
            vid=self.d3coord[aid]-self.d3coord[idx]
            slist.append(S[aid])
            x,y,z=np.dot(vid,ex),np.dot(vid,ey),np.dot(vid,ez)
            coord_feature+=[S[aid]*x/R[aid],S[aid]*y/R[aid],S[aid]*z/R[aid]]

        return order,slist,coord_feature

    def D3coordinate_feature_on_3D_space(self,idx,Rc,Rcs):
        # order: the original node index in graph 
        R=self.d3dismat[idx]
        S=np.zeros(len(R))
        for rid,dis in enumerate(R):
            if rid==idx:
                S[rid]=1000
            elif dis<=Rcs and rid!=idx:
                S[rid]=1/dis
            elif Rcs<dis<=Rc:
                S[rid]=1/dis*(0.5*np.cos((dis-Rcs)/(Rc-Rcs)*math.pi)+0.5)
        order=[i for i in np.argsort(R) if R[i] <=Rc and i!=idx]
        slist=[]
        s1=order[0]
        s2=order[1]
        ex,ey,ez=cal_base_vector(self.d3coord[idx],self.d3coord[s1],self.d3coord[s2])
        coord_feature=[]
        for aid in order:
            vid=self.d3coord[aid]-self.d3coord[idx]
            x,y,z=np.dot(vid,ex),np.dot(vid,ey),np.dot(vid,ez)
            slist.append(S[aid])
            coord_feature+=[S[aid],S[aid]*x/R[aid],S[aid]*y/R[aid],S[aid]*z/R[aid],self.atoms[aid]*S[aid]]
        return order,slist,coord_feature

    def generate_graph_node_coordinate_feature_on_2D_graph(self,Rc,Rcs):
        self.coordinate_feature_on_2D_graph=[]
        self.coordinate_feature_order_map_on_2D_graph=[]
        self.coordinate_feature_S_list_on_2D_graph=[]
        for i in range(self.n_atoms):
            order,slist,coord_feature=self.D3coordinate_feature_on_2D_graph(i,Rc,Rcs)
            self.coordinate_feature_on_2D_graph.append(coord_feature)
            self.coordinate_feature_order_map_on_2D_graph.append(order)
            self.coordinate_feature_S_list_on_2D_graph.append(slist)
        maxlen=np.max([len(alist) for alist in self.coordinate_feature_on_2D_graph])
        print (maxlen)
        self.coordinate_feature_on_2D_graph=np.array([np.pad(x,(0,maxlen-len(x)),'constant',constant_values=0) for x in self.coordinate_feature_on_2D_graph])
        return 

    def order_graph_node_coordinate_feature_on_2D_graph(self):
        return [self.coordinate_feature_on_2D_graph[i] for i in self.Rank_on_2D_graph]
        
    def decode_coordinate_feature_on_2D_graph(self,idx,coord_feature):
        reverse_coordinate=np.zeros((self.n_atoms,3))
        tmpfeature=np.reshape(coord_feature,(-1,3))
        loss=0
        for i,tmpcrd in enumerate(tmpfeature):
            if i <len(self.coordinate_feature_order_map_on_2D_graph[idx]):
                tmpcoord=[]
                for j in range(3):                    
                    tmpcoord.append(tmpcrd[j]*self.d2dismat[idx][self.coordinate_feature_order_map_on_2D_graph[idx][i]]/self.coordinate_feature_S_list_on_2D_graph[idx][i])
                reverse_coordinate[self.coordinate_feature_order_map_on_2D_graph[idx][i]]=np.array(tmpcoord)
                for connection in self.connection_weight:
                    if connection[0] in self.coordinate_feature_order_map_on_2D_graph[idx] and \
                       connection[1] in self.coordinate_feature_order_map_on_2D_graph[idx]:
                        dis=norm(reverse_coordinate[connection[0]]-reverse_coordinate[connection[1]])
                        error=np.sqrt((dis-connection[2])**2)
                        if error>connection[2]*0.2:
                            loss+=error**2
        
        for i in range(1,self.n_atoms):
            if i in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx]:
                if self.zbond_connectivity[i] in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx] and i>=1:
                    self.decode_zbond[i].append(norm(reverse_coordinate[i]-reverse_coordinate[self.zbond_connectivity[i]]))
                    if self.zangle_connectivity[i] in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx] and i>=2:
                        self.decode_zangle[i].append(self.calc_angle(i,self.zbond_connectivity[i],self.zangle_connectivity[i],reverse_coordinate))
                        if self.zdihedral_connectivity[i] in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx] and i>=3:
                            self.decode_zdihedral[i].append(self.calc_dihedral(i,self.zbond_connectivity[i],self.zangle_connectivity[i],self.zdihedral_connectivity[i],reverse_coordinate))
        return reverse_coordinate,loss 

    def decode_total_coordinate_feature_on_2D_graph(self,coord_features,with_rank=False,samplenum=1,temperature=0):
        coordlist=[];losslist=[]
        self.decode_zbond=[[] for i in range(self.n_atoms)]
        self.decode_zangle=[[] for i in range(self.n_atoms)]
        self.decode_zdihedral=[[] for i in range(self.n_atoms)]
        self.decode_zbond_distribution=[[] for i in range(self.n_atoms)]
        self.decode_zangle_distribution=[[] for i in range(self.n_atoms)]
        self.decode_zdihedral_distribution=[[] for i in range(self.n_atoms)]
        for i in range(self.n_atoms):
            #print ('------------------')
            if with_rank:
                idx=self.Rank_on_2D_graph[i]
                reverse_coordinate,loss=self.decode_coordinate_feature_on_2D_graph(idx,coord_features[i])
                coordlist.append(reverse_coordinate)
                losslist.append(loss)
            else:
                idx=i
                reverse_coordinate,loss=self.decode_coordinate_feature_on_2D_graph(idx,coord_features[i])
                coordlist.append(reverse_coordinate)
                losslist.append(loss)
            #print (i,self.decode_zbond[i],self.decode_zangle[i],self.decode_zdihedral[i])
        for i in range(self.n_atoms):
            #print ('-------',self.zangle[i])
            self.decode_zbond_distribution[i]=denoise_distribution(self.decode_zbond[i],eps=0.1)
            self.decode_zangle_distribution[i]=denoise_distribution(self.decode_zangle[i],eps=20,period=360)
            self.decode_zdihedral_distribution[i]=denoise_distribution(self.decode_zdihedral[i],eps=20,period=360)

        if temperature==0:
            sample_zb=np.zeros(self.n_atoms,dtype=float)
            sample_za=np.zeros(self.n_atoms,dtype=float)
            sample_zd=np.zeros(self.n_atoms,dtype=float)
            for i in range(self.n_atoms):
                if i>=1:
                    maxpindex=np.argmax([c[2] for c in self.decode_zbond_distribution[i]])
                    sample_zb[i]=self.decode_zbond_distribution[i][maxpindex][0]
                if i>=2:
                    maxpindex=np.argmax([c[2] for c in self.decode_zangle_distribution[i]])
                    sample_za[i]=self.decode_zangle_distribution[i][maxpindex][0]
                if i>=3:
                    maxpindex=np.argmax([c[2] for c in self.decode_zdihedral_distribution[i]])
                    sample_zd[i]=self.decode_zdihedral_distribution[i][maxpindex][0]
            newcoord=self.build_d3coord_on_2D_graph(zb=sample_zb,za=sample_za,zd=sample_zd)
            return newcoord

        if temperature>0:
            sample_coordlist=[]
            sample_zb=np.zeros((samplenum,self.n_atoms),dtype=float)
            sample_za=np.zeros((samplenum,self.n_atoms),dtype=float)
            sample_zd=np.zeros((samplenum,self.n_atoms),dtype=float)
            for num in range(samplenum):
                for i in range(self.n_atoms):
                    if i>=1:
                        sample_zb[:,i]=sample_from_multigaussian(samplenum,parameters=self.decode_zbond_distribution[i],ranges=(0,3))
                    if i>=2:
                        sample_za[:,i]=sample_from_multigaussian(samplenum,parameters=self.decode_zangle_distribution[i],ranges=(0,180))
                    if i>=3:
                        sample_zd[:,i]=sample_from_multigaussian(samplenum,parameters=self.decode_zdihedral_distribution[i],ranges=(0,360))
                newcoord=self.build_d3coord_on_2D_graph(zb=sample_zb[num],za=sample_za[num],zd=sample_zd[num])
                sample_coordlist.append(newcoord)
        #return  coordlist,losslist
            return sample_coordlist 


    def clean(self):
        self.molobj=None

    def decode_final_feature_on_2D_graph(self):
        pass 

    def calc_angle(self, atom1, atom2, atom3, d3coord=None):
        """Calculate angle between 3 atoms"""
        if d3coord is not None:
            coords=d3coord 
        else:
            coords = self.d3coord
        vec1 = coords[atom2] - coords[atom1]
        uvec1 = vec1 / norm(vec1)
        vec2 = coords[atom2] - coords[atom3]
        uvec2 = vec2 / norm(vec2)
        return np.arccos(np.dot(uvec1, uvec2))*(180.0/math.pi)
    
    def calc_dihedral(self,atom1,atom2,atom3,atom4,d3coord=None):
        """
           Calculate dihedral angle between 4 atoms
           For more information, see:
               http://math.stackexchange.com/a/47084
        """
        if d3coord is not None :
            coords=d3coord 
        else:
            coords=self.d3coord
        # Vectors between 4 atoms
        b1 = coords[atom2] - coords[atom1]
        b2 = coords[atom2] - coords[atom3]
        b3 = coords[atom4] - coords[atom3]

        # Normal vector of plane containing b1,b2
        n1 = np.cross(b1, b2)
        un1 = n1 / norm(n1)

        # Normal vector of plane containing b1,b2
        n2 = np.cross(b2, b3)
        un2 = n2 / norm(n2)

        # un1, ub2, and m1 form orthonormal frame
        ub2 = b2 / norm(b2)
        um1 = np.cross(un1, ub2)

        # dot(ub2, n2) is always zero
        x = np.dot(un1, un2)
        y = np.dot(um1, un2)

        dihedral = np.arctan2(y, x)*(180.0/math.pi)
        if dihedral < 0:
            dihedral = 360.0 + dihedral
        return dihedral

class MGSet():
    def __init__(self,name='MoleculeGraphs',mollist=[]):
        self.molgraphs=[]
        for mol in mollist:
            #print(type(mol.coordinate_feature_on_2D_graph),mol.n_atoms,np.sum(mol.atoms))
            if type(mol.coordinate_feature_on_2D_graph)!=type(None) and mol.n_atoms>5 and np.sum(mol.atoms)<1000:
                flag=True 
                for aindex in mol.atoms:
                    if aindex not in [1,6,7,8,9,15,16,17]:
                        flag=False
                        break
                #print(flag)    
                if flag and not(np.isnan(mol.coordinate_feature_on_2D_graph).any()):
                    self.molgraphs.append(mol)
        print (len(self.molgraphs))
        self.trainingsets=[]
        self.validsets=[]
        self.testsets=[]
        self.name=name
        self.scaler=None 

    def prepare(self,ifscale=False):
        self.molnum=len(self.molgraphs)
        xdata=[]
        for i in range(self.molnum):
            xdata+=list(self.molgraphs[i].coordinate_feature_on_2D_graph)
        self.node_feature_max_length=np.max([len(feature) for feature in xdata])
        for i in range(self.molnum):
            self.molgraphs[i].coordinate_feature_on_2D_graph=[np.pad(x,(0,self.node_feature_max_length-len(x)),'constant',constant_values=0) for x in self.molgraphs[i].coordinate_feature_on_2D_graph]
        if ifscale:
            self.scaler=StandardScaler()
            self.scaler.fit(xdata)
            for i in range(self.molnum):
                self.molgraphs[i].coordinate_feature_on_2D_graph=self.scaler.transform(self.molgraphs[i].coordinate_feature_on_2D_graph)
        return

    def split_random(self,rate=(0.8,0.9,1)):
        random.shuffle(self.molgraphs)
        self.cutnum=[0]+[math.ceil(self.molnum*rate[i]) for i in range(3)]
        self.trbatchpt=0
        self.vbatchpt=self.cutnum[1]
        self.tebatchpt=self.cutnum[2]
        return 

    def get_BatchMGs(self,mode=''):
        if mode=='':
            if GP.NNsetting.batchsize> self.molnum:
                batchsize=self.molnum 
            else:
                batchsize=GP.NNsetting.batchsize 
            MGs=self.molgraphs[self.trbatchpt:self.trbatchpt+batchsize]
            BatchMGs=BatchMolGraph(MGs)
            self.trbatchpt+=batchsize
            if self.trbatchpt>=self.molnum:
                self.trbatchpt=0 
        elif mode=='train':
            if GP.NNsetting.batchsize> self.cutnum[1]:
                batchsize=self.cutnum[1]
            else:
                batchsize=GP.NNsetting.batchsize 
            MGs=self.molgraphs[self.trbatchpt:self.trbatchpt+batchsize]
            BatchMGs=BatchMolGraph(MGs)
            self.trbatchpt+=batchsize
            if self.trbatchpt>=self.cutnum[1]:
                self.trbatchpt=0
        elif mode=='validation':
            if GP.NNsetting.batchsize>self.cutnum[2]-self.cutnum[1]:
                batchsize=self.cutnum[2]-self.cutnum[1]
            else:
                batchsize=GP.NNsetting.batchsize
            MGs=self.molgraphs[self.vbatchpt:self.vbatchpt+batchsize]
            BatchMGs=BatchMolGraph(MGs)
            self.vbatchpt+=batchsize
            if self.vbatchpt>=self.cutnum[2] :
                self.vbatchpt=self.cutnum[1]
        elif mode=='test':
            if GP.NNsetting.batchsize>self.cutnum[3]-self.cutnum[2]:
                batchsize=self.cutnum[3]-self.cutnum[2]
            else:
                batchsize=GP.NNsetting.batchsize
            MGs=self.molgraphs[self.tebatchpt:self.tebatchpt+batchsize]
            BatchMGs=BatchMolGraph(MGs)
            self.tebatchpt+=batchsize
            if self.tebatchpt>=self.cutnum[3]:
                self.tebatchpt=self.cutnum[2]
        return BatchMGs

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
    def __init__(self, MGs):
        #self.id_batch = [mol_graph.id for mol_graph in mol_graphs]
        self.molgraphs=MGs
        self.n_mols = len(MGs)
        self.atom_fdim = ATOM_FDIM 
        self.bond_fdim = BOND_FDIM + (not GP.NNsetting.atom_messages) * self.atom_fdim
        self.GIERCM_dim=GP.NNsetting.GIERCM_dim
        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        
        f_GIERCMs= [[0] * self.GIERCM_dim] 
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0,0]]
        f_energies=[]
        for mol_graph in MGs:
            f_energies.append([mol_graph.properties['Energy']])
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            f_GIERCMs.extend(mol_graph.coordinate_feature_on_2D_graph)
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
        self.f_energies = torch.FloatTensor(f_energies)
        #print (f_GIERCMs,len(f_GIERCMs))
        self.f_GIERCMs=torch.FloatTensor(f_GIERCMs)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        if GP.NNsetting.cuda:
            self.f_atoms.cuda()
            self.f_bonds.cuda()
            self.a2b.cuda()
            self.b2a.cuda()
            self.b2revb.cuda()
            self.f_GIERCMs.cuda()
            self.f_energies.cuda()

            
    def get_components(self):
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds, self.f_GIERCMs,self.f_energies 

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

def cal_base_vector(p0,p1,p2):
    v1=p1-p0
    v2=p2-p0
    l1=np.sqrt(np.sum(v1**2))
    ex=v1/l1
    v2pe1=np.dot(v2,ex)*ex
    v2=v2-v2pe1
    l2=np.sqrt(np.sum(v2**2))
    ey=v2/l2 
    v3=np.cross(v1,v2)
    l3=np.sqrt(np.sum(v3**2))
    ez=v3/l3
    return ex,ey,ez 
        
def distance(crd1,crd2):
    return np.sqrt(np.sum((crd1-crd2)**2))

def cal_torsion_angle(coordinates):
    """
    Calculate single torsion angle
    :param coordinates: Multidimensional array of coordinates
    :return: Torsion angle
    """
    """
    a, b, c, d are the points that make up a torsion angle.
    """
    a = coordinates[0]
    b = coordinates[1]
    c = coordinates[2]
    d = coordinates[3]
    vector_1, vector_2, vector_3 = b - a, c - b, d - c
    norm_vector_1 = np.cross(vector_1, vector_2)   # a,b,c所在平面法向量
    norm_vector_2 = np.cross(vector_2, vector_3)   # b,c,d所在平面法向量
    norm_vector_1x2 = np.cross(norm_vector_1, norm_vector_2)
    x = np.dot(norm_vector_1, norm_vector_2)
    y = np.dot(norm_vector_1x2, vector_2) / np.linalg.norm(vector_2)
    radian = np.arctan2(y, x)
    angle = radian * 180 / math.pi
    return angle

def write_xyz(filename,atoms,coords,rms):
    with open(filename,'w') as f:
        f.write(f'{len(atoms)}\n')
        f.write(f'{rms}\n')
        for i in range(len(atoms)):
            f.write(f'{atoms[i]} {coords[i][0]} {coords[i][1]} {coords[i][2]}\n')

def denoise_distribution(alist,eps=0.1,period=0,maxtime=5):
    alist=np.array(alist).reshape((-1,1))
    distribution=[]
    counter=[]
    deps=eps
    if len(alist)>0:
        time=0
        while len(counter)==0 and time<maxtime:
            if period!=0:
                distA=pdist(alist,metric='euclidean')
                distB=squareform(distA)
                M=np.where(distB>0.5*period,period-distB,distB)
                dbscan_cluster=DBSCAN(eps=deps,metric='precomputed').fit_predict(M)
            else:
                dbscan_cluster=DBSCAN(eps=deps).fit_predict(alist)
            time+=1
            counter=[c for c in Counter(dbscan_cluster).most_common() if c[0]!=-1]
            deps=deps+0.5*eps
            if time==maxtime and len(counter)==0:
                counter=[c for c in Counter(dbscan_cluster).most_common()]
                #print (counter)
        for c in counter:
                if period==0:
                    indexs=np.where(dbscan_cluster==c[0])
                    vlist=alist[indexs]
                    stdv=np.std(vlist)
                    mean=np.mean(vlist)
                    distribution.append((mean,stdv,c[1]))
                else:
                    indexs=np.where(dbscan_cluster==c[0])[0]
                    vlist=alist[indexs]
                    minindex=np.argmin(vlist)
                    vM=np.reshape(np.where(distB>0.5*period,distB-period,distB)[indexs[minindex]][indexs],(-1,1))

                    stdv=np.std(vlist[minindex]+vM)
                    mean=np.mean(vlist[minindex]+vM)
                    if mean<0:
                        mean+=period
                    distribution.append((mean,stdv,c[1]))
    #print (alist,distribution,counter)
    return distribution

def sample_from_multigaussian(samplenum,parameters,ranges=(0,3)):
    intervals=np.linspace(ranges[0],ranges[1],samplenum*100)
    possibility=np.zeros(len(intervals))
    for para in parameters:
        if len(para)>0:
            mu=para[0]
            sigma=para[1]
            possibility+= 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (intervals - mu)**2 / (2 * sigma**2))
    possibility=possibility/np.sum(possibility)
    samples=np.random.choice(intervals,size=samplenum,p=possibility)
    return samples 
    
def cal_energy(mg,filename='g16'):
    xyz=mg.d3coord
    atoms=mg.atoms
    with open (filename+'.com','w') as f:
        f.write('%'+'nproc=20\n')
        f.write('%'+'mem=600MW\n')
        f.write('%'+'chk=1.chk\n')
        f.write('# PM6 nosymm\n')
        f.write('\n')
        f.write('Hello world!\n')
        f.write('\n')
        f.write('%d %d\n'%(charge,spin))
        for i in range(len(self.atoms)):
            f.write('%s %.3f %.3f %.3f\n'%(atoms[i],xyz[i][0],xyz[i][1],xyz[i][2]))
        f.write('\n')
        f.close()
    os.system(f'g16 {filename}.com')
    normalflag=True 
    with open(filename+'.log','r') as f:
        line=f.readline()
        while line:
            if 'SCF Done' in line:
                var=line.split()
                energy=float(var[4])
            if 'Predicted change' in line:
                DBLOCK=''
                while 'Normal termination' not in line:
                    if 'Error termination' in line:
                        print (f'{filename}.com is end with error')
                        normalflag=False
                    line=file.readline()
            line=f.readline()
    if not normalflag:
        return None 
    else:
        return energy

def G16_TS_search(mg1,mg2,filename,charge=1,spin=1):
    xyz1=mg1.d3coord
    xyz2=mg2.d3coord
    atoms1=mg1.atoms
    atoms2=mg2.atoms 
    with open (filename+'.com','w') as f:
        f.write('%'+'nproc=20\n')
        f.write('%'+'mem=600MW\n')
        f.write('%'+'chk=1.chk\n')
        f.write('# PM6 nosymm scf=qc geom=connectivity opt=QST2 freq \n')
        f.write('\n')
        f.write('Hello world!\n')
        f.write('\n')
        f.write('%d %d\n'%(charge,spin))
        for i in range(len(atoms1)):
            f.write('%s %.3f %.3f %.3f\n'%(atoms1[i],xyz1[i][0],xyz1[i][1],xyz1[i][2]))
        f.write('%d %d\n'%(charge,spin))
        for i in range(len(atoms2)):
            f.write('%s %.3f %.3f %.3f\n'%(atoms2[i],xyz2[i][0],xyz2[i][1],xyz2[i][2]))
        f.write('\n')
        f.close()
     

