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
class Molgraph:
    def __init__(self,**kwargs):
        sdfname=kwargs.get('sdfname',None)
        pdbname=kwargs.get('pdbname','None')
        if sdfname:
            self.name=sdfname
            self.molobj=Chem.SDMolSupplier(sdfname)[0]
        if  pdbname:
            self.name=pdbname
            self.molobj=Chem.rdmolfiles.MolFromPDBFile(pdbname)
        self.atoms=[atom.GetAtomicNum() for atom in self.molobj.GetAtoms()]
        self.d3coord=self.molobj.GetConformer(0).GetPositions()
        self.natoms=len(self.atoms)
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
    
    def generate_2D_graph(self):
        mp=ChemicalForceFields.MMFFGetMoleculeProperties(self.molobj)
        self.connection=[]
        self.connection_weight=[]
        for i in range(self.natoms):
            for j in range(i+1,self.natoms):
                bond=self.molobj.GetBondBetweenAtoms(i,j)
                if bond:
                    standard_bondlength=mp.GetMMFFBondStretchParams(self.molobj,i,j)[-1]
                    self.connection.append((i,j))
                    self.connection_weight.append((i,j,standard_bondlength))
        self.G=nx.Graph()
        self.G.add_edges_from(self.connection)
        self.G.add_weighted_edges_from(self.connection_weight)
        connection_length=dict(nx.all_pairs_dijkstra_path_length(self.G))
        self.d2dismat=np.zeros((self.natoms,self.natoms))
        for key1 in connection_length.keys():
            for key2 in connection_length[key1].keys():
                self.d2dismat[key1][key2]=connection_length[key1][key2]
        return

    def build_Zmatrix_on_2D_graph(self):
        self.zbond_connectivity=np.zeros(self.natoms,dtype=int)
        self.zangle_connectivity=np.zeros(self.natoms,dtype=int)
        self.zdihedral_connectivity=np.zeros(self.natoms,dtype=int)
        self.zbond=np.zeros(self.natoms)
        self.zangle=np.zeros(self.natoms)
        self.zdihedral=np.zeros(self.natoms)
        for atom in range(1,self.natoms):
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
        for i in range(self.natoms):
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
        self.newd3coords=np.zeros((self.natoms,3))
        for i in range(self.natoms):

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
        self.Coulomb_mat_on_2D_graph=np.zeros((self.natoms,self.natoms))
        for i in range(self.natoms):
            for j in range(i,self.natoms):
                if i==j:
                    self.Coulomb_mat_on_2D_graph[i][j]=0.5*self.atoms[i]**2.4
                else:
                    self.Coulomb_mat_on_2D_graph[i][j]=self.atoms[i]*self.atoms[j]/self.d2dismat[i][j]
        self.EGCM_on_2D_graph,EVEC=np.linalg.eig(self.Coulomb_mat_on_2D_graph)
        self.Rank_on_2D_graph=np.argsort(self.EGCM_on_2D_graph)
        self.Rankmap_on_2D_graph={i:x for i,x in enumerate(self.Rank_on_2D_graph)}
        return 

    def EGCM_and_Rank_on_3D_graph(self):
        self.Coulomb_mat_on_3D_space=np.zeros((self.natoms,self.natoms))
        for i in range(self.natoms):
            for j in range(i,self.natoms):
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
        for i in range(self.natoms):
            order,slist,coord_feature=self.D3coordinate_feature_on_2D_graph(i,Rc,Rcs)
            self.coordinate_feature_on_2D_graph.append(coord_feature)
            self.coordinate_feature_order_map_on_2D_graph.append(order)
            self.coordinate_feature_S_list_on_2D_graph.append(slist)
        maxlen=np.max([len(alist) for alist in self.coordinate_feature_on_2D_graph])
        #print (maxlen)
        self.coordinate_feature_on_2D_graph=np.array([np.pad(x,(0,maxlen-len(x)),'constant',constant_values=0) for x in self.coordinate_feature_on_2D_graph])
        return 

    def order_graph_node_coordinate_feature_on_2D_graph(self):
        return [self.coordinate_feature_on_2D_graph[i] for i in self.Rank_on_2D_graph]
        
    def decode_coordinate_feature_on_2D_graph(self,idx,coord_feature):
        reverse_coordinate=np.zeros((self.natoms,3))
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
        
        for i in range(1,self.natoms):
            #print ('----',i,self.zbond_connectivity[i],self.zangle_connectivity[i],self.zdihedral_connectivity[i])
            #print (self.coordinate_feature_order_map_on_2D_graph[idx])
            #print (self.decode_zbond[i],self.decode_zangle[i],self.decode_zdihedral[i])
            if i in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx]:
                if self.zbond_connectivity[i] in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx] and i>=1:
                    self.decode_zbond[i].append(norm(reverse_coordinate[i]-reverse_coordinate[self.zbond_connectivity[i]]))
                    if self.zangle_connectivity[i] in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx] and i>=2:
                        self.decode_zangle[i].append(self.calc_angle(i,self.zbond_connectivity[i],self.zangle_connectivity[i],reverse_coordinate))
                        if self.zdihedral_connectivity[i] in [idx]+self.coordinate_feature_order_map_on_2D_graph[idx] and i>=3:
                            self.decode_zdihedral[i].append(self.calc_dihedral(i,self.zbond_connectivity[i],self.zangle_connectivity[i],self.zdihedral_connectivity[i],reverse_coordinate))
            #print (self.decode_zdihedral[4])
            #print ('++++',self.decode_zbond[i],self.decode_zangle[i],self.decode_zdihedral[i])
        #print (idx,self.decode_zdihedral)
        return reverse_coordinate,loss 

    def decode_total_coordinate_feature_on_2D_graph(self,coord_features,with_rank=False,samplenum=1,temperature=0):
        coordlist=[];losslist=[]
        self.decode_zbond=[[] for i in range(self.natoms)]
        self.decode_zangle=[[] for i in range(self.natoms)]
        self.decode_zdihedral=[[] for i in range(self.natoms)]
        self.decode_zbond_distribution=[[] for i in range(self.natoms)]
        self.decode_zangle_distribution=[[] for i in range(self.natoms)]
        self.decode_zdihedral_distribution=[[] for i in range(self.natoms)]
        for i in range(self.natoms):
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
        for i in range(self.natoms):
            #print ('-------',self.zangle[i])
            self.decode_zbond_distribution[i]=denoise_distribution(self.decode_zbond[i],eps=0.1)
            self.decode_zangle_distribution[i]=denoise_distribution(self.decode_zangle[i],eps=20,period=360)
            self.decode_zdihedral_distribution[i]=denoise_distribution(self.decode_zdihedral[i],eps=20,period=360)
        #print (self.decode_zangle_distribution)
        #print (self.decode_zangle_distribution)
        #print (self.decode_zdihedral_distribution)
        
        if temperature==0:
            sample_zb=np.zeros(self.natoms,dtype=float)
            sample_za=np.zeros(self.natoms,dtype=float)
            sample_zd=np.zeros(self.natoms,dtype=float)
            for i in range(self.natoms):
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
            sample_zb=np.zeros((samplenum,self.natoms),dtype=float)
            sample_za=np.zeros((samplenum,self.natoms),dtype=float)
            sample_zd=np.zeros((samplenum,self.natoms),dtype=float)
            for num in range(samplenum):
                for i in range(self.natoms):
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

class MGSet(Dataset):
    def __init__(self,name='MoleculeGraphs',mollist=[],**kwargs):
        self.molgraphs=mollist
        self.name=name
        self.scaler=None 
        super().__init__(**kwargs)

    def prepare(self,ifscale=True,fmaxlength=None):
        self.molnum=len(self.molgraphs)
        xdata=[]
        from tqdm import tqdm 
        for i in tqdm(range(self.molnum)):
                if type(self.molgraphs[i].coordinate_feature_on_2D_graph)==type(None):
                    self.molgraphs[i].generate_2D_graph()
                    self.molgraphs[i].generate_graph_node_coordinate_feature_on_2D_graph(Rc=12,Rcs=10)
                    self.molgraphs[i].build_Zmatrix_on_2D_graph() 
                xdata+=list(self.molgraphs[i].coordinate_feature_on_2D_graph)
            
            #self.molgraphs[i].clean()
        if not fmaxlength:
            self.node_feature_max_length=np.max([len(feature) for feature in xdata])
        else:
            self.node_feature_max_length=fmaxlength 
        for i in range(self.molnum):
            #print (self.molgraphs[i].coordinate_feature_on_2D_graph[0])
            self.molgraphs[i].coordinate_feature_on_2D_graph=[np.pad(x,(0,self.node_feature_max_length-len(x)),'constant',constant_values=0) for x in self.molgraphs[i].coordinate_feature_on_2D_graph]
            #print (self.molgraphs[i].coordinate_feature_on_2D_graph[0])
        if ifscale:
            self.scaler=StandardScaler()
            self.scaler.fit(xdata)
            for i in range(self.molnum):
                if i==0:
                    print (self.molgraphs[i].coordinate_feature_on_2D_graph[0])
                self.molgraphs[i].coordinate_feature_on_2D_graph=self.scaler.transform(self.molgraphs[i].coordinate_feature_on_2D_graph)
                if i==0:
                    print (self.molgraphs[i].coordinate_feature_on_2D_graph[0])
        return

    def split(self,rate):
        import copy 
        cutnum=math.ceil(self.molnum*rate)
        random.shuffle(self.molgraphs)
        trainingset=copy.deepcopy(self)
        trainingset.name=self.name+'_train'
        trainingset.molgraphs=trainingset.molgraphs[:cutnum]
        #MGSet(name=self.name+'_train',mollist=self.molgraphs[:cutnum])
        testset=copy.deepcopy(self)
        testset.name=self.name+'_test'
        testset.molgraphs=testset.molgraphs[cutnum:]
        #testset=MGSet(name=self.name+'_test',mollist=self.molgraphs[cutnum:])
        return trainingset,testset 

    def read(self):
        graphlist=[]
        for molgraph in self.molgraphs:
            graphlist.append(sp.data.Graph(x=np.array(molgraph.coordinate_feature_on_2D_graph),a=np.array(molgraph.connection),y=np.array(molgraph.coordinate_feature_on_2D_graph)))
        return graphlist
            
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
    
