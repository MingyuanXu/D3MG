import os
import re
import numpy as np
import math

coorddata=[]
pdbdata=[]
with open (f'plp.pdb.1','r') as f:
    for line in f.readlines():
        line=line.strip('\n')
        line=line.split()
        if line[0]=='ATOM' and line[-1]!='H':
            coorddata.append([float(line[5]),float(line[6]),float(line[7])])    
            pdbdata.append(line[2])   
            
f=open('plp.prmtop','r')
flag=False
datas=[[],[],[]]
k=-1
for line in f.readlines():
    if re.search('WITHOUT_HYDROGEN',line)!=None:
        flag=True
        k=k+1
    if re.search('INC_HYDROGEN',line)!=None:
        flag=False
    if re.search('EXCLUDED_ATOMS_LIST',line)!=None:
        break
    if flag and re.search('FORMAT',line)==None:
        line=line.strip('\n')
        line=line.split()
        datas[k].extend(line)
        
feature_list=['bond_list','angle_list','dihedral_list']
feature_dict={}
for feature in feature_list:
    num=0
    f_list=[]
    feature_type=feature_list.index(feature)
    p=feature_type+3
    for data in datas[feature_type]:
        if re.search('\d',data)!=None:
            if num%p==0:
                f_list.append([])
            if num%p!=p-1:
                f_list[int(num/p)].append(int(float(data)/3))
            num+=1
    feature_dict[feature]=f_list
    
    new_f_list=[]
    for atom_list in f_list:
        if all(map(lambda x: x >= 0 and x <= len(pdbdata), atom_list)):
            new_f_list.append(atom_list)
    feature_dict[feature]=new_f_list
 
distance_total=[]
angle_total=[]
dihedral_total=[]
n=0
for i in range(1,41):
    for j in range(20):
        coordlist=[]
        #pdbdata=[]
        with open (f'Interpolation_{i}/plp_{j}.pdb','r') as f:
            for line in f.readlines():
                line=line.strip('\n')
                line=line.split()
                if line[0]=='ATOM':# and line[-1]!='H':
                    coordlist.append([float(line[5]),float(line[6]),float(line[7])])    
                    #pdbdata.append(line[3]+line[4]+'_'+line[2])
        
        distance_total.append([])
        for a,b in feature_dict['bond_list']:
            distance=np.linalg.norm(np.array(coordlist[a])-np.array(coordlist[b]))
            distance_total[n].append(distance)
    
        angle_total.append([])
        for a,b,c in feature_dict['angle_list']:
            vec1=np.array(coordlist[b])-np.array(coordlist[a])
            uvec1=vec1/np.linalg.norm(vec1)
            vec2=np.array(coordlist[b])-np.array(coordlist[c])
            uvec2=vec2/np.linalg.norm(vec2)
            angle=np.arccos(np.dot(uvec1,uvec2))*(180.0/math.pi)
            angle_total[n].append(angle)
        
        dihedral_total.append([])
        for a,b,c,d in feature_dict['dihedral_list']:
            b1=np.array(coordlist[abs(b)])-np.array(coordlist[abs(a)])
            b2=np.array(coordlist[abs(b)])-np.array(coordlist[abs(c)])
            b3=np.array(coordlist[abs(d)])-np.array(coordlist[abs(c)])

            n1=np.cross(b1, b2) #b1,b2所在平面的法向量
            un1=n1 / np.linalg.norm(n1) #法向量方向的单位向量

            n2=np.cross(b2, b3) #b2,b3所在平面的法向量
            un2=n2 / np.linalg.norm(n2) #法向量方向的单位向量

            ub2=b2 / np.linalg.norm(b2) #两平面交线的单位向量
            um1=np.cross(un1,ub2) #两个相互垂直的单位向量叉乘得到一个垂直于它们的单位向量

            x=np.dot(un1,un2) 
            y=np.dot(um1,un2)
            '''当二面角为锐角时，x=两向量模乘积再乘cos二面角，y=两向量模乘积再乘cos二面角的余角'''
            '''当二面角为钝角时，x=两向量模乘积再乘负的cos二面角的补角，y=两向量模乘积再乘cos二面角补角的余角'''

            dihedral=np.arctan2(y,x)*(180.0/math.pi)
            if dihedral<0:
                dihedral=360.0+dihedral
            dihedral_total[n].append(dihedral)
        n=n+1

for k in range(len(np.array(distance_total).T)):
    output=str(np.array(distance_total).T[k]).lstrip('[').strip(']')
    f1=open(f'Distance_interpolation/'+str(pdbdata[feature_dict['bond_list'][k][0]])+'-'+str(pdbdata[feature_dict['bond_list'][k][1]])+'_interpolation.txt','a')
    f1.write(output)
    f1.close()
    
for k in range(len(np.array(angle_total).T)):
    output=str(np.array(angle_total).T[k]).lstrip('[').strip(']')
    f1=open(f'Angle_interpolation/'+str(pdbdata[feature_dict['angle_list'][k][0]])+'-'+str(pdbdata[feature_dict['angle_list'][k][1]])+'-'
               +str(pdbdata[feature_dict['angle_list'][k][2]])+'_interpolation.txt','a')
    f1.write(output)
    f1.close()
    
for k in range(len(np.array(dihedral_total).T)):
    output=str(np.array(dihedral_total).T[k]).lstrip('[').strip(']')
    f1=open(f'Dihedral_interpolation/'+str(pdbdata[feature_dict['dihedral_list'][k][0]])+'-'+str(pdbdata[feature_dict['dihedral_list'][k][1]])+'-'
               +str(pdbdata[feature_dict['dihedral_list'][k][2]])+'-'+str(pdbdata[feature_dict['dihedral_list'][k][3]])+'_interpolation.txt','a')
    f1.write(output)
    f1.close()