import os
import numpy as np
import re
import math

def xyz_to_pdb(dire,filename):

    coord_data=[]
    f=open(f'{dire}/plp_{filename}.xyz','r')
    for line in f.readlines():
        line=line.strip('\n')
        line=line.split()
        coord_data.append(line) 
    
    coordlist=[]        
    for coord in range(len(coord_data)):
        if len(coord_data[coord])==4:
            coordlist.append([])
            for xyz in range(1,4):
                coordlist[-1].append(str(round(float(coord_data[coord][xyz]),3)).rjust(8))
    f.close()
    
    pdblist=[]
    f=open('plp.pdb.1','r')
    m=0
    for line in f.readlines():
        if line[0:4]=='ATOM' and m<len(coordlist):
            sub=coordlist[m][0]+coordlist[m][1]+coordlist[m][2]
            m+=1
            linelist=list(line)
            n=0
            for word in range(30,54):
                linelist[word]=sub[n]
                n+=1
            pdblist.append(''.join(linelist))
        else:
            pdblist.append(line)
    f.close()

    with open(f'{dire}/plp_{filename}.pdb','a') as f_new:
        for pdb in pdblist:
            f_new.write(pdb)

for root, dirs, files in os.walk('./'):
    if dirs!=[] and re.search('Interpolation',dirs[-1])!=None:
        dir_1s=dirs
        print(dirs)
#for dir_1 in dir_1s:
#    if re.search('Interpolation',dir_1)!=None:
#        for j in range(20):
#            xyz_to_pdb(dir_1,j)

for i in range(1,42):
    for j in range(20):
        xyz_to_pdb(f'Interpolation_{i}',j)