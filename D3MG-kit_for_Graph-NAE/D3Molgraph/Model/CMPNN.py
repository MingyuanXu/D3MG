import torch.nn as nn
from typing import List, Union
import torch
import torch.nn as nn
import numpy as np
#from .nn_ttils import * 
import math
import torch.nn.functional as F
from ..comparm import * 
from ..Base import * 
from ..Datastruc import * 
import pickle 
import tempfile 
import os
import tensorflow as tf
import shutil
import zipfile
from copy import deepcopy
import time 
def index_select_ND(source, index) :
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """

    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target[index==0] = 0
    return target

class MPNEncoder(nn.Module):
    def __init__(self):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = GP.NNsetting.atom_fdim
        self.bond_fdim = GP.NNsetting.bond_fdim
        self.hidden_size = GP.NNsetting.hidden_size
        self.bias = GP.NNsetting.bias
        self.depth = GP.NNsetting.depth
        self.dropout = GP.NNsetting.dropout
        self.layers_per_message = 1
        self.undirected = GP.NNsetting.undirected
        self.atom_messages = GP.NNsetting.atom_messages
        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Activation
        self.act_func = get_activation_function(GP.NNsetting.activation)
        # Input
        input_dim = self.atom_fdim
        
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        if not GP.NNsetting.atom_messages:
            input_dim = self.bond_fdim+self.atom_fdim
        print (input_dim)
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
        w_h_input_size_bond = self.hidden_size
        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(
                (self.hidden_size)*2,
                self.hidden_size)        
        self.gru = BatchGRU()
        self.lr = nn.Linear(self.hidden_size*3, self.hidden_size, bias=self.bias)
        
    def forward(self,BatchMGs):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds,f_GIERCMs,f_energies = BatchMGs.get_components()
        if GP.NNsetting.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_GIERCM,f_energies = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(),f_GIERCMs.cuda(),f_energies.cuda())
        # Input
        
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
         
        message_atom = input_atom.clone()
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)

        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message
            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)
        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden        
        return atom_hiddens 

class GIERCM_encoder(nn.Module):
    def __init__(self):
        super(GIERCM_encoder,self).__init__()
        self.inputdim=GP.NNsetting.GIERCM_dim
        self.hidden_size=GP.NNsetting.hidden_size
        self.lantdim=GP.NNsetting.lantdim
        self.input=nn.Linear(self.inputdim,self.hidden_size)
        self.enc_layer=nn.Linear(self.hidden_size,self.hidden_size)
        self.enc_mean=nn.Linear(self.hidden_size,self.lantdim)
        self.enc_var=nn.Linear(self.hidden_size,self.lantdim)

    def _sample_z(self,mean,var):
        if GP.NNsetting.cuda:
            epsilon=torch.randn(mean.shape).cuda()
        else:
            epsilon=torch.randn(mean.shape)
        return mean+torch.sqrt(var)*epsilon
        #return mean

    def forward(self,BatchMGs):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds,f_GIERCMs,f_energies= BatchMGs.get_components()
        if GP.NNsetting.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_GIERCMs,f_energies = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(),f_GIERCMs.cuda(),f_energies.cuda())

        x=F.relu(self.input(f_GIERCMs))
        x=F.relu(self.enc_layer(x))
        mean=self.enc_mean(x)
        var=F.softplus(self.enc_var(x))
        z=self._sample_z(mean,var)

        return mean,var,z 

class GIERCM_decoder(nn.Module):
    def __init__(self):
        super(GIERCM_decoder,self).__init__()
        self.inputdim=GP.NNsetting.GIERCM_dim
        self.hidden_size=GP.NNsetting.hidden_size
        self.lantdim=GP.NNsetting.lantdim
        self.dec_input=nn.Linear(self.lantdim+self.hidden_size,self.hidden_size)
        self.dec_layer=nn.Linear(self.hidden_size,self.hidden_size)
        self.dec_out=nn.Linear(self.hidden_size,self.inputdim)

    def forward(self,z):
        x=F.relu(self.dec_input(z))
        x=F.relu(self.dec_layer(x))
        x=self.dec_out(x)
        return x 


class BatchGRU(nn.Module):
    def __init__(self):
        super(BatchGRU, self).__init__()
        self.hidden_size = GP.NNsetting.hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2,1,1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2*self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1), 
                             cur_message_unpadding], 0)
        return message
    
class Graph_DCGM(nn.Module):
    def __init__(self):
        super(Graph_DCGM, self).__init__()
        self.graph_encoder = MPNEncoder()
        self.GIERCM_encoder=GIERCM_encoder()
        self.GIERCM_decoder=GIERCM_decoder()
        first_linear_dim=GP.NNsetting.lantdim*2+GP.NNsetting.hidden_size
        dropout     = nn.Dropout(GP.NNsetting.dropout)
        output_size = 1
        activation=nn.ReLU()
        ffn=[dropout,nn.Linear(first_linear_dim,GP.NNsetting.Emodelstruc[0])]
        for i in range(len(GP.NNsetting.Emodelstruc)-2):
            ffn.extend([activation,dropout,nn.Linear(GP.NNsetting.Emodelstruc[i],GP.NNsetting.Emodelstruc[i+1])])
        ffn.extend([activation,dropout,nn.Linear(GP.NNsetting.Emodelstruc[-2],GP.NNsetting.Emodelstruc[-1])])
        ffn.extend([dropout,nn.Linear(GP.NNsetting.Emodelstruc[-1],1)])
        # Create FFN model
        self.Energy_predictor = nn.Sequential(*ffn)
    
    def forward(self, BatchMGs):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds,f_GIERCMs,f_energies = BatchMGs.get_components()
        if GP.NNsetting.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_GIERCM,f_energies = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(),f_GIERCMs.cuda(),f_energies.cuda())
        graph_atom_info=self.graph_encoder(BatchMGs)
        mean,var,z=self.GIERCM_encoder(BatchMGs)
        energy_descriptor=torch.cat([graph_atom_info,mean,var],dim=1)
        decode_z=torch.cat([graph_atom_info,z],dim=1)
        RCM_output=self.GIERCM_decoder(decode_z)
        atomic_energy=self.Energy_predictor(energy_descriptor)
        energies=[]
        GIERCM=[]
        for i, (a_start,a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            mol_energy = atomic_energy.narrow(0, a_start, a_size)
            energies.append(mol_energy.sum(0))
            GIERCM.append(RCM_output.narrow(0,a_start,a_size))
        energies=torch.stack(energies,dim=0)
        #GIERCM=torch.cat(GIERCM,dim=0)
        return RCM_output,GIERCM,energies,z,mean,var,graph_atom_info 

    def decode_graph_z(self,graph_atom_info,z,BatchMGs):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds,f_GIERCMs,f_energies = BatchMGs.get_components()
        if GP.NNsetting.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_GIERCM,f_energies = (
                    f_atoms.cuda(), f_bonds.cuda(),
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(),f_GIERCMs.cuda(),f_energies.cuda())
        GIERCM=[]
        decode_z=torch.cat([graph_atom_info,z],dim=1)
        RCM_output=self.GIERCM_decoder(decode_z)
        for i, (a_start,a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            GIERCM.append(RCM_output.narrow(0,a_start,a_size))
        return GIERCM

def compute_energy_loss(refe,prede):
    loss_e=torch.nn.MSELoss()
    return loss_e(refe,prede) 

def compute_reconstruct_loss(ref,pred):
    loss_reconstruct=torch.nn.MSELoss()
    return loss_reconstruct(ref,pred)

def compute_KL_loss(mean,var):
    KL_loss = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
    return KL_loss

import os 
from datetime import datetime  
class Graph_DCGM_Model():
    def __init__(self,**kwargs):
        if "modelname" not in kwargs:
            if 'dataset' in kwargs:
                self.mode="train"
                dataset=kwargs.get('dataset') 
                self.modelname=dataset.name
                self.training_history=[]
                self.Total_loss=0
                self.Total_loss_e=0
                self.Total_loss_KL=0
                self.Total_loss_reconstruct=0
                self.recordnum=0
                self.Valid_loss=0
                self.Valid_loss_e=0
                self.Valid_loss_KL=0
                self.Valid_loss_reconstruct=0
                self.vrecordnum=0
                self.min_val_loss=100000
                self.training_history=[]
                self.epochs=0
                self.model=None 
                self.optimizer=None 
                self.lr_scheduler=None 
                
            else:
                self.mode="retrain"
        elif "modelname" in kwargs:
            self.mode="test"     
        if self.mode=="train":
            if not os.path.exists(f'./{self.modelname}/model'):
                os.system(f'mkdir -p ./{self.modelname}/model')
                pickle.dump(self.__dict__,open(self.modelname+"/modelsetting.pickle", "wb"))
            self.__build_model()
            if GP.NNsetting.cuda:
                self.model.to('cuda')
            self.logger=open(f'./{self.modelname}/Training.log','a')
            now = datetime.now()
            self.logger.write('='*40+now.strftime("%d/%m/%Y %H:%M:%S")+'='*40+'\n') 
            self.logger.flush()
        else:
            self.modelname=kwargs.get("modelname")
            self.load(self.modelname)
        return 

    def __build_model(self):
        self.model=Graph_DCGM()
        
    def load(self, model_name):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(model_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            # Load metadata
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            self.__dict__.update(metadata)
            #try:
            if True:
                self.__build_model()
                modelcpkt=torch.load(dirpath+"/model/model.cpk")
                self.model.load_state_dict(modelcpkt["state_dict"])
                self.epochs=modelcpkt['epochs']
                self.training_history=modelcpkt['training_history']
                if GP.NNsetting.cuda:
                    self.model.to('cuda') 
            #except:
            #    print("'model' not found, setting to None.")
            #    self.model = None

    def save(self,modelname=''):
        self.model=None
        self.optimizer=None
        self.lr_scheduler=None
        self.__dict__['logger']=f'./{self.modelname}/Training.log'
        print(self.__dict__)
        pickle.dump(self.__dict__,open(self.modelname+"/modelsetting.pickle", "wb"))
        shutil.make_archive(self.modelname,"zip",self.modelname)
        return

    def training_step(self,BMGs):
        t1=time.time()
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds,f_GIERCM,f_energies= BMGs.get_components()
        if GP.NNsetting.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_GIERCM,f_energies = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(),f_GIERCM.cuda(),f_energies.cuda()) 
        t2=time.time()
        self.model.zero_grad()
        R,G,E,z,mean,var,graph_info=self.model(BMGs)
        loss_e=compute_energy_loss(f_energies,E)
        loss_reconstruct=compute_reconstruct_loss(f_GIERCM,R)
        loss_KL=compute_KL_loss(mean,var)
        loss_sum=loss_KL*GP.NNsetting.Lossw['KL']+loss_reconstruct*GP.NNsetting.Lossw['Re']+loss_e*GP.NNsetting.Lossw['E']
        t3=time.time()
        loss_sum.backward()
        self.optimizer.step()
        t4=time.time()
        self.lr_scheduler.step(loss_sum)
        self.Total_loss+=loss_sum.item()
        self.Total_loss_e+=loss_e.item()
        self.Total_loss_reconstruct+=loss_reconstruct.item()
        self.Total_loss_KL+=loss_KL.item()
        self.recordnum+=1
        self.lr=self.optimizer.state_dict()['param_groups'][0]['lr']
        t5=time.time()
        #print (f' Total Time of Training Batch: {t5-t1:.4E}, Input: {(t2-t1)/(t5-t1):.2E}, Loss: {(t3-t2)/(t5-t1):.2E}, Backward: {(t4-t3)/(t5-t1):.2E}')
        return 

    def evaluate_step(self,BMGs):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds,f_GIERCM,f_energies= BMGs.get_components()
        if GP.NNsetting.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_GIERCM,f_energies = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(),f_GIERCM.cuda(),f_energies.cuda()) 
        self.model.zero_grad()
        R,G,E,z,mean,var,graph_info=self.model(BMGs)
        loss_e=compute_energy_loss(f_energies,E)
        loss_reconstruct=compute_reconstruct_loss(f_GIERCM,R)
        loss_KL=compute_KL_loss(mean,var)
        loss_sum=loss_KL*GP.NNsetting.Lossw['KL']+loss_reconstruct*GP.NNsetting.Lossw['Re']+loss_e*GP.NNsetting.Lossw['E']
        self.Valid_loss+=loss_sum.item()
        self.Valid_loss_e+=loss_e.item()
        self.Valid_loss_reconstruct+=loss_reconstruct.item()
        self.Valid_loss_KL+=loss_KL.item()
        self.vrecordnum+=1
        return 

    def log_training(self,index,epochnum,steps):
        logstr=f'Step:{index}/{epochnum} -- {steps} ; Lr: {self.lr:.3E} ; Total Loss: {self.Total_loss/self.recordnum:.3E} / {self.Valid_loss/self.vrecordnum:.3E}; E: {self.Total_loss_e/self.recordnum:.3E} / {self.Valid_loss_e/self.vrecordnum:.3E} ; Restruc: {self.Total_loss_reconstruct/self.recordnum:.3E} / {self.Valid_loss_reconstruct/self.vrecordnum:.3E} ; KL: {self.Total_loss_KL/self.recordnum:.3E} / {self.Valid_loss_KL/self.vrecordnum:.3E} \n'
        return logstr
    
    def fit(self,MGdatasets,epochnum=100,rate=0.95,save_freq=5,rmsd_freq=10):
        tsteps_per_epoch=math.ceil(len(MGdatasets.molgraphs)*rate/GP.NNsetting.batchsize)
        vsteps_per_epoch=math.ceil(len(MGdatasets.molgraphs)*(1-rate)/GP.NNsetting.batchsize)
        if GP.NNsetting.optimizer=='Adam':
            self.optimizer=torch.optim.Adam(self.model.parameters(),lr=GP.NNsetting.initlr)
        if GP.NNsetting.lrsch=='ReduceLR':
            self.lr_scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.6, patience=tsteps_per_epoch*10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=tsteps_per_epoch*20, min_lr=1e-07, eps=1e-07)
        import time 
        from tqdm import tqdm 
        for epoch in range(epochnum):
            pbar= tqdm(range(tsteps_per_epoch))
            for steps in pbar:
                t1=time.time()
                BMGs=MGdatasets.get_BatchMGs(mode='train')
                t2=time.time()
                self.training_step(BMGs)
                t3=time.time()
                pbar.set_description(f"Reconstruct: {self.Total_loss_reconstruct/self.recordnum:0.3E} ")
                #print (f'Total Time: {t3-t1}, Prepare: {(t2-t1)/(t3-t1)}')
            
            for steps in range(vsteps_per_epoch):
                BMGs=MGdatasets.get_BatchMGs(mode='validation')
                self.evaluate_step(BMGs)
            logstr=self.log_training(epoch,epochnum,steps)
            #print (logstr)
            self.logger.write(logstr)
            self.logger.flush()
            self.training_history.append([epoch,self.Total_loss,self.Total_loss_e,self.Total_loss_reconstruct,self.Total_loss_KL,self.Valid_loss,self.Valid_loss_e,self.Valid_loss_reconstruct,self.Valid_loss_KL])
            self.epochs+=1
            if epoch%rmsd_freq==0:
                BMGs=MGdatasets.get_BatchMGs(mode='test')
                rmsdstr=self.evaluate_rmsd_step(BMGs)
                logstr=f'Step:{epoch}/{epochnum} -- '+rmsdstr 
                self.logger.write(logstr)
            if epoch%save_freq==0 or epoch>100:
                #print (self.Valid_loss,self.min_val_loss)
                if self.Valid_loss < self.min_val_loss :
                    self.min_val_loss=self.Valid_loss
                    print (f'Save New check point at Epoch: {epoch}!')
                    savepath=f'{self.modelname}/model/model.cpk'
                    savedict={'epochs':self.epochs, 'learningrate':self.lr,'lossmin':self.min_val_loss,'state_dict':self.model.state_dict(),'training_history':self.training_history}
                    torch.save(savedict,savepath)
            self.__tmprecord_clean()

    def __tmprecord_clean(self):
        self.Total_loss=0
        self.Total_loss_e=0
        self.Total_loss_reconstruct=0
        self.Total_loss_KL=0
        self.recordnum=0 
        self.Valid_loss=0
        self.Valid_loss_e=0
        self.Valid_loss_reconstruct=0 
        self.Valid_loss_KL=0
        self.vrecordnum=0
        return 
    
    def evaluate_rmsd_step(self,BMGs):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds,f_GIERCM,f_energies= BMGs.get_components()
        if GP.NNsetting.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb,f_GIERCM,f_energies = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(),f_GIERCM.cuda(),f_energies.cuda())
        self.model.zero_grad()
        R,G,E,z,mean,var,graph_info=self.model(BMGs)
        GIERCM=[G[i].cpu().detach().numpy() for i in range(len(G))]
        totalrms=[]
        totaldrms=[]
        mnum=0
        from tqdm import tqdm 
        os.system(f'mkdir -p {self.modelname}/{self.epochs}_TestRMSD')
        pbar=tqdm(BMGs.molgraphs)
        for mid,mol in enumerate(pbar):
            try:
                feature_matrix=np.array(mol.coordinate_feature_on_2D_graph)
                reverse_coord1=mol.decode_total_coordinate_feature_on_2D_graph(feature_matrix,with_rank=False)
                decode_feature_matrix=GIERCM[mid]
                print (decode_feature_matrix.shape)
                reverse_coord2=mol.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=False)
                rms=kabsch_rmsd(reverse_coord1,reverse_coord2,translate=True)
                totalrms.append(rms)
                ori_zd=mol.zdihedral
                tmpmol=deepcopy(mol)
                tmpmol.d3coord=reverse_coord2
                tmpmol.build_Zmatrix_on_2D_graph()
                decode_zd=tmpmol.zdihedral
                dm=abs(decode_zd-ori_zd)
                dm=np.where(dm>360*0.5,360-dm,dm)
                drms=np.sqrt(np.sum((dm)**2)/len(decode_zd))
                totaldrms.append(drms)
                pbar.set_description(f"RMSD: {rms:0.2f} , Torsion RMSD: {drms:0.2f}")
                write_xyz(f'{self.modelname}/{self.epochs}_TestRMSD/model_{self.epochs}_id_{mnum}_decode.xyz',mol.atoms,reverse_coord2,rms)
                write_xyz(f'{self.modelname}/{self.epochs}_TestRMSD/model_{self.epochs}_id_{mnum}_original.xyz',mol.atoms,reverse_coord1,rms)
                mnum+=1
            except Exception as e:
                print (f'{mid}th test molecule failed due to {e}')
        logstr=f'Testset RMSD (min/average): {np.min(totalrms)} / {np.average(totalrms)} ; Torsion RMSD (min/average): {np.min(totaldrms)} / {np.average(totaldrms)}'
        return logstr
    
    def Interpolation(self,mg1,mg2,num,path):
        os.system(f'mkdir -p {path}')
        BMGs1=BatchMolGraph([mg1])
        R1,G1,E1,z1,mean1,var1,graph_info1=self.model(BMGs1)
        BMGs2=BatchMolGraph([mg2])
        R2,G2,E2,z2,mean2,var2,graph_info2=self.model(BMGs2)
        #print (graph_info1,graph_info2)
        interval=(z2-z1)/20
        conf_dict={}
        conf_dict["Molgraph1"]=mg1
        conf_dict["Molgraph2"]=mg2
        conf_dict["interval"]={}
        for i in range(num+1):
            tmpz=interval*i+z1 
            G=self.model.decode_graph_z(graph_info1,tmpz,BMGs1) 
            decode_feature_matrix=G[0].cpu().detach().numpy()
            reverse_coord2=mg1.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=False)
            conf_dict["interval"][i]=[(reverse_coord2[index]) for index in range(len(reverse_coord2))]
            write_xyz(f'{path}/{i}_decode.xyz',mg1.atoms,reverse_coord2,0)
        return conf_dict

    def TS_search(self,mg1,mg2,path,step=50,samplenum=50,charge=1):
        os.system(f'mkdir -p {path}')
        BMGs1=BatchMolGraph([mg1])
        xyz1=mg1.d3coord 
        xyz2=mg2.d3coord 
        print (xyz1,xyz2)
        R1,G1,E1,z1,mean1,var1,graph_info1=self.model(BMGs1)
        return 
