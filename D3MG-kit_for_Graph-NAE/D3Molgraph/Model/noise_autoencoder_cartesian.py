 
import pickle 
import tempfile 
import tensorflow as tf
import shutil
import zipfile
import numpy as np
from tensorflow.keras.models import Model, load_model 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler,ModelCheckpoint,Callback
from tensorflow.keras import backend as K
import os 
import math
from ..Base import * 
from tqdm import tqdm  
from ..Datastruc import *
from copy import deepcopy
 
class Noise_AE_coordinate(tf.keras.Model):
    def __init__(self,inputdim,latentdim,noise_percent):
        super(Noise_AE_coordinate,self).__init__()
        self.latentdim = latentdim
        self.noise_percent=noise_percent 
        """
        self.encoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(inputdim[0],inputdim[1])),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(self.latentdim)
        ])
        self.decoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latentdim,)),
            tf.keras.layers.Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(inputdim[0]*inputdim[1]),
            tf.keras.layers.Reshape(target_shape=(inputdim[0],inputdim[1])),
        ])
        """
        self.encoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(inputdim[0],inputdim[1],1)),
            tf.keras.layers.Conv2D(128,kernel_size=3,strides=(1,1),activation='relu',padding="SAME"),
            tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=(1,1),activation='relu',padding="SAME"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(self.latentdim)
        ])
        self.decoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latentdim,)),
            tf.keras.layers.Dense(inputdim[0]*inputdim[1]*128,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Reshape(target_shape=(inputdim[0],inputdim[1],128)),
            tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=3,strides=(1, 1),padding="SAME",activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=3,strides=(1, 1),padding="SAME",activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")
        ])
    @tf.function
    def sample(self,eps=None):
        if eps is None:
            eps=tf.random.normal(shape=(100,self.latentdim))
        return self.decode(eps,apply_sigmoid=False)
    def encode(self,x):
        mean=self.encoder(x)
        return mean
    def reparameterize(self,mean,with_noise=True):
        if with_noise:
            eps=tf.random.normal(shape=mean.shape)
            return (1+eps*self.noise_percent)*mean
        else:
            return mean 
    def decode(self,z,apply_sigmoid=False):
        logits=self.decoder(z)
        if apply_sigmoid:
            probs=tf.sigmoid(logits)
            return probs
        return logits 

class Convolutional_Noise_Autoencoder_coordinate:
    def __init__(self,**kwargs):
        if 'x' in kwargs:
            xdata=kwargs.get("x")
            if "modelname" not in kwargs:
                self.mode="train"
                self.dataname=kwargs.get('dataname','Data')
                self.latentdim=kwargs.get('latentdim',128)
                self.inputdim=xdata.shape[1:]
                print(self.inputdim)
                self.trainingsteps=0
                self.training_history=[]
                self.type=kwargs.get('type','dcartesian')
            else:
                self.mode="retrain"
        elif "modelname" in kwargs:
            self.mode="test"     
        else:
            raise NameError("Cannot infer mode from arguments.")        
        if self.mode=="train":
                self.batchsize=kwargs.get("batchsize",128)
                self.activate_function=kwargs.get("activate_function","relu")
                self.learningrate=kwargs.get("lr",0.001)
                self.noise_percent=kwargs.get("noise_percent",0.01)
                if os.path.exists(f'{self.dataname}/model'):
                    os.system(f'mkdir -p {self.dataname}/model')
                self.__build_model()
        else:
            self.modelname=kwargs.get("modelname")
            self.load(self.modelname)
        #self.__build_model()
        #print (self.model.summary())
        return 

    def load(self, model_name):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(model_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            # Load metadata
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            self.__dict__.update(metadata)
            try:
                self.__build_model()
                lastest=tf.train.latest_checkpoint(dirpath + "/model/")
                self.model.load_weights(lastest)
            except:
                print("'model' not found, setting to None.")
                self.model = None

    def __clean(self):
        self.model=None

    def save(self,modelname=''):
        self.__clean()
        pickle.dump(self.__dict__,open(self.dataname+"/modelsetting.pickle", "wb"))

        if modelname=='':
            shutil.make_archive("Model_For_"+self.dataname,"zip",self.dataname)
        else:
            shutil.make_archive(modelname,"zip",self.dataname)
        os.system('rm -r %s'%self.dataname) 
        return 

    def __build_model(self):
        self.model=Noise_AE_coordinate(latentdim=self.latentdim,inputdim=self.inputdim,noise_percent=self.noise_percent)

    def __compute_loss(self,x,with_noise):
        mean=self.model.encode(x)
        z=self.model.reparameterize(mean,with_noise)
        x_logit=self.model.decode(z)
        mse=tf.keras.losses.MeanSquaredError()
        reconstruct_loss=tf.cast(mse(x_logit,x),tf.float32)
        #print (kl_div,reconstruct_loss)
        loss=reconstruct_loss
        return loss,reconstruct_loss

    def fit(self,x,epochnum=10,valx=None,logfile='train.log',splitrate=0.9,lr=0.0001,with_noise=True):
        cutnum=math.ceil(len(x)*0.9)
        x=np.reshape(x,(-1,self.inputdim[0],self.inputdim[1],1))
        traindb=tf.data.Dataset.from_tensor_slices(x[:cutnum]).shuffle(self.batchsize*5).batch(self.batchsize)
        valdb=tf.data.Dataset.from_tensor_slices(x[cutnum:]).batch(self.batchsize)
        trainstepnum=math.ceil(cutnum/self.batchsize)
        if not os.path.exists(self.dataname+'/model'):
            os.system('mkdir -p %s'%(self.dataname+'/model'))
        optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        for epoch in range(epochnum):
            self.trainingsteps+=1
            train_reloss=0
            for step, inputx in enumerate(traindb):
                with tf.GradientTape() as tape:
                    loss,reloss=self.__compute_loss(inputx,with_noise) 
                gradients=tape.gradient(loss,self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
                train_reloss+=reloss*inputx.shape[0]
            val_reloss=0
            for step, inputx in enumerate(valdb):
                loss,reloss=self.__compute_loss(inputx,with_noise)
                val_reloss+=reloss*inputx.shape[0]
            self.training_history.append([train_reloss,val_reloss])
            if epoch>10:
                if val_reloss < np.min(np.array(self.training_history)[:,1][:-1]) and epoch%10==0:
                    os.system(f'rm {self.dataname}/model/model* -r ')
                    self.model.save_weights(self.dataname+f'/model/model-{epoch}-{val_reloss:0.4f}',overwrite=True)
                    print (f'model-{self.trainingsteps}-{val_reloss:0.4f} is saved!')
            if epoch%5==0:
                print (f'Epoch {epoch}/{epochnum} Training Reconstruction Loss: {train_reloss/cutnum} ; Val Reconstruction Loss: {val_reloss/(len(x)-cutnum)}')
        return

    def evaluate_molgraphrmsd(self,MGset,with_noise=True):
        pbar=tqdm(MGset.molgraphs)
        totalrms=[]
        totaldrms=[]
        mnum=0
        os.system(f'mkdir -p {self.trainingsteps}')
        for mol in pbar:
            #try:
                if self.type=='dcartesian' or self.type=='rcartesian':
                    mol.EGCM_and_Rank_on_2D_graph()
                    if self.type=='dcartesian':
                        d3coord=mol.d3coord
                    else:
                        d3coord=mol.build_d3coord_on_2D_graph()
                    inputcoord=d3coord.reshape((-1,self.inputdim[0],self.inputdim[1],1))
                    mean=self.model.encode(inputcoord)
                    z=self.model.reparameterize(mean,with_noise)
                    decodecoord=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
                    rms=kabsch_rmsd(d3coord,decodecoord,translate=True)
                    totalrms.append(rms)
                    write_xyz(f'{self.dataname}/{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_decode.xyz',mol.atoms,decodecoord,rms)
                    write_xyz(f'{self.dataname}/{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_original.xyz',mol.atoms,d3coord,rms)
                    ori_zd=mol.zdihedral
                    tmpmol=deepcopy(mol)
                    tmpmol.d3coord=decodecoord
                    tmpmol.build_Zmatrix_on_2D_graph()
                    decode_zd=tmpmol.zdihedral
                    dm=abs(decode_zd-ori_zd)
                    dm=np.where(dm>360*0.5,360-dm,dm)
                    drms=np.sqrt(np.sum((dm)**2)/len(decode_zd))
                    totaldrms.append(drms)
                    pbar.set_description(f"RMSD: {rms:0.2f} , Torsion RMSD: {drms:0.2f}")
                elif self.type=="internal":
                    mol.EGCM_and_Rank_on_2D_graph()
                    d3coord=mol.d3coord
                    mol.build_Zmatrix_on_2D_graph()
                    Zmat=np.array([mol.zbond,mol.zangle,mol.zdihedral])
                    inputZmat=Zmat.reshape((-1,self.inputdim[0],self.inputdim[1],1))
                    mean=self.model.encode(inputZmat)
                    z=self.model.reparameterize(mean,with_noise)
                    decodeZmat=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
                    tmpmol=deepcopy(mol)
                    decodecoord=tmpmol.build_d3coord_on_2D_graph(zb=decodeZmat[0],za=decodeZmat[1],zd=decodeZmat[2])
                    rms=kabsch_rmsd(d3coord,decodecoord,translate=True)
                    totalrms.append(rms)
                    inputzd=mol.zdihedral
                    decodezd=decodeZmat[2]
                    dm=np.abs(decodezd-inputzd)
                    dm=np.where(dm>360*0.5,360-dm,dm)
                    drms=np.sqrt(np.sum(dm**2)/len(decodezd))
                    totaldrms.append(drms)
                    write_xyz(f'{self.dataname}/{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_decode.xyz',mol.atoms,decodecoord,rms)
                    write_xyz(f'{self.dataname}/{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_original.xyz',mol.atoms,d3coord,rms)                
                    ori_zd=mol.zdihedral
                    pbar.set_description(f"RMSD: {rms:0.2f} , Torsion RMSD: {drms:0.2f}")
                mnum+=1  
            #except:
            #    pass
        print (f'Teset min rms: {np.min(totalrms)} , average rms: {np.average(totalrms)}')
        print (f'Teset min drms: {np.min(totaldrms)} , average drms: {np.average(totaldrms)}')
        return totalrms,totaldrms 

    def Interpolation(self,mg1,mg2,num=20):
        orimol=deepcopy(mg1)
        rdkitmol=orimol.molobj
        rdkitmol_H=Chem.AddHs(rdkitmol) 
        AllChem.UFFGetMoleculeForceField(rdkitmol_H)
        ff.Minimize()
        newmol=Chem.RemoveHs(rdkitmol_H)
        conformer=newmol.GetConformer(0).GetPositions()
        orimol.d3coord=conformer
        orimol.build_Zmatrix_on_2D_graph()
        orizb=orimol.zbond
        oriza=orimol.zangle
        if self.type=='dcartesian':
            feature1=mg1.d3coord
            feature2=mg2.d3coord
        elif self.type=='rcartesian':
            #mg1.build_Zmatrix_on_2D_graph()
            #mg2.build_Zmatrix_on_2D_graph()
            feature1=mg1.build_d3coord_on_2D_graph().reshape((-1,self.inputdim[0],self.inputdim[1]))
            feature2=mg1.build_d3coord_on_2D_graph().reshape((-1,self.inputdim[0],self.inputdim[1]))
        elif self.type=='internal':
            mg1.build_Zmatrix_on_2D_graph()
            mg2.build_Zmatrix_on_2D_graph()
            feature1=np.array([mg1.zbond,mg1.zangle,mg1.zdihedral]).reshape((-1,self.inputdim[0],self.inputdim[1]))
            feature2=np.array([mg2.zbond,mg2.zangle,mg2.zdihedral]).reshape((-1,self.inputdim[0],self.inputdim[1]))

        z1=self.model.encode(feature1)
        z2=self.model.encode(feature2)
        interval=(z2-z1)/20
        conf_dict={}
        conf_dict["Molgraph1"]=mg1
        conf_dict["Molgraph2"]=mg2
        conf_dict["interval"]={}
        for i in range(num+1):
            try:
                mean=z1+interval*i
                z=self.model.reparameterize(mean,with_noise=False)
                decode_feature_matrix=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
                if self.type=='dcartesian':
                    reverse_coord2=decode_feature_matrix
                elif self.type=='rcartesian':
                    reverse_coord2=decode_feature_matrix
                elif self.type=='internal':
                    tmpzb=reverse_coord2[0]
                    tmpza=reverse_coord2[1]
                    tmpzd=reverse_coord2[2]
                    reverse_coord2=mg1.build_d3coord_on_2D_graph(zb=tmpzb,za=tmpza,zd=tmpzd)
                tmpmol=deepcopy(mg1)
                tmpmol.d3coord=reverse_coord2
                tmpmol.build_Zmatrix_on_2D_graph()
                reverse_coord2_optimize=tmpmol.build_d3coord_on_2D_graph(zb=orizb,za=oriza,zd=tmpmol.zdihedral)
                rms=kabsch_rmsd(reverse_coord2_optimize,reverse_coord2,translate=True)
                conf_dict["interval"][i]=(reverse_coord2 ,reverse_coord2_optimize, rms)
            except:
                print (i)

        return conf_dict 
              
            


        

         

    


            
        
        
        

