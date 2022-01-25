 
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
 
class Noise_AE(tf.keras.Model):
    def __init__(self,inputdim,latentdim,noise_percent):
        super(Noise_AE,self).__init__()
        self.latentdim = latentdim
        self.noise_percent=noise_percent 
        self.encoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(inputdim[0],inputdim[1],1)),
            tf.keras.layers.Conv2D(256,kernel_size=4,strides=(3,3),activation='tanh',padding="SAME"),
            tf.keras.layers.Conv2D(128,kernel_size=4,strides=(3,3),activation='tanh',padding="SAME"),
            tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=(1,1),activation='tanh',padding="SAME"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Dense(self.latentdim)
        ])
        self.decoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latentdim,)),
            tf.keras.layers.Dense(inputdim[0]*inputdim[1]*4,activation="tanh",kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.Reshape(target_shape=(inputdim[0],inputdim[1],4)),
            tf.keras.layers.Conv2DTranspose(filters=4,kernel_size=3,strides=(1, 1),padding="SAME",activation='tanh'),
            tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=(1, 1),padding="SAME",activation='tanh'),
            tf.keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=(1, 1),padding="SAME",activation='tanh'),
            tf.keras.layers.Conv2DTranspose(filters=4,kernel_size=3,strides=(1, 1),padding="SAME",activation='tanh'),
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

    def reparameterize(self,mean,with_noise=True,noise_percent=0):
        if with_noise:
            if noise_percent==0:
                noise_percent=self.noise_percent
            eps=tf.random.normal(shape=mean.shape)
            return (1+eps*noise_percent)*mean
        else:
            return mean
            
    def decode(self,z,apply_sigmoid=False):
        logits=self.decoder(z)
        if apply_sigmoid:
            probs=tf.sigmoid(logits)
            return probs
        return logits 

class Convolutional_Noise_Autoencoder:
    def __init__(self,**kwargs):
        if 'x' in kwargs:
            xdata=kwargs.get("x")
            if "modelname" not in kwargs:
                self.mode="train"
                self.dataname=kwargs.get('dataname','Data')
                self.latentdim=kwargs.get('latentdim',128)
                self.inputdim=xdata.shape[1:]
                self.trainingsteps=0
                self.training_history=[]
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
        self.model=Noise_AE(latentdim=self.latentdim,inputdim=self.inputdim,noise_percent=self.noise_percent)
        self.exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.0001, decay_steps=300, decay_rate=0.9)
        self.optimizer=tf.keras.optimizers.Adam(self.exponential_decay)
        self.model.encoder.summary()
        self.model.decoder.summary()

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
        print (self.dataname)
        if not os.path.exists(self.dataname+'/model'):
            print (self.dataname)
            os.system('mkdir -p %s'%(self.dataname+'/model'))
        #optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        for epoch in range(epochnum):
            self.trainingsteps+=1
            #print (f'--------------- epoch {self.trainingsteps} -----------------' )
            train_reloss=0
            for step, inputx in enumerate(traindb):
                with tf.GradientTape() as tape:
                    loss,reloss=self.__compute_loss(inputx,with_noise) 
                gradients=tape.gradient(loss,self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
                train_reloss+=reloss*inputx.shape[0]
                #if step %2 ==0: 
                #    print (f'Training Step: {step}/{trainstepnum} KL Div: {kldiv} ; Reconstruction Loss: {reloss}')

            val_reloss=0
            for step, inputx in enumerate(valdb):
                loss,reloss=self.__compute_loss(inputx,with_noise)
                val_reloss+=reloss*inputx.shape[0]
                #    print (f'Validation Step: {step}/{trainstepnum} KL Div: {kldiv} ; Reconstruction Loss: {reloss}')
            self.training_history.append([train_reloss,val_reloss])
            if epoch>10:
                if val_reloss < np.min(np.array(self.training_history)[:,1][:-1]) and epoch%10==0:
                    os.system(f'rm {self.dataname}/model/model* -r ')
                    self.model.save_weights(self.dataname+f'/model/model-{epoch}-{val_reloss:0.4f}',overwrite=True)
                    print (f'model-{self.trainingsteps}-{val_reloss:0.4f} is saved!')
            if epoch>200:
                if np.mean(np.array(self.training_history)[:,1][-50:-1])-np.mean(np.array(self.training_history)[:,1][-100:-50])>-0.000001:
                    print (f'Overfitting: Training process stopped!!')
                    break 
            if epoch%5==0:
                print (f'Epoch {epoch}/{epochnum} Training Reconstruction Loss: {train_reloss/cutnum} ; Val Reconstruction Loss: {val_reloss/(len(x)-cutnum)}')
            
        return

    def evaluate_molgraphrmsd2(self,MGset,with_noise=True,savedir='Teset'):
        pbar=tqdm(MGset.molgraphs)
        totalrms=[]
        totaldrms=[]
        mnum=0
        os.system(f'mkdir -p {self.trainingsteps}')
        for mol in pbar:
            try:
                mol.EGCM_and_Rank_on_2D_graph()
                feature_matrix=np.array(mol.order_graph_node_coordinate_feature_on_2D_graph())
                reverse_coord1=mol.decode_total_coordinate_feature_on_2D_graph(feature_matrix,with_rank=True)
                feature_matrix=feature_matrix.reshape((-1,self.inputdim[0],self.inputdim[1],1))
                mean=self.model.encode(feature_matrix)
                z=self.model.reparameterize(mean,with_noise)
                decode_feature_matrix=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
                reverse_coord2=mol.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=True)
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
                write_xyz(f'{self.dataname}/{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_decode.xyz',mol.atoms,reverse_coord2,rms)
                write_xyz(f'{self.dataname}/{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_original.xyz',mol.atoms,reverse_coord1,rms)                
                mnum+=1  
            except:
                pass
        print (f'Teset min rms: {np.min(totalrms)} , average rms: {np.average(totalrms)}')
        print (f'Teset min drms: {np.min(totaldrms)} , average rms: {np.average(totaldrms)}')
        return  

    def evaluate_molgraphrmsd(self,MGset,with_noise=True,savedir='Teset',noise_percent=0):
        random.shuffle(MGset.molgraphs)
        print (len(MGset.molgraphs))
        pbar=tqdm(MGset.molgraphs[:100])
        totalrms=[]
        totaldrms=[]
        mnum=0
        os.system(f'mkdir -p {self.dataname}/{self.trainingsteps}_{savedir}')
        for mol in pbar:
            #try:
                #mol.EGCM_and_Rank_on_2D_graph()
                feature_matrix=np.array(mol.order_graph_node_coordinate_feature_on_2D_graph())
                reverse_coord1=mol.decode_total_coordinate_feature_on_2D_graph(feature_matrix,with_rank=True)
                print (feature_matrix.shape)
                feature_matrix=feature_matrix.reshape((-1,self.inputdim[0],self.inputdim[1],1))
                mean=self.model.encode(feature_matrix)
                z=self.model.reparameterize(mean,with_noise,noise_percent=noise_percent)
                decode_feature_matrix=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
                reverse_coord2=mol.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=True)
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
                write_xyz(f'{self.dataname}/{self.trainingsteps}_{savedir}/model_{self.trainingsteps}_id_{mnum}_decode.xyz',mol.atoms,reverse_coord2,rms)
                write_xyz(f'{self.dataname}/{self.trainingsteps}_{savedir}/model_{self.trainingsteps}_id_{mnum}_original.xyz',mol.atoms,reverse_coord1,rms)
                mnum+=1
            #except:
            #    pass
        print (f'Teset min rms: {np.min(totalrms)} , average rms: {np.average(totalrms)}')
        print (f'Teset min drms: {np.min(totaldrms)} , average rms: {np.average(totaldrms)}')
        return
    def Interpolation_simple(self,mg1,mg2,num=20):
        if not mg1.coordinate_feature_on_2D_graph:
            mg1.generate_2D_graph()
            mg1.generate_graph_node_coordinate_feature_on_2D_graph(Rc=12,Rcs=10)
        mg1.EGCM_and_Rank_on_2D_graph()
        feature1=np.array(mg1.order_graph_node_coordinate_feature_on_2D_graph()).reshape((-1,self.inputdim[0],self.inputdim[1],1))
        if not mg2.coordinate_feature_on_2D_graph:
            mg2.generate_2D_graph()
            mg2.generate_graph_node_coordinate_feature_on_2D_graph(Rc=12,Rcs=10)
        mg2.EGCM_and_Rank_on_2D_graph()
        feature2=np.array(mg2.order_graph_node_coordinate_feature_on_2D_graph()).reshape((-1,self.inputdim[0],self.inputdim[1],1))
        z1=self.model.encode(feature1)
        z2=self.model.encode(feature2)
        interval=(z2-z1)/20
        conf_dict={}
        conf_dict["Molgraph1"]=mg1
        conf_dict["Molgraph2"]=mg2
        conf_dict["interval"]={}
        
        for i in range(num+1):
            #try:
                print (i)
                mean=z1+interval*i
                z=self.model.reparameterize(mean,with_noise=False)
                decode_feature_matrix=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
                reverse_coord2=mg1.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=True)
                conf_dict["interval"][i]=reverse_coord2         
            #except:
            #    print (i)

        return conf_dict 
    def Interpolation(self,mg1,mg2,num=20):
        if not mg1.coordinate_feature_on_2D_graph:
            mg1.generate_2D_graph()
            mg1.generate_graph_node_coordinate_feature_on_2D_graph(Rc=12,Rcs=10)
        mg1.EGCM_and_Rank_on_2D_graph()
        feature1=np.array(mg1.order_graph_node_coordinate_feature_on_2D_graph()).reshape((-1,self.inputdim[0],self.inputdim[1],1))
        if not mg2.coordinate_feature_on_2D_graph:
            mg2.generate_2D_graph()
            mg2.generate_graph_node_coordinate_feature_on_2D_graph(Rc=12,Rcs=10)
        mg2.EGCM_and_Rank_on_2D_graph()
        feature2=np.array(mg2.order_graph_node_coordinate_feature_on_2D_graph()).reshape((-1,self.inputdim[0],self.inputdim[1],1))
        z1=self.model.encode(feature1)
        z2=self.model.encode(feature2)
        interval=(z2-z1)/20
        conf_dict={}
        conf_dict["Molgraph1"]=mg1
        conf_dict["Molgraph2"]=mg2
        conf_dict["interval"]={}
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
        for i in range(num+1):
            #try:
                print (i)
                mean=z1+interval*i
                z=self.model.reparameterize(mean,with_noise=False)
                decode_feature_matrix=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
                reverse_coord2=mg1.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=True)
                tmpmol=deepcopy(mg1)
                tmpmol.d3coord=reverse_coord2
                tmpmol.build_Zmatrix_on_2D_graph()
                reverse_coord2_optimize=tmpmol.build_d3coord_on_2D_graph(zb=orizb,za=oriza,zd=tmpmol.zdihedral)
                rms=rmsd(reverse_coord2_optimize,reverse_coord2)
                conf_dict["interval"][i]=(reverse_coord2 ,reverse_coord2_optimize, rms)
            #except:
            #    print (i)
        return conf_dict              
            


        

         

    


            
        
        
        

