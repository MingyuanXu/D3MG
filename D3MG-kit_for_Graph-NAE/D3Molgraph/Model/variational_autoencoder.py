 
import pickle 
import tempfile 
from pickle import STOP
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
class CVAE(tf.keras.Model):
    def __init__(self,inputdim,latentdim):
        super(CVAE,self).__init__()
        self.latentdim = latentdim
        self.encoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(inputdim[0],inputdim[1],1)),
            tf.keras.layers.Conv2D(128,kernel_size=3,strides=(1,1),activation='relu',padding="SAME"),
            tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=(1,1),activation='relu',padding="SAME"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(self.latentdim+self.latentdim)
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
        mean,logvar=tf.split(self.encoder(x),num_or_size_splits=2,axis=1)
        return mean,logvar
    def reparameterize(self,mean,logvar):
        eps=tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar*0.5)+mean
    def decode(self,z,apply_sigmoid=False):
        logits=self.decoder(z)
        if apply_sigmoid:
            probs=tf.sigmoid(logits)
            return probs
        return logits 



class Convolutional_Variational_Autoencoder:
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
        self.model=CVAE(latentdim=self.latentdim,inputdim=self.inputdim)

    def __compute_loss(self,x):
        mean,logvar=self.model.encode(x)
        z=self.model.reparameterize(mean,logvar)
        x_logit=self.model.decode(z)
        mse=tf.keras.losses.MeanSquaredError()
        reconstruct_loss=tf.cast(mse(x_logit,x),tf.float32)
        kl_div=-0.5*(logvar+1-mean**2-tf.exp(logvar))
        kl_div=tf.cast(tf.reduce_sum(kl_div)/x.shape[0]*1.0 ,tf.float32)
        #print (kl_div,reconstruct_loss)
        loss=reconstruct_loss+0.0001*kl_div
        return loss,reconstruct_loss,kl_div 

    def fit(self,x,epochnum=10,valx=None,logfile='train.log',splitrate=0.9,lr=0.0001):
        cutnum=math.ceil(len(x)*0.9)
        x=np.reshape(x,(-1,self.inputdim[0],self.inputdim[1],1))
        traindb=tf.data.Dataset.from_tensor_slices(x[:cutnum]).shuffle(self.batchsize*5).batch(self.batchsize)
        valdb=tf.data.Dataset.from_tensor_slices(x[cutnum:]).batch(self.batchsize)
        trainstepnum=math.ceil(cutnum/self.batchsize)
        
        print (self.dataname)

        if not os.path.exists(self.dataname+'/model'):
            print (self.dataname)
            os.system('mkdir -p %s'%(self.dataname+'/model'))

        optimizer=tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        for epoch in range(epochnum):
            self.trainingsteps+=1
            #print (f'--------------- epoch {self.trainingsteps} -----------------' )
            train_kldiv=0;train_reloss=0
            for step, inputx in enumerate(traindb):
                with tf.GradientTape() as tape:
                    loss,reloss,kldiv=self.__compute_loss(inputx) 
                gradients=tape.gradient(loss,self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
                train_kldiv+=kldiv*inputx.shape[0]
                train_reloss+=reloss*inputx.shape[0]
                #if step %2 ==0: 
                #    print (f'Training Step: {step}/{trainstepnum} KL Div: {kldiv} ; Reconstruction Loss: {reloss}')

            val_kldiv=0;val_reloss=0
            for step, inputx in enumerate(valdb):
                loss,reloss,kldiv=self.__compute_loss(inputx)
                val_kldiv+=kldiv*inputx.shape[0]
                val_reloss+=reloss*inputx.shape[0]
                #    print (f'Validation Step: {step}/{trainstepnum} KL Div: {kldiv} ; Reconstruction Loss: {reloss}')

            self.training_history.append([train_kldiv,train_reloss,val_kldiv,val_reloss])

            if epoch>1:
                if val_reloss < np.min(np.array(self.training_history)[:,3][:-1]) and epoch%50==0:
                    os.system(f'rm {self.dataname}/model/model* -r ')
                    self.model.save_weights(self.dataname+f'/model/model-{epoch}-{val_reloss:0.4f}',overwrite=True)
                    print (f'model-{self.trainingsteps}-{val_reloss:0.4f} is saved!')

            if epoch%5==0:
                print (f'Epoch {epoch}/{epochnum} Training KL Div: {train_kldiv/cutnum} Reconstruction Loss: {train_reloss/cutnum} ; Val KL Div: {val_kldiv/(len(x)-cutnum)} Reconstruction Loss: {val_reloss/(len(x)-cutnum)}')
        return

    def evaluate_molgraphrmsd(self,MGset):
        pbar=tqdm(MGset.molgraphs)
        totalrms=[]
        mnum=0
        os.system(f'mkdir -p {self.trainingsteps}')
        for mol in pbar:
            mol.EGCM_and_Rank_on_2D_graph()
            feature_matrix=np.array(mol.order_graph_node_coordinate_feature_on_2D_graph())
            reverse_coord1,loss1=mol.decode_total_coordinate_feature_on_2D_graph(feature_matrix,with_rank=True)
            feature_matrix=feature_matrix.reshape((-1,self.inputdim[0],self.inputdim[1],1))
            mean,logvar=self.model.encode(feature_matrix)
            z=self.model.reparameterize(mean,logvar)
            decode_feature_matrix=self.model.decode(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
            matrixrms=np.mean(np.sqrt((np.reshape(feature_matrix,(self.inputdim[0],self.inputdim[1]))-decode_feature_matrix)**2))
            reverse_coord2,loss2=mol.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=True)
            rmslist=[]
            for i in range(len(reverse_coord1)):
                rms=rmsd(reverse_coord1[i],reverse_coord2[i])
                rmslist.append(rms)
            minrms=np.min(rmslist)
            minid=np.argmin(rmslist)
            totalrms.append(rmslist)
            write_xyz(f'{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_decode.xyz',mol.atoms,reverse_coord2[minid],minrms)
            write_xyz(f'{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_original.xyz',mol.atoms,reverse_coord1[minid],minrms)                

            pbar.set_description(f"Min RMSD: {minrms:0.4f} average RMSD: {np.average(rmslist):0.4f} matrix RMS: {matrixrms :0.4f}")
            mnum+=1  
        print (f'Teset min rms: {np.min(totalrms)} , average rms: {np.average(totalrms)}')
        return    

     


            
        
        
        

