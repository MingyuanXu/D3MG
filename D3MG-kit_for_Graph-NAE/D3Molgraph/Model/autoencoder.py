 
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
from tqdm import tqdm 
from ..Datastruc import * 
from ..Base import * 
class Convolutional_Autoencoder:
    def __init__(self,**kwargs):
        if 'x' in kwargs:
            xdata=kwargs.get("x")
            if "modelname" not in kwargs:
                self.mode="train"
                self.dataname=kwargs.get('dataname','Data')
                self.latentdim=kwargs.get('latentdim',128)
                self.inputdim=xdata.shape[1:]
                if os.path.exists(self.dataname):
                    os.system(f'mkdir -p {self.dataname}/model') 
            else:
                self.mode="retrain"
        elif "modelname" in kwargs:
            self.mode="test"     
        else:
            raise NameError("Cannot infer mode from arguments.")
        
        if self.mode=="train":
                self.structure=kwargs.get("structrue",None)
                self.batchsize=kwargs.get("batchsize",128)
                self.activate_function=kwargs.get("activate_function","relu")
                self.learningrate=kwargs.get("lr",0.001)
                self.trainingsteps=0
                self.__build_model()
        else:
            self.modelname=kwargs.get("modelname")
            self.load(self.modelname)
        #self.__build_model()
        print (self.model.summary())
        return 

    def load(self, model_name):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(model_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            # Load metadata
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            # Re-load metadata
            #Loaddict2obj(metadata,self)
            self.__dict__.update(metadata)
            #try:
            if True:
                filelist=os.listdir(dirpath+'/model')
                print('------------------',filelist)
                scorelist=[filename.split('-')[-1].strip('\.hdf') for filename in filelist]
                minfile=filelist[np.argmin(scorelist)]
                print (minfile)
                self.model = load_model(dirpath + "/model/"+minfile)

                
            #except:
            #    print("'model' not found, setting to None.")
            #    self.model = None

    def __clean(self):
        self.model=None
        self.encoder=None
        self.decoder=None
        self.traininghistory=None

    def save(self,modelname=''):
        self.__clean()
        pickle.dump(self.__dict__,open(self.dataname+"/modelsetting.pickle", "wb"))
        filelist=os.listdir(self.dataname+'/model')
        scorelist=[filename.split('-')[-1].strip('\.hdf') for filename in filelist]
        minfilelist=[filelist[i] for i in np.argsort(scorelist)[:5]]
        for filename in filelist:
            if filename not in minfilelist:
                os.system('rm %s/%s'%(self.dataname+'/model',filename))
        if modelname=='':
            shutil.make_archive("Model_For_"+self.dataname,"zip",self.dataname)
        else:
            shutil.make_archive(modelname,"zip",self.dataname)
        os.system('rm -r %s'%self.dataname) 
        return 
    def __build_model(self):
        self.encoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.inputdim[0],self.inputdim[1],1)),
            tf.keras.layers.Conv2D(128,kernel_size=3,strides=(1,1),activation='relu',padding="SAME"),
            tf.keras.layers.Conv2D(filters=256,kernel_size=3,strides=(1,1),activation='relu',padding="SAME"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(self.latentdim)
        ])
        self.decoder=tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latentdim,)),
            tf.keras.layers.Dense(80640,activation="relu",kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Reshape(target_shape=(15,42,128)),
            tf.keras.layers.Conv2DTranspose(filters=256,kernel_size=3,strides=(1, 1),padding="SAME",activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=128,kernel_size=3,strides=(1, 1),padding="SAME",activation='relu'),        
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")
        ])
        self.encoder.summary()
        self.decoder.summary()
        input=tf.keras.layers.Input((self.inputdim[0],self.inputdim[1],1))
        encoded=self.encoder(input)
        decoded=self.decoder(encoded)
        self.model=tf.keras.models.Model(inputs=input,outputs=decoded)

    def fit(self,x,epochnum=10,valx=None,logfile='train.log',lr=0.0001,ifcontinue=False):
        if not os.path.exists(self.dataname+'/model'):
            os.system('mkdir -p %s'%(self.dataname+'/model'))

        optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8,
                              patience=10, min_lr=0.00000001,verbose=1,min_delta=0.0005)
        def get_callbacks():
            return [
                tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=200),
                ModelCheckpoint(filepath=self.dataname+'/model/model-{epoch}-{val_loss:0.4f}.h5', verbose=1, save_best_only=True,period=50),
                reduce_lr,
                tf.keras.callbacks.CSVLogger(filename=logfile,separator=' ',append=False)
            ]
        if not ifcontinue:
            self.model.compile(loss='mean_squared_error',optimizer=optimizer,
            metrics=['mean_squared_error'])
        
        self.traininghistory=self.model.fit(x,x,batch_size=self.batchsize,epochs=epochnum,validation_data=(valx,valx),callbacks=get_callbacks(),shuffle=True)
        self.trainingsteps+=epochnum 
        return 

    def evaluate(self,x,filename=''):
        result=self.model.evaluate(x,x,batch_size=self.batchsize)
        if filename=='':
            print ("test loss,test mse:",result)
        return result 

    def predict(self,x):
        result= self.model.predict(x)
        return result 

    def evaluate_molgraphrmsd(self,MGset):
        pbar=tqdm(MGset.molgraphs)
        totalrms=[]
        mnum=0
        os.system(f'mkdir -p {self.trainingsteps}')
        for mol in pbar:
            mol.EGCM_and_Rank_on_2D_graph()
            feature_matrix=mol.order_graph_node_coordinate_feature_on_2D_graph()
            feature_matrix=np.array(feature_matrix)
            if MGset.scaler:
                feature_matrix=MGset.scaler.inverse_transform(feature_matrix)
            reverse_coord1,loss1=mol.decode_total_coordinate_feature_on_2D_graph(feature_matrix,with_rank=True)
            feature_matrix=feature_matrix.reshape((-1,self.inputdim[0],self.inputdim[1],1))
            #z=self.encoder(feature_matrix)
            #decode_feature_matrix=self.decoder(z).numpy().reshape((self.inputdim[0],self.inputdim[1]))
            decode_feature_matrix=self.model.predict(feature_matrix).reshape((self.inputdim[0],self.inputdim[1]))
            if MGset.scaler:
                decode_feature_matrix=MGset.scaler.inverse_transform(decode_feature_matrix)
            #print (feature_matrix.shape,decode_feature_matrix.shape)
            reverse_coord2,loss2=mol.decode_total_coordinate_feature_on_2D_graph(decode_feature_matrix,with_rank=True)
            #print (reverse_coord1,reverse_coord2)
            rmslist=[]
            for i in range(len(reverse_coord1)):
                rms=rmsd(reverse_coord1[i],reverse_coord2[i])
                rmslist.append(rms)
            minrms=np.min(rmslist)
            minid=np.argmin(rmslist)
            totalrms.append(rmslist)
            write_xyz(f'{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_decode.xyz',mol.atoms,reverse_coord2[minid],minrms)
            write_xyz(f'{self.trainingsteps}/model_{self.trainingsteps}_id_{mnum}_original.xyz',mol.atoms,reverse_coord1[minid],minrms)                

            pbar.set_description(f"Min RMSD: {minrms:0.4f} average RMSD: {np.average(rmslist):0.4f}")
            mnum+=1  
        print (f'Teset min rms: {np.min(totalrms)} , average rms: {np.average(totalrms)}')
        return    
        

