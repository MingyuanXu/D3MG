from ..Datastruc import * 
import shutil
import zipfile
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler,ModelCheckpoint,Callback
from tensorflow.keras.layers import Input ,Dense 
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from spektral.layers import  GCNConv,GeneralConv 
from spektral.data import BatchLoader 
import pickle 
import tempfile 
import os 

class Graph_Autoencoder:
    def __init__(self,**kwargs):
        self.dataset=kwargs.get('dataset',None)
        self.modelname=kwargs.get('modelname',None)
        if self.dataset:
            if not self.modelname:
                self.dataname=kwargs.get('dataname','Data')
                if not os.path.exists(self.dataname):
                    os.system(f'mkdir -p {self.dataname}/model')
                self.mode="train"
                self.batchsize=kwargs.get('batchsize',128)
                self.structure=kwargs.get('structure',[64,128,64,self.dataset.node_feature_max_length])
                self.activate_function=kwargs.get("activate_function","relu")
                self.learningrate=kwargs.get("lr",0.0001) 
            else:
                self.mode="retrain"
        elif self.modelname:
            self.mode="test"
        else:
            raise NameError("Cannot infer mode from arguments")
        
        if self.mode=="train":
            self.__build_model()
        else:
            self.load(self.modelname)
            print (self.model.summary())
        return 
    def __build_model(self):        
        X_in=Input(shape=(self.dataset.n_node_features,))
        A_in=Input(shape=(None,),sparse=True)
        X_1=GCNConv(self.structure[0],activation=tf.nn.leaky_relu)([X_in,A_in])
        for channelnum in self.structure[1:]:
            X_1=GCNConv(channelnum,activation=tf.nn.leaky_relu)([X_1,A_in])
        X1=Dense(self.dataset.n_node_features)(X_1)
        #X_1=GeneralConv(self.structure[0],activation='prelu')([X_in,A_in])
        #for channelnum in self.structure[1:]:
        #    X_1=GeneralConv(channelnum,activation='prelu')([X_1,A_in])
        output=X_1
        self.model=Model(inputs=[X_in,A_in],outputs=output)
        return 

    def fit(self,epoch_num=1000,splitrate=0.9,logfile='train.log'):
        optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        reduce_lr=ReduceLROnPlateau(monitor='loss', factor=0.8,
                              patience=10, min_lr=0.00000001,verbose=1,min_delta=0.0005)
        def get_callbacks():
            return [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
                ModelCheckpoint(filepath=self.dataname+'/model/model-{epoch}-{val_loss:0.4f}.h5', verbose=1, save_best_only=True),
                reduce_lr,
                tf.keras.callbacks.CSVLogger(filename=logfile,separator=' ',append=False)
            ]
        trainingset,valset=self.dataset.split(splitrate)
        loader_trainingset=BatchLoader(trainingset,batch_size=self.batchsize)
        loader_valset=BatchLoader(valset,batch_size=self.batchsize)
        self.model.compile(loss='mse',optimizer=optimizer,metrics=['mean_squared_error'])
        self.traininghistory=self.model.fit(
            loader_trainingset.load(),
            steps_per_epoch=loader_trainingset.steps_per_epoch,
            validation_data=loader_valset.load(),
            validation_steps=loader_valset.steps_per_epoch,
            epochs=epoch_num,
            callbacks=get_callbacks()
        )
        return 

    def evaluate(self,graphset):
        loader=BatchLoader(graphset,batch_size=self.batchsize)
        eval_results=self.model.evaluate(loader.load(),steps=loader.steps_per_epoch)    
        print("Done.\n" "Test loss: {}\n" "Test MSE: {}".format(*eval_results))
        return eval_results

    def predict(self,graphset):
        loader=BatchLoader(graphset,batch_size=self.batchsize)
        print (loader.load().__dict__)
        predict_results=self.model.predict(loader.load(),steps=loader.steps_per_epoch)
        #predict_results=self.model.predict(x=input)#,steps=loader.steps_per_epoch)
        return predict_results

    def load(self, model_name):
        with tempfile.TemporaryDirectory() as dirpath:
            with zipfile.ZipFile(model_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)
            # Load metadata
            metadata = pickle.load(open(dirpath + "/modelsetting.pickle", "rb"))
            self.__dict__.update(metadata)
            #try:
            if True:
                filelist=os.listdir(dirpath+'/model')
                scorelist=[filename.split('-')[-1].strip('\.hdf') for filename in filelist]
                minfile=filelist[np.argmin(scorelist)]
                self.model = load_model(dirpath + "/model/"+minfile,custom_objects={'GCNConv':GCNConv,'GeneralConv':GeneralConv,'leaky_relu':tf.nn.leaky_relu})
            #except:
            #    print("'model' not found, setting to None.")
            #    self.model = None

    def __clean(self):
        self.model=None
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

    



