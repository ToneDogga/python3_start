 #!/usr/bin/env python


# sales_trans predict v1 by Anthony Paech written 25/5/20
# A simple, efficient sales analysis tool
# uses only TF2 functions for speed
# simple intended to test use of TFRecords, TF functions






# sales trans lib contains the sales trans class
# contains
# load - loads the salestrans files from CSV or excel into TFRecords
# query - loads the queryfile from excel and creates a plot dictionary of each query
# preprocess - preprocesses the TFRecords from sales trans.  Applies a MAT's and updates the query dictionary
#                 save the plot dictionary and the queries
# loop through each query 
#   batch - create X batches
#   create Y - create Y batches from X batches
#   train - apply the batches to model and save each model
#  
#   predict - load model and predict into plot dictionary
#
#   results - plot the plot dictionary and send each prediction to excel by month

import os
os.chdir("/home/tonedogga/Documents/python_dev")
cwdpath = os.getcwd()


# print("\n\nActual vs Expected- Sales crystal ball stack : TF2 Salestrans predict - By Anthony Paech 25/5/20")
# print("================================================================================================\n")       
 


#import sales_trans_lib_v6
#import ave_lib

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# TensorFlow ≥2.0 is required
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.set_memory_growth(gpus[0], True)

tf.config.run_functions_eagerly(False)
#tf.config.experimental_run_functions_eagerly(False)   #True)   # false

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

#tf.autograph.set_verbosity(3, True)




from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"

#if not tf.config.list_physical_devices('GPU'):
#    print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
#    if IS_COLAB:
#        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")


# Disable all GPUS 
#tf.config.set_visible_devices([], 'GPU') 



 #visible_devices = tf.config.get_visible_devices() 
# for device in visible_devices: 
#     print(device)
#     assert device.device_type != 'GPU' 

#tf.config.set_visible_devices([], 'GPU') 
#tf.config.set_visible_devices(visible_devices, 'GPU') 


#import keras.backend as K


# # Common imports
import numpy as np
#import os
from pathlib import Path
import pandas as pd
import pickle
# #import random
#import datetime as dt

#import multiprocessing

#from numba import cuda
# ""
import collections
from collections import defaultdict
# from datetime import datetime
# #import SCBS0 as c

#import ave_lib

# # to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

import datetime as dt
from datetime import date
from datetime import timedelta
from datetime import datetime

from pathlib import Path

from p_tqdm import p_map,p_umap

import dash2_dict as dd2
#


# # To plot pretty figures
# #%matplotlib inline
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
#ion() # enables interactive mode
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


   
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

#st=ave_lib.actual_vs_expected_class()   # instantiate the sales trans class

   
np.random.seed(42)
tf.random.set_seed(42)
       



###########################################3



# actual vs expected
class ave_class(object): 
    def __init__(self):   
    #     self.epochs=8
    # #    self.steps_per_epoch=100 
    #     self.no_of_batches=1000
    #     self.no_of_repeats=2
        
    #     self.dropout_rate=0.2
    #     self.start_point=0
    #     self.end_point=732
    #     self.predict_ahead_length=500
    #     self.minimum_length=800
        
    #     self.batch_length=365
    #     self.predict_length=365
    #     self.one_year=366
    #     self.batch_jump=self.batch_length  # predict ahead steps in days. This is different to the batch length  #int(round(self.batch_length/2,0))
    #     self.neurons=1000   #self.batch_length
        
    #     self.data_start_date="02/02/18"
    #     self.date_len=1300      
    #     self.dates = pd.period_range(self.data_start_date, periods=self.date_len)   # 2000 days


    #     self.pred_error_sample_size=12
    #     self.no_of_stddevs_on_error_bars=1
    #     self.patience=5

    #     self.mat=28   #moving average in days #tf.constant([28],dtype=tf.int32)
            
    #   #  self.plot_dict_types=dict({0:"raw_database_or_model",1:"raw_query",2:"moving_total",3:"prediction",4:"stddev"})
        
    #     self.train_percent=0.7
    #     self.valid_percent=0.2
    #     self.test_percent=0.1
        
    #    self.filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]
    #    self.queryfilename="queryfile.xlsx"
    #    self.plot_dict_filename="plot_dict.pkl"
    #    self.sales_df_savename="sales_trans_df.pkl"
        
             
       # self.output_dir = self.log_dir("SCBS2")
        self.output_dir = self.log_dir("dash2")
        os.makedirs(self.output_dir, exist_ok=True)
        
      #  self.images_path = os.path.join(self.output_dir, "images/")
      #  os.makedirs(self.images_path, exist_ok=True)
     
     #   self.__version__="0.6.0"
        return
    
   
    
    def log_dir(self,prefix=""):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "./dash2_outputs"
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)
    
  
           
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
 
     
      
    
    @tf.function
    def sequential_indices(self,start_points,length_of_indices): 
        grid_indices=tf.meshgrid(tf.range(0,length_of_indices),start_points)   #int((end_point-start_point)/batch_length)+1)) #   print("gt=",gridtest)
        return tf.add(grid_indices[0],grid_indices[1])   #[:,:,np.newaxis
       
    
  
    # print("new Y shape",Y.shape)
    # for step_ahead in range(1, predict_ahead_length + 1):
    #     Y[...,step_ahead - 1] = series[..., step_ahead:step_ahead+batch_length-predict_ahead_length, 0]  #+1
   
    @tf.function
    def create_X_batches(self,series,batch_length,no_of_batches,start_point,end_point):
        start_points=tf.random.uniform(shape=[no_of_batches],minval=start_point,
                     maxval=end_point-(2*batch_length+1),dtype=tf.int32)
        return self.sequential_indices(start_points,batch_length)[...,tf.newaxis],self.sequential_indices(start_points,2*batch_length+1)[...,tf.newaxis]
 
    


 #   @tf.function
    def create_X_and_Y_batches(self,series,batch_length,no_of_batches):
        X_indices,full_indices=self.create_X_batches(series,batch_length,no_of_batches,dd2.dash2_dict['sales']['predictions']['start_point'],dd2.dash2_dict['sales']['predictions']['end_point'])
     #   print("X indices shape",X_indices.shape)
     #   print("full indices shape",full_indices.shape)
       
        
        batch_depth=X_indices.shape[1]
        Y_indices = np.empty((no_of_batches,batch_depth,batch_length),dtype=np.int32)
        
        for step_ahead in range(1, batch_depth + 1):
            Y_indices[:,:, step_ahead-1] = full_indices[:, step_ahead:step_ahead + batch_length,0]
     #   test_Y_indices=tf.gather(Y_indices[:,:, 0],full_indices[:, :batch_length,0],axis=1)
      #  print("X, X.shape",X_indices,X_indices.shape)  
      #  print("Y, Y.shape",Y_indices,Y_indices.shape)
                
 #       X=tf.cast(tf.gather(series[0],X_indices,axis=0),tf.int32)
 #       Y=tf.cast(tf.gather(series[0],Y_indices,axis=0),tf.int32)

        X=tf.gather(series[0],X_indices,axis=0)
        Y=tf.gather(series[0],Y_indices,axis=0)


    #    tf.print("2X[1]=",X[1],X.shape,"\n")
    #    tf.print("2Y[1]=",Y[1],Y.shape,"\n")

        return X,Y
  

    
    #@tf.autograph.experimental.do_not_convert
    def model_training_GRU(self,train_set,valid_set,query_name,output_dir):
        print("\nTraining with GRU and dropout")
        model = keras.models.Sequential([
      #     keras.layers.Conv1D(filters=st.batch_length,kernel_size=4, strides=1, padding='same', input_shape=[None, 1]),  #st.batch_length]), 
      #     keras.layers.BatchNormalization(),
           keras.layers.GRU(dd2.dash2_dict['sales']['predictions']['neurons'], return_sequences=True, input_shape=[None, 1]), #st.batch_length]),
           keras.layers.BatchNormalization(),
           keras.layers.GRU(dd2.dash2_dict['sales']['predictions']['neurons'], return_sequences=True),
           keras.layers.AlphaDropout(rate=dd2.dash2_dict['sales']['predictions']['dropout_rate']),
           keras.layers.BatchNormalization(),
           keras.layers.TimeDistributed(keras.layers.Dense(dd2.dash2_dict['sales']['predictions']['batch_length']))
        ])
    
        model.compile(loss="mse", optimizer="adam", metrics=[self.last_time_step_mse])
       
    #    model.summary() 
       
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=dd2.dash2_dict['sales']['predictions']['patience']),self.MyCustomCallback()]
       
        history = model.fit(train_set ,epochs=dd2.dash2_dict['sales']['predictions']['epochs'],
                           validation_data=(valid_set), callbacks=callbacks)
            
        print("\nsave model",query_name,":GRU_Dropout_sales_predict_model.h5\n")
        model.save(output_dir+query_name+":GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
             
        self.plot_learning_curves(history.history["loss"], history.history["val_loss"],dd2.dash2_dict['sales']['predictions']['epochs'],"GRU and dropout:"+str(query_name))
        self._save_fig("GRU and dropout learning curve_"+query_name,output_dir)
    #    plt.draw()
        
        
       # plt.show(block=False)
       # plt.pause(0.001)
        plt.close()                

     #   plt.show(block=False)
        return model    
        
 
        
 
    
 
    
 
    
 
    
 
    
     
    def last_time_step_mse(self,Y_true, Y_pred):
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
    
    
       
     
    def plot_learning_curves(self,loss, val_loss,epochs,title):
        if ((np.min(loss)<=0) or (np.max(loss)==np.inf) or (np.isnan(loss).any())):
            return
        if ((np.min(val_loss)<=0) or (np.max(val_loss)==np.inf) or (np.isnan(val_loss).any())):
            return
        if np.min(loss)>10:
            lift=10
        else:
            lift=1
        ax = plt.gca()
        ax.set_yscale('log')
  
        plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
        plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #    plt.axis([1, epochs+1, 0, np.max(loss[1:])])
      #  plt.axis([1, epochs+1, np.min(loss), np.max(loss)])
        plt.axis([1, epochs+1, np.min(loss)-lift, np.max(loss)])
    
        plt.legend(fontsize=14)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
    


    
  
            
  
    
  
    # @tf.function
    # def train_test_split(self,X):
    #     X=tf.random.shuffle(X)
    #     batch_length=X.shape[0]
    #     n_train=int(tf.round(batch_length*self.train_percent,0))
    #     n_valid=int(tf.round(batch_length*self.valid_percent,0))
        
    #     X_train=X[:n_train]
    #     X_valid=X[n_train:n_train+n_valid]
    #     X_test=X[n_train+n_valid:]

    #     return X_train,X_valid,X_test

   
    def simple_predict(self,model,series):
        # turn series into X batch format
        new_prediction=np.empty((1,0,1)) #self.batch_length))
        new_stddev=np.empty((1,0,1),dtype=np.float32)  #self.batch_length),dtype=np.float32)
 
        for r in range(0,dd2.dash2_dict['sales']['predictions']['predict_ahead_length'],dd2.dash2_dict['sales']['predictions']['batch_length']):           
            X_new=series[:,dd2.dash2_dict['sales']['predictions']['end_point']-dd2.dash2_dict['sales']['predictions']['predict_length']:dd2.dash2_dict['sales']['predictions']['end_point'],:]
       #     print("\nsimple predict - X new",X_new,X_new.shape,"\n\n")
       #     print("prediction line[0,-1]",model(X_new,training=False)[0,-1],model(X_new,training=False)[0,-1].shape)
       #     print("prediction line[0,batch_length-1]",model(X_new,training=False)[0,self.batch_length-1],model(X_new,training=False)[0,self.batch_length-1].shape)

       #     print("prediction line[0,:,-1]",model(X_new,training=False)[0,:,-1],model(X_new,training=False)[0,:,-1].shape)

        #model(X_new,training=True)[0,-1]
       
            Y_probs=np.stack([model(X_new,training=True)[0,-1].numpy() for sample in range(dd2.dash2_dict['sales']['predictions']['pred_error_sample_size'])])         
    
            Y_mean=Y_probs.mean(axis=0)
            Y_mean=Y_mean[np.newaxis,...,np.newaxis]
            
            Y_stddev=dd2.dash2_dict['sales']['predictions']['no_of_stddevs_on_error_bars']*Y_probs.std(axis=0)   #[np.newaxis,...]
            Y_stddev=Y_stddev[np.newaxis,...,np.newaxis]
    
            new_prediction=np.concatenate((new_prediction,Y_mean[:,:dd2.dash2_dict['sales']['predictions']['batch_length']]),axis=1) 
            new_stddev=np.concatenate((new_stddev,Y_stddev[:,:dd2.dash2_dict['sales']['predictions']['batch_length']]),axis=1) 
            series=np.concatenate((series,Y_mean[:,:dd2.dash2_dict['sales']['predictions']['batch_length']]),axis=1) 

        return new_prediction ,new_stddev

 

    
    class MyCustomCallback(tf.keras.callbacks.Callback):
    
      def on_train_begin(self, logs=None):
        print('Training:  begins at {}'.format(dt.datetime.now().time()))
    
      def on_train_end(self, logs=None):
        print('Training:  ends at {}'.format(dt.datetime.now().time()))
    
      def on_predict_begin(self, logs=None):
        print('Predicting: begins at {}'.format(dt.datetime.now().time()))
    
      def on_predict_end(self, logs=None):
        print('Predicting: ends at {}'.format(dt.datetime.now().time()))
    
    
    
    class MCDropout(keras.layers.Dropout):
         def call(self,inputs):
            return super().call(inputs,training=True)
    
    
    class MCAlphaDropout(keras.layers.AlphaDropout):
        def call(self,inputs):
            return super().call(inputs,training=True)
    
    
    
    
    
      
         
        
    # def _save(self,sales_df,save_dir,savefile):
    #     os.makedirs(save_dir, exist_ok=True)
    #     if isinstance(sales_df, pd.DataFrame):
    #         if not sales_df.empty:
    #            # sales_df=pd.DataFrame([])
    #            sales_df.to_pickle(save_dir+savefile,protocol=-1)
    #            return True
    #         else:
    #            return False
    #     else:
    #         return False
     
    
          
            
        
    # def _query_df(self,new_df,query_name):
    # # =============================================================================
    # #         
    # #         #   query of AND's - input a list of tuples.  ["AND",(field_name1,value1) and (field_name2,value2) and ...]
    # #             the first element is the type of query  -"&"-AND, "|"-OR, "!"-NOT, "B"-between
    # # #            return a slice of the df as a copy
    # # # 
    # # #        a query of OR 's  -  input a list of tuples.  ["OR",(field_name1,value1) or (field_name2,value2) or ...]
    # # #            return a slice of the df as a copy
    # # #
    # # #        a query_between is only a triple tuple  ["BD",(fieldname,startvalue,endvalue)]
    # #                "BD" for between dates, "B" for between numbers or strings
    # # # 
    # # #        a query_not is only a single triple tuple ["NOT",(fieldname,value)]   
    # # 
    # #         
    # # =========================================================================
      
    
   # @tf.function
    def _create_series(self,query_df,dollars):
      #  print("query_df=\n",query_df)
      #  print(query_df['qty'])  #.numpy())
       # df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        df=query_df.resample('D').sum().round(0).copy()
        df=df.rolling(dd2.dash2_dict['sales']['predictions']['mat']).mean()
      #  print("resampled df=\n",df)
      
      #  divide by 1000 so as not to overfloww on tf.int32
      
        if dollars:
            q=df.iloc[dd2.dash2_dict['sales']['predictions']['mat']:]['salesval'].to_numpy()[tf.newaxis,...]/1000   #,tf.newaxis]       
        else:    
            q=df.iloc[dd2.dash2_dict['sales']['predictions']['mat']:]['qty'].to_numpy()[tf.newaxis,...]/1000  #,tf.newaxis] 
           
      #  q=q.astype(tf.float32)   
      #  print("q=\n",q)
        return tf.convert_to_tensor(q,tf.float32),df.iloc[dd2.dash2_dict['sales']['predictions']['mat']:].index 
        
    
    
    def actual_vs_expected(self,query_dict,output_dir):
        start_date = pd.to_datetime(dd2.dash2_dict['sales']['predictions']['data_start_date']) + pd.DateOffset(days=dd2.dash2_dict['sales']['predictions']['start_point'])
        end_date = pd.to_datetime(dd2.dash2_dict['sales']['predictions']['data_start_date']) + pd.DateOffset(days=dd2.dash2_dict['sales']['predictions']['end_point'])
 
        query_count=1
        #total_queries=1
        total_queries=len(query_dict.keys())
    
        for query_name,v in query_dict.items():
            query_df=v.copy()      #query_dict[query_name]
                       #    print("query aname",query_name,"query_df=\n",query_df)
            print("\nQuery name to train on:",query_name,": (",query_count,"/",total_queries,")\n")  #," : ",plot_dict[k][0,-10:])
            actuals,dates=self._create_series(query_df,dollars=dd2.dash2_dict['sales']['predictions']['dollars'])
            assert not np.any(np.isnan(actuals))
            tf.print(query_name,"actuals=\n",actuals,actuals.shape)
           # print(query_name,query_df.shape)
            if actuals.shape[1]>dd2.dash2_dict['sales']['predictions']['minimum_length']:

                #   print("query",query_name)
            #    print("dates=\n",dates,len(dates))
               
                
   ####################################################################       
            
   
                
                X,Y=self.create_X_and_Y_batches(actuals,dd2.dash2_dict['sales']['predictions']['batch_length'],dd2.dash2_dict['sales']['predictions']['no_of_batches'])
               # X,Y=st.create_X_and_Y_batches(plot_dict[k],st.batch_length,st.no_of_batches)
                dataset=tf.data.Dataset.from_tensor_slices((X,Y)).cache().repeat(dd2.dash2_dict['sales']['predictions']['no_of_repeats'])
                dataset=dataset.shuffle(buffer_size=dd2.dash2_dict['sales']['predictions']['no_of_batches']+1,seed=42)      
                
                train_set=dataset.batch(1).prefetch(1)
                valid_set=dataset.batch(1).prefetch(1)
           
            
    ######################################################################       
            
            
                model=self.model_training_GRU(train_set,valid_set,query_name,output_dir)
                new_query_name=query_name+str("_GRU")
                 
                print("\nGRU Predicting....",new_query_name)
                actuals=actuals[...,tf.newaxis]
             #   tf.print("series to predict on:",series,series.shape)
     #           new_prediction,new_stddev=st.predict_series(model,plot_dict[k][:,st.start_point:st.end_point+1][...,tf.newaxis])
              #  new_prediction,new_stddev=st.predict_series(model,plot_dict[k][...,tf.newaxis])
                new_prediction,new_stddev=self.simple_predict(model,actuals)
                
                
    ###############################################################################            
                
                actuals=actuals.numpy()
            #    print("before new predictopn=",new_prediction,new_prediction.shape)
      
                new_prediction=new_prediction[0].reshape(1,-1)[0].astype(np.float32)
                new_stddev=new_stddev[0].reshape(1,-1)[0].astype(np.float32)
                actuals=actuals[0].reshape(1,-1)[0].astype(np.float32)
               
              #  print("1new predictopn=",new_prediction,new_prediction.shape)
              #  print("1new stddev=",new_stddev,new_stddev.shape)
             
                
                extra_date_list=pd.date_range(dates[-1], periods=dd2.dash2_dict['sales']['predictions']['predict_ahead_length'],closed='right')
            #    extra_date_list = [(start_date + dt.timedelta(days = day)).isoformat().strftime("%y-%m-%d") for day in range(365)]
                dates=np.concatenate((dates,extra_date_list))
              #  print("before dates=",dates,dates.shape)
                padding= len(dates)-len(new_prediction)
                the_rest=len(dates)-len(actuals)
                
             #   dates=dates[:(dd2.dash2_dict['sales']['predictions']['end_point']+dd2.dash2_dict['sales']['predictions']['predict_ahead_length'])]

                #  print("padding",padding,len(dates))
                actuals=np.concatenate((actuals,np.zeros(the_rest,dtype=np.float32)))
                new_prediction=np.concatenate((np.zeros(padding,dtype=np.float32),new_prediction))
                new_stddev=np.concatenate((np.zeros(padding,dtype=np.float32),new_stddev))
              #  series=np.concatenate((np.zeros(padding),series),axis=0)
     
                
              #  print("2new predictopn=",new_prediction,new_prediction.shape)
              #  print("2new stddev=",new_stddev,new_stddev.shape)
              #  print("actuals=",actuals,actuals.shape)
              #  print("after dates=",dates,dates.shape)
              #  print("extra date list=",extra_date_list,len(extra_date_list))
              #  dates=dates[st.mat:]
              #  print("after dates=",dates,dates.shape)
    
               # print("new stddev=",new_stddev,new_stddev.shape)
                new_plot_df=pd.DataFrame({"actual":actuals,"expected":new_prediction,"expected_stddev":new_stddev},index=dates,dtype=np.int32)
                display_df=1000*new_plot_df.copy()
                sheet_name = 'Sheet1'
                if dd2.dash2_dict['sales']['predictions']['dollars']: 
                    name="dollars"
                else:
                    name="units"
                writer = pd.ExcelWriter(output_dir+query_name+"_"+name+"_"+dd2.dash2_dict['sales']['predictions']['avepredfile'],engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
                
                display_df.to_excel(writer, sheet_name=sheet_name)
                writer.save()
                
                new_plot_df.replace(0,np.nan,inplace=True)
             #   print("new_plot_df=\n",new_plot_df)
                ax=new_plot_df[new_plot_df.columns[0]].plot(style='g-',grid=True,lw=1,fontsize=8,title="["+query_name+"] "+name+" sales by day")
                new_plot_df[new_plot_df.columns[1]].plot(style='b:',yerr=new_plot_df[new_plot_df.columns[2]],lw=1,elinewidth=1, ecolor=['red'],errorevery=10)
 
                ax.legend(title="",fontsize=7)
                ax.set_xlabel("")
                if dd2.dash2_dict['sales']['predictions']['dollars']: 
                    ax.set_ylabel("$('000) per day",fontsize=8)
                else:
                    ax.set_ylabel("('000) units per day",fontsize=8)

                ax.axvline(pd.to_datetime(start_date), color='k', linestyle='--')
                ax.axvline(pd.to_datetime(end_date), color='k', linestyle='--')
               # ax.xaxis.set_label_text('date')
                ax.xaxis.label.set_visible(False)

          #      print("predict ahead=",predict_ahead)
                #st.plot_new_plot_df(new_prediction,latest_date)
                if dd2.dash2_dict['sales']['predictions']['dollars']: 
                    self._save_fig("actual_vs_expected_"+query_name+"_dollars",output_dir)   #dd2.dash2_dict['sales']['output_dir'])
                else:    
                    self._save_fig("actual_vs_expected_"+query_name+"_units",output_dir)   #dd2.dash2_dict['sales']['output_dir'])

        
              #  plt.show()
                plt.close()
              
                tf.keras.backend.clear_session()
            else:
                print(query_name,actuals.shape[1],"days. Query needs to have more than 731 days of records to predict into the future.") 
     
            query_count+=1

        plt.close('all')
        print("\n\nFinished.")
       
            
            
        return
    
    
             
              
          
          

