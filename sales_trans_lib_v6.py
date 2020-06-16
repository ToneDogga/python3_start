#
# Common imports
import numpy as np
import os
from pathlib import Path
import pandas as pd
#import random
import datetime as dt
import gc
from numba import cuda

import random
import csv
import pickle
from natsort import natsorted
from pickle import dump,load

from datetime import date
from datetime import timedelta

from collections import defaultdict
from datetime import datetime
#import SCBS0 as c

import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

from scipy.ndimage.filters import uniform_filter1d  # for running means

import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

#tf.config.experimental_run_functions_eagerly(False)   #True)
#tf.config.experimental_run_functions_eagerly(True)

from tensorflow import keras
# assert tf.__version__ >= "2.0"


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors
#print("eager exec:",tf.executing_eagerly())
# tf.random.set_seed(42)
# series=tf.constant(tf.random.uniform(shape=[3,826], maxval=100, dtype=tf.int32, seed=10))

# mat_days=28  #tf.constant(28,dtype=tf.int32)
# print("series=\n",series)

# output_2d=mat_add_2d(series,mat_days)
# print("output",output_2d)

#s
  

class salestrans:
    def __init__(self):   
        self.epochs=12
    #    self.steps_per_epoch=100 
        self.no_of_batches=1000
        self.no_of_repeats=2
        
        self.dropout_rate=0.2
        self.start_point=0
        self.end_point=732
        self.predict_ahead_length=500
        self.batch_length=365
        self.predict_length=365
        self.one_year=366
        self.batch_jump=self.batch_length  # predict ahead steps in days. This is different to the batch length  #int(round(self.batch_length/2,0))
        self.neurons=1000   #self.batch_length
        
        self.data_start_date="02/02/18"
        self.date_len=1300      
        self.dates = pd.period_range(self.data_start_date, periods=self.date_len)   # 2000 days


        self.pred_error_sample_size=12
        self.no_of_stddevs_on_error_bars=1
        self.patience=5

        self.mats=[28]   #moving average in days #tf.constant([28],dtype=tf.int32)
            
        self.plot_dict_types=dict({0:"raw_database_or_model",1:"raw_query",2:"moving_total",3:"prediction",4:"stddev"})
        
        self.train_percent=0.7
        self.valid_percent=0.2
        self.test_percent=0.1
        
        self.filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]
        self.queryfilename="queryfile.xlsx"
        self.plot_dict_filename="plot_dict.pkl"
        
             
        self.output_dir = self.log_dir("SCBS2")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.images_path = os.path.join(self.output_dir, "images/")
        os.makedirs(self.images_path, exist_ok=True)
     
        self.__version__="0.6.0"
        return
    

      
    
    def save_fig(self,fig_id, images_path, tight_layout=True, fig_extension="png", resolution=300):
        path = os.path.join(images_path, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
    
      


    def load_sales(self,filenames):  # filenames is a list of xlsx files to load and sort by date
        print("load:",filenames[0])
        df=pd.read_excel(filenames[0],sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   
        print("df size=",df.shape,df.columns)
        for filename in filenames[1:]:
            print("load:",filename)
            new_df=pd.read_excel(filename,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows  
            new_df['date'] = pd.to_datetime(new_df.date)  #, format='%d%m%Y')
         #   new_df=new_df.set_index('date')
        #    print("cols:",new_df.columns)
       #     print(new_df.head(5))
       #     print(new_df.tail(5))
            print("appending",filename,":size=",new_df.shape)
            df=df.append(new_df)
            print("appended df size=",df.shape)
        
        
        df.fillna(0,inplace=True)
        
        #print(df)
        print("drop duplicates")
        df.drop_duplicates(keep='first', inplace=True)
        print("after drop duplicates df size=",df.shape)
        print("sort by date",df.shape[0],"records.\n")
        df.sort_values(by=['date'], inplace=True, ascending=False)
          
     #   print(df.head(5))
     #   print(df.tail(5))
   
 
        df["period"]=df.date.dt.to_period('D')
        df['period'] = df['period'].astype('category')
        
      #  df.sort_index('date',axis=0,ascending=True,inplace=True)    
     #   print("period=\n",df.head(5))
     #   print(df.tail(5))
   
 
        return df           
 
    
    
    def save_plot_dict(self,plot_dict,savename):
        with open(savename,"wb") as f:
            pickle.dump(plot_dict, f,protocol=-1)
   #     print("plot dict saved to",savename)    
        
        
 
    def load_plot_dict(self,loadname):
        with open(loadname,"rb") as f:
            return pickle.load(f)
  
   
    def query_sales(self,sales_df,queryfilename,plot_dict):  
        print("query sales:",queryfilename)
        
        #        sales_df=[plot_dict[k] for k in plot_dict.keys()][0] 
     #   print("loading query list '",queryfilename,"'")
        query_dict = pd.read_excel(queryfilename, index_col=0, header=0,  skiprows=0).T.to_dict()  #  doublequote=False
        
     #   print("\nquery dict=\n",query_dict,"\n")
        
        table_list = defaultdict(list)
        q=0
        
        for query_name in query_dict.keys():
             table_list[q].append(query_name)
             table_list[q].append(pd.pivot_table(sales_df.query(self.quotes(query_dict[query_name]['condition'])), values='qty', index=query_dict[query_name]['index_code'].split(),columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T) 
             q+=1
            
        table_dict = dict((k, tuple(v)) for k, v in table_list.items())  #.iteritems())
        del table_list
        
   #     print("\n ttable dict keys=\n",table_dict.keys())
  #      print("\n ttable dict items=\n",table_dict.items())
 
        plot_number=100 
        
        for k in table_dict.keys():
                self.plot_query(table_dict[k][1],table_dict[k][0])  # 1].copy(),table_dict[k][0].copy(deep=True))
                query_values=table_dict[k][1].to_numpy().swapaxes(0,1)
            #    print("query values=",query_values.shape)
                querycount=0
                if query_values.shape[0]>1:
                    *querysplit,=query_values
                    for queryv in querysplit:             
                        key=tuple([table_dict[k][0]+str(table_dict[k][1].T.index.values[querycount]),1,self.start_point,plot_number+querycount])   # actuals
                        querycount+=1
                        plot_dict[key]=queryv[np.newaxis,...]    #.swapaxes(0,1)
                     #   plot_dict[key]=nptd   #[:,self.start_point:self.end_point+1]  #tf_value  # 2D only tensor shape [1,series]
                else:         
                    key=tuple([table_dict[k][0],1,self.start_point,plot_number+querycount])   # actuals        
                    #nptd=table_dict[k][1].to_numpy()     #.swapaxes(0,1)
                    querycount+=1
                  #  print("query values=",query_values,query_values.shape)
                    plot_dict[key]=query_values   #nptd   #[:,self.start_point:self.end_point+1]  #tf_value  # 2D only tensor shape [1,series]
                    
                        
           #     print("query sales -after length of new series",plot_dict[key].shape)
                plot_number=plot_number+querycount+10

                
             
    #    print("plot dict keys proit to mats",plot_dict.keys())        
         # create mat  s 
        for key in plot_dict.copy():
            if key[3]>0:
                plot_number=key[3]
                for elem in self.mats:   
                    mat_key=tuple([key[0]+"@"+str(elem)+"u:mt",2,self.start_point,plot_number])
            #    mat_value=self.mat_add_1d(tf.transpose(tf_value, [1, 0]),self.mats) 
                    tf_value=tf.convert_to_tensor(plot_dict[key],tf.int32)
                    mat_value=self.mat_add_1d(tf_value,elem) 
    
                    plot_dict[mat_key]=mat_value   #.numpy()
                    plot_number+=1
           #   plot_number+=1
        del table_dict     
        return plot_dict
 
     
    
 #   @tf.function
  #   def mat_add_1d(self,series,mat_days):
  #       weight_1d = tf.ones(mat_days,tf.int32)
  #       strides_1d = 1   #tf.constant(1,dtype=tf.int32)
        
  #    #   print("mat add 1 d series shape",series.shape)
  #       in_1d = series #tf.constant(series, dtype=tf.int32)
        
  #       #in_1d = tf.constant(ones_1d, dtype=tf.float32)
  #       filter_1d = tf.constant(weight_1d, dtype=tf.int32)
        
  #       in_width = in_1d.shape[1]
  #       filter_width = filter_1d.shape[0]
        
  #       input_1d   = tf.reshape(in_1d, [1, in_width, 1])
  #       kernel_1d = tf.reshape(filter_1d, [filter_width, 1, 1])
  #       output_1d = tf.cast(tf.divide(tf.squeeze(tf.nn.conv1d(input_1d, kernel_1d, strides_1d, padding='SAME')),mat_days),dtype=tf.int32)
  # #      output_1d = tf.cast(tf.divide(tf.squeeze(tf.nn.conv1d(input_1d, kernel_1d, strides_1d, padding='VALID')),mat_days),dtype=tf.int32)

  #       return output_1d[tf.newaxis,...]
  
    
  #  @tf.function
    def mat_add_1d(self,series,mat_days):  
     #   print("mat add 1d series.shape",series.shape)
        pds=pd.DataFrame(series[0])
      #  print("pds=",pds,pds.shape)
        nds=pds.rolling(mat_days).mean()
        fill_val=nds.iloc[mat_days+1]  #.to_numpy()
      #  print("fill val",fill_val)
 
      #  print("nds=",nds.shape)
      #  query_df=query_df.replace(np.nan,fill_val)
        nds.fillna(fill_val,inplace=True)
      #  print("nds",nds.head(40))
        return nds.to_numpy().swapaxes(0,1)

      #  print("new nds=",nds.shape)
        
      #  return nds

        # mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
#    The mode parameter determines how the array borders are handled, where cval is the value when mode is equal to ‘constant’. Default is ‘reflect’
     #   return(uniform_filter1d(series[0], size=mat_days,mode='nearest')[np.newaxis,...])
  
  
  
    def plot_query(self, query_df_from_dict,query_name):
        query_df=query_df_from_dict.copy(deep=True)
        query_details=str(query_df.columns[0])
        query_df.columns = query_df.columns.get_level_values(0)
    #    print("query df shae",query_df.shape)
     #   query_df=self.mat_add_1d(query_df.to_numpy().swapaxes(0,1),self.mats[0])
        query_df['qdate']=query_df.index.copy(deep=True)  #.to_timestamp(freq="D",how='s') #
#        query_df['qdate']=pd.to_datetime(pd.Series(query_list).to_timestamp(freq="D",how='s'), format='%Y/%m/%d')
       # print("query list",query_list)
        query_df['qdate'].apply(lambda x : x.to_timestamp())
    #    query_df['qdate']=query_list.to_timestamp(freq="D",how='s')
        query_list=query_df['qdate'].tolist()
      #  print("qudf=\n",query_df,query_df.columns[1][0])
    #    print("f",query_list)
        #   query_df['qdate'] = query_df.qdate.tolist()
     #   print("query_df=",query_df)
        query_df=query_df.rolling(self.mats[0]).mean()
        fill_val=query_df.iloc[self.mats[0]+1,0]  #.to_numpy()
    #    print("fill val",fill_val)

        query_df=query_df.fillna(fill_val)
       # query_df.reset_index()   #['qdate']).sort_index()
     #   query_df.reset_index(level='specialpricecat')
        query_df.reset_index(drop=True, inplace=True)
        query_df['qdate']=query_list   #.set_index(['qdate',''])
     #   print("query df=\n",query_df)
      #  query_df=query_df.replace(0, np.nan)
   #     ax=query_df.plot(y=query_df.columns[0],style="b-")   # actual
     #   ax=query_df.plot(x=query_df.columns[1][0],style="b-")   #,use_index=False)   # actual
        ax=query_df.plot(x='qdate',y=query_df.columns[0],style="b-")   #,use_index=False)   # actual

      #  col_no=1
     #   query.plot(style='b-')
     #   ax.axvline(pd.to_datetime(start_date), color='k', linestyle='--')
     #   ax.axvline(pd.to_datetime(end_date), color='k', linestyle='--')

        plt.title("Unit sales:"+query_name+query_details,fontsize=10)   #str(new_plot_df.columns.get_level_values(0)))
     #   plt.legend(fontsize=8)
        plt.ylabel("units/day sales")
        plt.grid(True)
        self.save_fig("actual_"+query_name+query_details,self.images_path)
        plt.show()
    
    
 # #   @tf.function
 #    def mat_add_2d(self,series,mat_days):
        
 #        weight_2d = np.ones((1,mat_days))
 #        strides_2d = [1, 1, 1, 1]
        
 #        in_2d = series #tf.constant(series, dtype=tf.float32)
 #        filter_2d = tf.constant(weight_2d, dtype=tf.int32)
        
 #        in_width = int(in_2d.shape[1])
 #        in_height = int(in_2d.shape[0])
        
 #        filter_width = int(filter_2d.shape[1])
 #        filter_height = int(filter_2d.shape[0])
        
 #        input_2d   = tf.reshape(in_2d, [1, in_height, in_width, 1])
 #        kernel_2d = tf.reshape(filter_2d, [filter_height, filter_width, 1, 1])
    
 #        output_2d = tf.cast(tf.divide(tf.squeeze(tf.nn.conv2d(input_2d, kernel_2d, strides=strides_2d, padding='SAME')),mat_days),dtype=tf.float32)
 #        return output_2d[tf.newaxis,...]
    



     
      
    
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
 
    


   # @tf.function
    def create_X_and_Y_batches(self,series,batch_length,no_of_batches):
        X_indices,full_indices=self.create_X_batches(series,batch_length,no_of_batches,self.start_point,self.end_point)
     #   print("X indices shape",X_indices.shape)
     #   print("full indices shape",full_indices.shape)
       
        
        batch_depth=X_indices.shape[1]
        Y_indices = np.empty((no_of_batches,batch_depth,batch_length),dtype=np.int32)
        
        for step_ahead in range(1, batch_depth + 1):
            Y_indices[:,:, step_ahead-1] = full_indices[:, step_ahead:step_ahead + batch_length,0]
     #   test_Y_indices=tf.gather(Y_indices[:,:, 0],full_indices[:, :batch_length,0],axis=1)
      #  print("X, X.shape",X_indices,X_indices.shape)  
      #  print("Y, Y.shape",Y_indices,Y_indices.shape)
                
        X=tf.cast(tf.gather(series[0],X_indices,axis=0),tf.int32)
        Y=tf.cast(tf.gather(series[0],Y_indices,axis=0),tf.int32)

    #    tf.print("2X[1]=",X[1],X.shape,"\n")
    #    tf.print("2Y[1]=",Y[1],Y.shape,"\n")

        return X,Y
  

    
    @tf.autograph.experimental.do_not_convert
    def model_training_GRU(self,train_set,valid_set,query_name):
        print("\nTraining with GRU and dropout")
        model = keras.models.Sequential([
      #     keras.layers.Conv1D(filters=st.batch_length,kernel_size=4, strides=1, padding='same', input_shape=[None, 1]),  #st.batch_length]), 
      #     keras.layers.BatchNormalization(),
           keras.layers.GRU(self.neurons, return_sequences=True, input_shape=[None, 1]), #st.batch_length]),
           keras.layers.BatchNormalization(),
           keras.layers.GRU(self.neurons, return_sequences=True),
           keras.layers.AlphaDropout(rate=self.dropout_rate),
           keras.layers.BatchNormalization(),
           keras.layers.TimeDistributed(keras.layers.Dense(self.batch_length))
        ])
    
        model.compile(loss="mse", optimizer="adam", metrics=[self.last_time_step_mse])
       
        model.summary() 
       
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),self.MyCustomCallback()]
       
        history = model.fit(train_set ,epochs=self.epochs,
                           validation_data=(valid_set), callbacks=callbacks)
            
        print("\nsave model",query_name,":GRU_Dropout_sales_predict_model.h5\n")
        model.save(self.output_dir+query_name+":GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
             
        self.plot_learning_curves(history.history["loss"], history.history["val_loss"],self.epochs,"GRU and dropout:"+str(query_name))
        self.save_fig("GRU and dropout learning curve_"+query_name,self.images_path)
    
        plt.show()
        return model    
        
 
        
 
    
 
 
    
  #  @tf.autograph.experimental.do_not_convert
    def model_training_wavenet(self,train_set,valid_set,query_name):
        print("\nTraining with Wavenet")
      
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=[None, 1]))
        for rate in (1, 2, 4, 8) * 2:
            model.add(keras.layers.Conv1D(filters=2*self.batch_length, kernel_size=2, padding="causal",
                                          activation="relu", dilation_rate=rate))
            if rate==8:
                model.add(keras.layers.AlphaDropout(rate=self.dropout_rate))
        model.add(keras.layers.Conv1D(filters=self.batch_length, kernel_size=1))
        model.compile(loss="mse", optimizer="adam", metrics=[self.last_time_step_mse])
 
        model.summary()   
 
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),self.MyCustomCallback()]
       
        history = model.fit(train_set ,epochs=self.epochs,
                           validation_data=(valid_set), callbacks=callbacks)
            
        model.compile(loss="mse", optimizer="adam", metrics=[self.last_time_step_mse])
       

#            history = model.fit(train_set,  steps_per_epoch=st.steps_per_epoch ,epochs=st.epochs,
#                               validation_data=(valid_set))
  
    #      history = model.fit_generator(X_train, Y_train, epochs=st.epochs,
  #                         validation_data=(X_valid, Y_valid))
       
       
 #       print("\nsave model\n")
        model.save(self.output_dir+query_name+":wavenet_sales_predict_model.h5", include_optimizer=True)
          

       
        self.plot_learning_curves(history.history["loss"], history.history["val_loss"],self.epochs,"Wavenet:"+str(query_name))
        self.save_fig("Wavenet learning curve_"+query_name,self.images_path)
    
        plt.show()
        return model    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
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
    
    
    def last_time_step_mse(self,Y_true, Y_pred):
        return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
    
    
       
     
    def plot_learning_curves(self,loss, val_loss,epochs,title):
        if ((np.min(loss)<=0) or (np.max(loss)==np.inf)):
            return
        if ((np.min(val_loss)<=0) or (np.max(val_loss)==np.inf)):
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
    


    
  
            
        
     
    def quotes(self,test2):
        test2=str(test2)
        quotescount=0
        k=0
        testlen=len(test2)
        while k< testlen: 
        #    print("k=",k,"testlen",testlen)
            if (test2[k].isupper() or test2[k].isnumeric()) and quotescount%2==0:  # even
                test2=test2[:k]+'"'+test2[k:]
                testlen+=1
                k+=1
                quotescount=+1
            
            # closing quotes
            if (test2[k]==" " and quotescount%2==1):
                    test2=test2[:k]+'"'+test2[k:]
                    testlen+=1
                    k+=1
                    quotescount+=1
            k+=1
            
        if quotescount%2==1:  # odd
            test2=test2+'\"'
    
        
        return(test2)       


    
    
    def log_dir(self,prefix=""):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "./SCBS2_outputs"
        if prefix:
            prefix += "-"
        name = prefix + "run-" + now
        return "{}/{}/".format(root_logdir, name)
    
  
    
  
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
 
        for r in range(0,self.predict_ahead_length,self.batch_jump):           
            X_new=series[:,self.end_point-self.predict_length:self.end_point,:]
       #     print("\nsimple predict - X new",X_new,X_new.shape,"\n\n")
       #     print("prediction line[0,-1]",model(X_new,training=False)[0,-1],model(X_new,training=False)[0,-1].shape)
       #     print("prediction line[0,batch_length-1]",model(X_new,training=False)[0,self.batch_length-1],model(X_new,training=False)[0,self.batch_length-1].shape)

       #     print("prediction line[0,:,-1]",model(X_new,training=False)[0,:,-1],model(X_new,training=False)[0,:,-1].shape)

        #model(X_new,training=True)[0,-1]
       
            Y_probs=np.stack([model(X_new,training=True)[0,-1].numpy() for sample in range(self.pred_error_sample_size)])         
    
            Y_mean=Y_probs.mean(axis=0)
            Y_mean=Y_mean[np.newaxis,...,np.newaxis]
            
            Y_stddev=self.no_of_stddevs_on_error_bars*Y_probs.std(axis=0)   #[np.newaxis,...]
            Y_stddev=Y_stddev[np.newaxis,...,np.newaxis]
    
            new_prediction=np.concatenate((new_prediction,Y_mean[:,:self.batch_jump]),axis=1) 
            new_stddev=np.concatenate((new_stddev,Y_stddev[:,:self.batch_jump]),axis=1) 
            series=np.concatenate((series,Y_mean[:,:self.batch_jump]),axis=1) 

        return new_prediction ,new_stddev


   #     return Y_mean,Y_stddev
   #     return model.predict(X_new)[0,-1][tf.newaxis,...,tf.newaxis]
        


# # @tf.function
#     def predict_series(self,model,series):
#         input_series=model(tf.cast(series[:,self.start_point:self.end_point+1,:],tf.float32))
#         new_prediction=np.empty((1,0,1)) #self.batch_length))
#         new_stddev=np.empty((1,0,1),dtype=np.float32)  #self.batch_length),dtype=np.float32)

#         for r in range(0,self.predict_ahead_length,self.batch_jump):        
#             # multiple more accurate prediction
#             predict_series=input_series[:,-self.one_year:self.batch_jump-self.one_year,:]             
#             Y_probs=np.stack([model(predict_series,training=True)[0,-1].numpy() for sample in range(self.pred_error_sample_size)])         
 
#         #    input_series=input_series[-(self.one_year+self.batch_length):]
#        #     print("Y_probs shape=\n",Y_probs,Y_probs.shape)
#          #   print("Y_probs2=\n",Y_probs2,Y_probs2.shape)
 
#             Y_mean=Y_probs.mean(axis=0)   #[np.newaxis,...]
#       #      print("Y mean shape 0",Y_mean.shape)
#             Y_mean=Y_mean[np.newaxis,...]
#             Y_mean=Y_mean[...,np.newaxis]
#       #      print("1Y mean",Y_mean, Y_mean.shape)
             
#             Y_stddev=self.no_of_stddevs_on_error_bars*Y_probs.std(axis=0)   #[np.newaxis,...]
#             Y_stddev=Y_stddev[np.newaxis,...]
#             Y_stddev=Y_stddev[...,np.newaxis]
     
#             new_prediction=np.concatenate((new_prediction,Y_mean),axis=1) 
#             new_stddev=np.concatenate((new_stddev,Y_stddev),axis=1) 
        
#             input_series=np.concatenate((input_series[:,self.batch_jump:,-1:],Y_mean),axis=1)
#       #      input_series=input_series[:,self.batch_jump:,-1:]

#          #   print("3input series shape after concat with Y_mean",input_series.shape)

#           #  print("mafter predict ahead=",predict_ahead.shape)
#             print("prediction shape @",r," day ahead=",new_prediction.shape)    
#         return new_prediction ,new_stddev
  


       
    # =============================================================================
    #  
    # 
    # # the plot dictionary is the holding area of all data
    # # it has a 4-tuple for a key
    # 
    # first is query name
    # second is 0= original data, 1 = actual query don't predict or  plot, 2 = plot actual, 3 = plot prediction, 4 = plot prediction with error bar
    # third is the start point of the plot and alos the finish point of the prediction
    # fourth is the plot number  
    #
    # the value is a 1D Tensor except at the start where sales_df is a pandas dataframe
    #     
    # =============================================================================
          



          
    def remove_key_from_dict(self,plot_dict,key):
      #  print("1plot_dict len",len(plot_dict))
     #   plot_dict.pop(key, None)
        del plot_dict[key]
   #     print("2plot_dict len",len(plot_dict))

        return plot_dict
    
   



    #@tf.function
    def append_plot_dict(self,plot_dict,query_name,new_prediction,new_stddev,plot_number):
        
   #     new_key="('"+str(query_name)+"_prediction', 2, "+str(self.end_point)+")"
        new_key=tuple([str(query_name)+"_prediction",3,self.end_point,plot_number])

   #     print("append plot dict new key=",new_key)
        new_prediction=self.add_start_point(new_prediction,self.end_point)
        plot_dict[new_key]=new_prediction[:,:,0]
    
        
 # error bar
        new_key=tuple([str(query_name)+"_prediction_stddev",4,self.end_point,plot_number])

    #     print("append plot dict new key=",new_key)
        new_stddev=self.add_start_point(new_stddev,self.end_point)
        plot_dict[new_key]=new_stddev[:,:,0]
        
        
        
        return plot_dict
        
    def tidy_up_plot_dict(self,plot_dict):
        # al the arrays are in the shape of [1,len]
        # they need to be 1D, shape [len] to plot directly
        # all the arrays need to be the same length
        # change keys in plot dict to be the name which is the first element in the key tuple
        for key in plot_dict.copy():
            
            plot_dict[key]=self.add_trailing_blanks(plot_dict[key],self.date_len)
            plot_dict[key]=plot_dict[key][0,:self.date_len]
            #key=key[0] # name only as key
         #   plot_dict[key[0]] = plot_dict.pop(key)
        return plot_dict
        
        
    def build_final_plot_df(self,plot_dict):
      #  final_plot_df=pd.DataFrame(columns=plot_dict.keys(),index=self.dates)
        plot_dict=self.tidy_up_plot_dict(plot_dict)  
     #   no_of_plots=0
        new_column_names=[]
        for key in plot_dict.copy():
    #        print("final series",key,plot_dict[key].shape)
           #      for key in plot_dict.copy():
            
           # plot_dict[key]=self.add_trailing_blanks(plot_dict[key],self.date_len)
           # plot_dict[key]=plot_dict[key][0,:self.date_len]
            new_column_names.append(key[0]) # name only as key
 
    
      #      no_of_plots+=1
        final_plot_df=pd.DataFrame.from_dict(plot_dict,orient="columns")
        final_plot_df.index=self.dates
    #    print("fpdf=",final_plot_df)
 #        for key in plot_dict.keys():
        return final_plot_df,new_column_names   #,no_of_plots  
 
    
    def simplify_col_names(self,plot_df,new_column_names): 
        plot_df.columns = plot_df.columns.droplevel(2)
        plot_df.columns = plot_df.columns.droplevel(2)
    # keep plot type 

    #    print("new col names",new_column_names)
        col_names=plot_df.columns
     #   plot_df = plot_df.T
 
        
#   plot_df=plot_df.reset_index(level=0)
      #  plot_df=plot_df.T
        rename_dict=dict(zip(col_names, new_column_names))
     #   print("rename dict",rename_dict)
        
        #  #   flat_column_names = [a_tuple[0][level] for a_tuple in np.shape(cols[level])[1] for level in np.shape(cols)[0]]
        #   #  print("fcn=",flat_column_names)
        plot_df=plot_df.rename(rename_dict, axis='columns')  #,inplace=True)
    #    print("simplifyed plot df =\n",plot_df)
        return plot_df
  
   
    def add_start_point(self,series,start_point):
       # print("add offset series 3.shape=",series.shape)
      #  offset=total_len-series.shape[1]
        a = np.empty((1,start_point,1)) #self.batch_length))
        a[:] = np.nan
        return np.concatenate((a,series),axis=1) 
     #                    new_plot_df[plot_name]=pred_plot_data
  
    
    def add_trailing_blanks(self,series,total_len):
    #    print("add offset 2 series.shape=",series.shape)
        offset=total_len-series.shape[1]
        if offset>=1:
            a = np.empty((1,offset))
            a[:] = np.nan
            return np.concatenate((series,a),axis=1) 
        else:
            return series[:,:total_len]
    
    
    def return_plot_number_df(self,plot_df,plot_number):
    #    plot_df=plot_df.T
        plot_df_slice=plot_df.xs(plot_number, axis='columns', level=3)
    #    print("plot df slice=",plot_df_slice)
        if not plot_df_slice.empty:
            return plot_df_slice
        else:
            return None


    def clean_up_col_names(self,df):
        for col in df.columns:
            newcol = col[0].replace(',', '_')
            newcol = newcol.replace("'", "")
  #          newcol = newcol.replace(" ", "")
        df=df.rename(columns={col[0]:newcol})   #, inplace=True)
        return df



    
 
    def plot_new_plot_df(self,new_plot_df):
    #    new_plot_df=self.build_final_plot_df(plot_dict)

    #    new_plot_df=pd.DataFrame(plot_dict,index=self.dates)
     #   print("newplot df=",new_plot_df)
        
        #  multiindex try querying
  #      print("0",new_plot_df.columns.get_level_values(0))
  #      print("1",new_plot_df.columns.get_level_values(1))
  #      print("2",new_plot_df.columns.get_level_values(2))
  #      print("level 3 multiindex ",new_plot_df.columns.get_level_values(3))
        start_date = pd.to_datetime("02/02/18") + pd.DateOffset(days=self.start_point)
        end_date = pd.to_datetime("02/02/18") + pd.DateOffset(days=self.end_point)
        
      #  end_date = pd.DateOffset("02/02/18", periods=self.end_point)   # 2000 days
        print("\nplot new df - start date",start_date,"end date",end_date)
 
      #  print("plot new plot df",new_plot_df)
        plot_nums=list(set(new_plot_df.columns.get_level_values(3)))
 
        print("plot new plot_df plot nums",plot_nums)
        new_plot_df=new_plot_df.reindex(new_plot_df.columns, axis=1,level=1)
        query_names=list(set(new_plot_df.columns.get_level_values(0)))
        print("query names=",query_names)
        # print("1new plot df sorted=\n",new_plot_df)
        query_number=0
        for plot_number in plot_nums:
            plot_number_df=self.return_plot_number_df(new_plot_df,plot_number)
       #     plot_number_df=plot_number_df.MultiIndex.sortlevel(level=1,ascending=True)
  
            
        #    print("2plot number df=\n",plot_number_df)
            if not plot_number_df.empty:
            #    plot_number_df.plot(yerr = "double_std")
         #       plot_number_df.plot()

            #    query_names=list(set(plot_number_df.columns.get_level_values(0)))[0]
                
                # df.mean1.plot()
                #df.mean2.plot(yerr=df.stddev)

                ax=plot_number_df[plot_number_df.columns[0]].plot(style="b")   # actual
                
                col_no=1
           #     prediction_names=list(set(plot_number_df.columns.get_level_values(1)))
           #     print("prediction namnes",prediction_names)
                #prediction + error bars
            #    for p in prediction_names:
                plot_number_df[plot_number_df.columns[col_no]].plot(yerr=plot_number_df[plot_number_df.columns[col_no+1]],style='r', ecolor=['red'],errorevery=10)
             #       col_no+=2    
                #           #      df[df.columns[2]].plot(yerr=df.stddev)
                # xposition = [pd.to_datetime('2010-01-01'), pd.to_datetime('2015-12-31')]
                # for xc in xposition:
                #     ax.axvline(x=xc, color='k', linestyle='-')

             #   ax = plot_number_df.plot()
                ax.axvline(pd.to_datetime(start_date), color='k', linestyle='--')
                ax.axvline(pd.to_datetime(end_date), color='k', linestyle='--')

                plt.title("Unit sales:"+str(plot_number_df.columns[0]),fontsize=10)   #str(new_plot_df.columns.get_level_values(0)))
                plt.legend(fontsize=8)
                plt.ylabel("units/day sales")
                plt.grid(True)
                self.save_fig("actual_v_prediction_"+str(plot_number_df.columns[0]),self.images_path)

                plt.show()
             
            query_number+=1    
        plt.close("all")

       
        
