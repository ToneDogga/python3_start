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
        self.epochs=2
        self.no_of_batches=100
        self.no_of_repeats=2
        

        self.dropout_rate=0.2
        self.start_point=32
        self.end_point=800
        self.predict_ahead_length=365
        self.batch_length=365
        self.neurons=self.batch_length
        
        self.date_len=1300      
        self.dates = pd.period_range("02/02/18", periods=self.date_len)   # 2000 days


        self.pred_error_sample_size=5
        self.patience=5

        self.mats=28   #tf.constant([28],dtype=tf.int32)
            
        self.train_percent=0.7
        self.valid_percent=0.2
        self.test_percent=0.1
        
        self.filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx"]
        self.queryfilename="queryfile.xlsx"
        self.plot_dict_filename="plot_dict.pkl"
        
             
        self.output_dir = self.log_dir("SCBS2")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.images_path = os.path.join(self.output_dir, "images/")
        os.makedirs(self.images_path, exist_ok=True)
     
        self.__version__="0.1.2"
        return
    

  



    def load_sales(self,filenames):  # filenames is a list of xlsx files to load and sort by date
        print("load:",filenames[0])
        df=pd.read_excel(filenames[0],-1)  # -1 means all rows   
        print("df size=",df.shape)
        for filename in filenames[1:]:
            print("load:",filename)
            new_df=pd.read_excel(filename,-1)  # -1 means all rows   
            print("new df size=",new_df.shape)
            df=df.append(new_df)
            print("df size=",df.shape)
        
        
        df.fillna(0,inplace=True)
        
        #print(df)
        print("drop duplicates")
        df.drop_duplicates(keep='first', inplace=True)
        print("after drop duplicates df size=",df.shape)
        
        df["period"]=df.date.dt.to_period('D')
        df['period'] = df['period'].astype('category')
        return df    
                
 
    
    
    def save_plot_dict(self,plot_dict,savename):
        with open(savename,"wb") as f:
            pickle.dump(plot_dict, f,protocol=-1)
        print("plot dict saved to",savename)    
        
        
 
    def load_plot_dict(self,loadname):
        with open(loadname,"rb") as f:
            return pickle.load(f)
  
            
    def remove_key_from_dict(self,plot_dict,key):
      #  print("1plot_dict len",len(plot_dict))
     #   plot_dict.pop(key, None)
        del plot_dict[key]
        print("2plot_dict len",len(plot_dict))

        return plot_dict
    
    
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
        
   #     print("\n ttable dict=\n",table_dict)
            
        for k in table_dict.keys():
              key=tuple([table_dict[k][0],1,self.start_point])
         #     print("key=",key)
              nptd=table_dict[k][1].to_numpy().swapaxes(0,1)
        #      print("nptd=",nptd,nptd.shape)
        #      if nptd.shape[1]==0:
        #          nptd=nptd[...,tf.newaxis]
                  
              tf_value=tf.convert_to_tensor(nptd,tf.int32)
        #      print("tf value=",tf_value)
              plot_dict[key]=nptd   #tf_value  # 1D only tensor
 
         # create mat   
           #   for elem in self.mats:   
              mat_key=tuple([table_dict[k][0]+"@"+str(self.mats)+"u:mt",2,self.start_point])
          #    mat_value=self.mat_add_1d(tf.transpose(tf_value, [1, 0]),self.mats) 
              mat_value=self.mat_add_1d(tf_value,self.mats) 

              plot_dict[mat_key]=mat_value.numpy()
           #   print("plot disyvc items",plot_dict[mat_key].items())
            # dataset=tf.data.Dataset.from_tensor_slices(tf_series).repeat(3)
              
        del table_dict     
        return plot_dict
 
     
    
 #   @tf.function
    def mat_add_1d(self,series,mat_days):
        weight_1d = tf.ones(mat_days,tf.int32)
        strides_1d = 1   #tf.constant(1,dtype=tf.int32)
        
     #   print("mat add 1 d series shape",series.shape)
        in_1d = series #tf.constant(series, dtype=tf.int32)
        
        #in_1d = tf.constant(ones_1d, dtype=tf.float32)
        filter_1d = tf.constant(weight_1d, dtype=tf.int32)
        
        in_width = in_1d.shape[1]
        filter_width = filter_1d.shape[0]
        
        input_1d   = tf.reshape(in_1d, [1, in_width, 1])
        kernel_1d = tf.reshape(filter_1d, [filter_width, 1, 1])
        output_1d = tf.cast(tf.divide(tf.squeeze(tf.nn.conv1d(input_1d, kernel_1d, strides_1d, padding='SAME')),mat_days),dtype=tf.int32)
        return output_1d[tf.newaxis,...]
    
    
  
    
    
    
 #   @tf.function
    def mat_add_2d(self,series,mat_days):
        
        weight_2d = np.ones((1,mat_days))
        strides_2d = [1, 1, 1, 1]
        
        in_2d = series #tf.constant(series, dtype=tf.float32)
        filter_2d = tf.constant(weight_2d, dtype=tf.int32)
        
        in_width = int(in_2d.shape[1])
        in_height = int(in_2d.shape[0])
        
        filter_width = int(filter_2d.shape[1])
        filter_height = int(filter_2d.shape[0])
        
        input_2d   = tf.reshape(in_2d, [1, in_height, in_width, 1])
        kernel_2d = tf.reshape(filter_2d, [filter_height, filter_width, 1, 1])
    
        output_2d = tf.cast(tf.divide(tf.squeeze(tf.nn.conv2d(input_2d, kernel_2d, strides=strides_2d, padding='SAME')),mat_days),dtype=tf.float32)
        return output_2d[tf.newaxis,...]
    



     
    
    
    @tf.function
    def build_all_possible_batches_from_series(self,series,batch_length):   #,start_point,end_point):
     
    #  we want to create an X from series of batch_length
    # for every combination
    # so series indexes are [[0,1,2,3,4,....,x]]
    # X combinations are [[0,1,2,..., batch_length]
    #        [1,2,3,....batch_length+1]
    #              ................
    #        [800,801,802,....,x]]
    #
    # this assumes x is 800+batch_length!
    #
    # so one batch of X is in the shape [1,batch_length,1]
    # and is randomly selected
    #
    # but the Y has to be in the shape
    #  [1,batch_length,batch_length]
    # but one step ahead of the X
    # so if X batch indicies of series are [[23,24,25,.....,39]]   (say batch length is 16)
    # Y is [[24,25,26..., 40]
    #       [25,26,27...., 41]
    #              ................
    #        [40,41,42,....,56]]
    #
    #
    # and the X and Y shape have to match in this setup
    # so X is also in shape [1,batch_length,batch_length]
    #
    # one X batch would have to be  
    # X is [[23,24,25..., 39]
    #       [24,25,26...., 40]
    #              ................
    #        [39,41,42,....,55]]
    #
    # to match the Y batch
    
    
        start_points=tf.range(0,self.batch_length,dtype=tf.int32)  
        grid_indices=(tf.meshgrid(start_points,tf.range(0,int((series.shape[1]-batch_length)+1),dtype=tf.int32)))   #int((end_point-start_point)/batch_length)+1))
        start_index=tf.convert_to_tensor(tf.add(grid_indices[0],grid_indices[1]))   #[:,:,np.newaxis
        return tf.gather(series[0],start_index,axis=0)  #[tf.newaxis,...]
    
    
    @tf.function
    def sequential_indices(self,start_points,length_of_indices): 
        grid_indices=tf.meshgrid(tf.range(0,length_of_indices),start_points)   #int((end_point-start_point)/batch_length)+1)) #   print("gt=",gridtest)
     #   return tf.convert_to_tensor(tf.add(gridtest[0],gridtest[1]))   #[:,:,np.newaxis
        return tf.add(grid_indices[0],grid_indices[1])   #[:,:,np.newaxis
       
    
    
    
    
    
    @tf.function
    def create_X_and_Y_batches(self,batches,batch_length,no_of_batches):
        start_points=tf.random.uniform(shape=[no_of_batches],minval=0,maxval=batches.shape[0]-batch_length-1,dtype=tf.int32)
     #   tf.print("start point=",start_point,"nob",no_of_batches,"batch len=",batch_length)
        X_indices=self.sequential_indices(start_points,batch_length)
        X=tf.gather(batches,X_indices,axis=0)
    
     #   tf.print("X=",X,X.shape)
         
        Y_indices=self.sequential_indices(start_points+1,batch_length)
        Y=tf.gather(batches,Y_indices,axis=0)
    
     #   tf.print("Y=",Y,Y.shape)
        return X,Y
    
            
        
     
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
        root_logdir = "./SCBS_outputs"
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

    
# @tf.function
    def predict_series(self,model,series):
    
       # predict_ahead=series   #[:,start_point:end_point,:]   #.astype(np.float32)   #[0,:,0]     #]
      #  predict_ahead_end_point=end_point-start_point
       # Y_probas = np.empty((1,batch_length),dtype=np.int32)  #predict_ahead_steps))
       # print("prediction series shape",series.shape)
        predict_ahead=model(tf.cast(series[:,-self.batch_length:,:],tf.float32),training=True)
        #print("predict ahead shape",predict_ahead.shape)

        new_prediction=np.empty((1,0,1))

        for batch_ahead in range(0,int(round(self.predict_ahead_length/self.batch_length,0)+1)):        
            # multiple more accurate prediction
    
    #            Y_probs=np.stack([model(predict_ahead[:,-batch_length:,:],training=True)[0,:,-1] for sample in range(pred_error_sample_size)])         
                           
            Y_probs=np.stack([model(predict_ahead[:,-self.batch_length:,:])[0,:,-1].numpy() for sample in range(self.pred_error_sample_size)])         
     #       print("Y_probs=\n",Y_probs.shape)
            Y_mean=Y_probs.mean(axis=0)[np.newaxis,...]
            Y_mean=Y_mean[...,np.newaxis]
            Y_mean=Y_mean[:,:self.batch_length,:]
      #  Y_stddev=Y_probs.std(axis=0)#[np.newaxis]
          
      #      print("Y_mean=",Y_mean.shape)
       #     print("before new prediction=",new_prediction.shape)
           # print("before predict ahead=",predict_ahead.shape)
    
            new_prediction=np.concatenate((new_prediction,Y_mean),axis=1) 
         #   pa=predict_ahead[0,:,-1]
         #   predict_ahead=np.concatenate((predict_ahead,Y_mean),axis=1)
      #      print("mafter new prediction=",new_prediction.shape)
          #  print("mafter predict ahead=",predict_ahead.shape)
            
        return new_prediction  
  


       
    # =============================================================================
    #  
    # 
    # # the plot dictionary is the holding area of all data
    # # it has a 3-tuple for a key
    # 
    # first is query name
    # second is 0= originsal data, 1 = actual query don't predict or  plot, 2 = plot actual, 3 = plot prediction, 4 = plot prediction with error bar
    # third is the start point
    #
    # the value is a 1D Tensor except at the start where sales_df is a pandas dataframe
    #     
    # =============================================================================
          


    #@tf.function
    def append_plot_dict(self,plot_dict,query_name,new_prediction):
        
   #     new_key="('"+str(query_name)+"_prediction', 2, "+str(self.end_point)+")"
        new_key=tuple(["'"+str(query_name)+"_prediction'",3,self.end_point])

        print("append plot dict new key=",new_key)
        plot_dict[new_key]=new_prediction
        return plot_dict
        
        
    def build_final_plot_df(self,plot_dict):
        final_plot_df=pd.DataFrame(columns=plot_dict.keys(),index=self.dates)
        print("fpdf=",final_plot_df)
 #        for key in plot_dict.keys():
        return final_plot_df    
        
        
        
