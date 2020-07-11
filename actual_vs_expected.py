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



print("\n\nSales Crystal Ball Stack2 : TF2 Salestrans predict - By Anthony Paech 25/5/20")
print("=============================================================================\n")       
 


import sales_trans_lib_v6

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

tf.config.experimental_run_functions_eagerly(False)   #True)   # false

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
import os
from pathlib import Path
import pandas as pd
import pickle
# #import random
import datetime as dt
import gc
import multiprocessing

from numba import cuda
# ""
import collections
from collections import defaultdict
# from datetime import datetime
# #import SCBS0 as c


# # to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# # To plot pretty figures
# #%matplotlib inline
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
#ion() # enables interactive mode
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)





###########################################3





def main():
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

    st=sales_trans_lib_v6.salestrans()   # instantiate the sales trans class

    
 #   print("\n\nSales Crystal Ball Stack2 : TF2 Salestrans predict - By Anthony Paech 25/5/20")
 #   print("=============================================================================\n")       
    
    print("Python version:",sys.version)
    print("\ntensorflow:",tf.__version__)
#    print("eager exec:",tf.executing_eagerly())

    print("keras:",keras.__version__)
    print("numpy:",np.__version__)
    print("pandas:",pd.__version__)
    print("matplotlib:",mpl.__version__)
    print("salestranslib:",st.__version__)
 #   print("sklearn:",sklearn.__version__)
   
    print("\nnumber of cpus : ", multiprocessing.cpu_count())

    visible_devices = tf.config.get_visible_devices('GPU') 

    print("tf.config.get_visible_devices('GPU'):",visible_devices)
    
 
    print("\n============================================================================\n")       


       
    np.random.seed(42)
    tf.random.set_seed(42)
           
 

    answer="y"
    answer=input("Load salestrans?")
    if answer=="y":
        sales_df=st.load_sales(st.filenames)  # filenames is a list of xlsx files to load and sort by date
      #  sales_df=st.preprocess_sales(sales_df)
  
        with open(st.sales_df_savename,"wb") as f:
            pickle.dump(sales_df, f,protocol=-1)
       
  
       # print("start sales dataframe=\n",sales_df)    # pandas dataframe

    # =============================================================================
    #  
    # 
    # # the plot dictionary is the holding area of all data
    # # it has a 3-tuple for a key
    # 
    # first is query name
    # second is 0= originsal data, 1 = actual query don't predict or  plot, 2 = plot actual mat, 3 = plot prediction, 4 = plot prediction with error bar
    # third is the start point
    # fourth is the plot number
    #
    # the value is a 1D Tensor except at the start where sales_df is a pandas dataframe
    #     
    # =============================================================================
          

    
    
        plot_dict=dict({('loaded_dataframe',0,0,0) : sales_df})
        st.save_plot_dict(plot_dict,st.plot_dict_filename)
    #    plot_dict[('loaded_model',0,0,0)]=
    #    st.save_plot_dict(plot_dict,st.plot_dict_filename)
        # model_filename="SCBS_model_"+str(qnames[query_number])+".h5"
        # print("\nsave model '",model_filename,"'")
        # model.save(model_filename, include_optimizer=True)
      
        # model_filename_list.append(model_filename)   

    
 
    else:
        
        plot_dict=st.load_plot_dict(st.plot_dict_filename)
     #   plot_dict=st.empty_plot_dict_except_loaded(plot_dict)
        sales_df=plot_dict[('loaded_dataframe',0,0,0)] 
        plot_dict=dict({('loaded_dataframe',0,0,0) : sales_df})    # clear out plot_dict mats and predictions
  #      print("\nsave model\n")
  #       model.save("GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
  
    #    for key in plot_dict.copy():
    #        if key[1]==2 | key[1]==3:
    #          del plot_dict[key]
              



    plot_dict=st.query_sales(sales_df,st.queryfilename,plot_dict)  
    
    
    ###########################33
    print("\n====================================================")       

    first_date=sales_df['date'].iloc[-1]
    last_date=sales_df['date'].iloc[0]
    print("\nData available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")
    
    start_date = pd.to_datetime(st.data_start_date) + pd.DateOffset(days=st.start_point)
    end_date = pd.to_datetime(st.data_start_date) + pd.DateOffset(days=st.end_point)
        
      #  end_date = pd.DateOffset("02/02/18", periods=self.end_point)   # 2000 days
    print("Data training window:\nstart date:",start_date,"\nend date:",end_date,"\n")
    print("====================================================\n")       

    ##################################
    
    
    
    del sales_df
    gc.collect()
    
    st.save_plot_dict(plot_dict,st.plot_dict_filename)
#    plot_dict=st.remove_key_from_dict(plot_dict,('loaded_dataframe',0,0))   # to save memory  

 #   print("\nplot_dict=\n",plot_dict.keys())
#    start_dict_keys=plot_dict.keys().copy()
#    print("start dict keys",start_dict_keys)
    
    plot_dict=st.load_plot_dict(st.plot_dict_filename)
    dct=sorted(plot_dict.items(), key=lambda x: x[0][3])
    plot_dict = collections.OrderedDict(dct)
    print("\nQueries created:\n",list(plot_dict.keys()))
    total_queries=0
    for k in plot_dict.keys():
        if k[1]==2:
            total_queries+=1
    print("\nNo of plottable queries created",total_queries)
    query_count=1
    for k in plot_dict.copy():
      #  series=plot_dict[k]
        query_name=k[0]
        plot_number=k[3]
        if k[1]==2:
            print("\nQuery name to train on:",query_name,": (",query_count,"/",total_queries,")\n")  #," : ",plot_dict[k][0,-10:])
        #    batches=st.build_all_possible_batches_from_series(plot_dict[k],st.batch_length*2+1)
         #   print("all batches shape=",batches.shape)
            X,Y=st.create_X_and_Y_batches(plot_dict[k],st.batch_length,st.no_of_batches)
           # X,Y=st.create_X_and_Y_batches(plot_dict[k],st.batch_length,st.no_of_batches)
            dataset=tf.data.Dataset.from_tensor_slices((X,Y)).cache().repeat(st.no_of_repeats)
     #       dataset=tf.data.Dataset.from_tensor_slices((X[:,:,:1],Y)).cache().repeat(st.no_of_repeats)

            # dataset=dataset.cache().repeat(st.no_of_repeats)
 
    #  #   dataset=dataset.map(preprocess,num_parallel_calls=None)
        #    dataset=dataset.cache() 
            dataset=dataset.shuffle(buffer_size=st.no_of_batches+1,seed=42)
         #   shapes = (tf.TensorShape([None,1]),tf.TensorShape([None,st.batch_length]))
          #  shapes = (tf.TensorShape([None,1]),tf.TensorShape([None,st.batch_length]))

#            train_set = dataset.padded_batch(1,padded_shapes=shapes, padding_values=(0,0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))
#            valid_set = dataset.padded_batch(1,padded_shapes=shapes, padding_values=(0,0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))

         #   train_set = dataset.padded_batch(1,padded_shapes=shapes).prefetch(1)   #, padding_values=(None, None))
         #   valid_set = dataset.padded_batch(1,padded_shapes=shapes).prefetch(1)   #, padding_values=(None, None))




        #    train_set = dataset.padded_batch(1,padded_shapes=([st.batch_length,st.batch_length], [st.batch_length,st.batch_length]), padding_values=(-1, 0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))
        #    valid_set = dataset.padded_batch(1,padded_shapes=([st.batch_length,st.batch_length], [st.batch_length,st.batch_length]), padding_values=(-1, 0), drop_remainder=True).prefetch(1)   #, padding_values=(None, None))

        #    train_set=dataset.batch(1,drop_remainder=True).prefetch(1)
        #    valid_set=dataset.batch(1,drop_remainder=True).prefetch(1)
            
            
            train_set=dataset.batch(1).prefetch(1)
            valid_set=dataset.batch(1).prefetch(1)
       
        
######################################################################       
        
        
            model=st.model_training_GRU(train_set,valid_set,query_name)
            new_query_name=query_name+str("_GRU")
             
            print("\nGRU Predicting....",new_query_name)
         #   series=series[...,tf.newaxis]
 #           new_prediction,new_stddev=st.predict_series(model,plot_dict[k][:,st.start_point:st.end_point+1][...,tf.newaxis])
          #  new_prediction,new_stddev=st.predict_series(model,plot_dict[k][...,tf.newaxis])
            new_prediction,new_stddev=st.simple_predict(model,plot_dict[k][...,tf.newaxis])

        #    print("new predictopn=",new_prediction,new_prediction.shape)
      #      print("predict ahead=",predict_ahead)
            
            plot_dict=st.append_plot_dict(plot_dict,new_query_name,new_prediction,new_stddev,plot_number)  
     #       plot_dict=st.append_plot_dict(plot_dict,query_name,new_prediction,plot_number)  
            print("save plot for",new_query_name)
            st.save_plot_dict(plot_dict,st.output_dir+st.plot_dict_filename)
            print("Clear tensorflow session and garbage collect..")
            query_count+=1
            tf.keras.backend.clear_session()
            gc.collect()
          #  cuda.select_device(0)
          #  cuda.close()
    
  
         #   plot_number+=1
  
# =============================================================================
# ##################################################################3333
#           
#             model=st.model_training_wavenet(train_set,valid_set,query_name)
#             new_query_name=query_name+str(":Wavenet")
#                          
#             print("\nWavenet Predicting....",new_query_name)
#             new_prediction,new_stddev=st.simple_predict(model,plot_dict[k][...,tf.newaxis])
#             
#             plot_dict=st.append_plot_dict(plot_dict,new_query_name,new_prediction,new_stddev,plot_number)  
#         
#             st.save_plot_dict(plot_dict,st.output_dir+st.plot_dict_filename)
#    
#  ##########################################################
# =============================================================================
    
 
    
 
    
 
         
    gc.collect()
    tf.keras.backend.clear_session()
        #cuda.select_device(0)
    #cuda.close()
    
    
    print("save plot_dict")
    st.save_plot_dict(plot_dict,st.plot_dict_filename)

    print("purging plot_dict of non plottable data")
    for key in plot_dict.copy():
        if ((key[1]==1) | (key[1]==0)):
              del plot_dict[key]
    
  #  print("plot dict purged")        
 #   for key in plot_dict.keys():
 #       print(key,plot_dict[key].shape)
     
        
    # sort by plot_number  
    dct=sorted(plot_dict.items(), key=lambda x: x[0][3])
    plot_dict = collections.OrderedDict(dct)
 #   print("\n2sorted plot_dict after prediction",plot_dict.keys())
  
    new_plot_df,new_column_names=st.build_final_plot_df(plot_dict)
  
    print("Plotting plot_dict...")
  #  print("new_plot df=\n",new_plot_df.columns,"->",new_column_names,new_plot_df.shape)

    st.plot_new_plot_df(new_plot_df)
    
    plt.show()
    plt.close("all")
    print("Saving ")
    
    
    new_plot_df=st.simplify_col_names(new_plot_df,new_column_names)
#    new_plot_df=st.clean_up_col_names(new_plot_df)
       #    plot_dict[key[0]] = plot_dict.pop(key)

    #plot_df=pd.DataFrame.from_dict(plot_dict,orient='index',dtype=np.int32)
  
    print("\nwrite predictions to sales_prediction(.....).CSV file....")
   
    print("Saving pickled final table - final_series_table.pkl",new_plot_df.shape)
    
        #    series_table=series_table.T       
    pd.to_pickle(new_plot_df,"final_series_tables.pkl")
 
    new_plot_df=st.clean_up_col_names(new_plot_df)  
    
    forecast_table = new_plot_df.resample('M', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)

    forecast_table.index=forecast_table.index.strftime("%Y-%m-%d")

    forecast_table.to_excel(st.output_dir+"SCBS2_forecast_table.xlsx") 
        
             
  #  print("\n\npredict module finish\n\n")
 
    print("\n\nFinished.")
    gc.collect()      
        
        
    return


if __name__ == '__main__':
    main()

        
          
          
          

