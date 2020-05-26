#  sales_trans predict v1 by Anthony Paech written 25/5/20
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






import sales_trans_lib_v1

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


# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)



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

from collections import defaultdict
# from datetime import datetime
# #import SCBS0 as c


# # to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# # To plot pretty figures
# #%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)



def plot_multiple_forecasts2(X_orig,X_pred, X_actual,title_name,first_start_point,first_end_point):
    n_steps = X_pred.shape[1]
    ahead = X_actual.shape[1]
    n_orig=X_orig.shape[1]-first_start_point
 #   plt.plot(np.arange(0, n_steps), X[0, :, 0], "r-", label="forecast", markersize=10)
 
  #  plt.plot(np.arange(0, ahead), X_actual[0, :, 0], "g-", label="Actual", markersize=10)


    plt.plot(np.arange(0, n_orig), X_orig[0, first_start_point:, 0], "b-", label="Actual", markersize=10)

 #   plot_series(X[0, :, 0])
    if title_name:
        plt.title(title_name)
    plt.plot(np.arange(0,n_steps), X_pred[0, :, 0], "r.", label="Predict",markersize=4)
#    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "b-", label="Forecast", markersize=10)
 
#    plt.axis([0, n_steps + ahead, -1, 1])
#    plt.axis([0, n_steps + ahead, 0 , np.max(Y_actual)])
    plt.axis([0, n_steps+10, 0 , np.max(X_actual)])

    plt.legend(fontsize=14)





def plot_log_learning_curves(title,epochs,loss, val_loss,query_name):
    ax = plt.gca()
    ax.set_yscale('log')
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, np.amin(loss), np.amax(loss)])
    plt.legend(fontsize=11)
    plt.title(title,fontsize=11)
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.grid(True)
    save_fig(c.output_dir+"log_learning_curve "+str(query_name))
    return









def save_fig(fig_id, images_path,tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


    
 
   

def graph_a_series(series_table,dates,column_names): 

 #   series_dict_elem=series_dict_elem.astype(str)  
 
#    print("1series table shaper",series_table,series_table.shape)
#    series_table = series_table.reindex(natsorted(series_table.columns), axis=1)
 #   print("2series_tsable=",series_table.shape)
 #   series_table = series_table.reindex(natsorted(series_table.columns), axis=1)
  #  series_table=series_table.T 
 #   print("series_table.columns",series_table.columns)
   # # dates=pd.to_timestamp(series_table.index,freq="d",how="S").to_list()
  #  ndates=series_table.index.astype(str).tolist()
  #  print("dates=",len(dates))

    series_table=series_table.T
  #  print("2series_tsable=",series_table.shape)
 
    series_table['period'] = pd.to_datetime(dates,infer_datetime_format=True)
    
   # print("series table before sorting=\n",series_table,series_table.shape)
    series_table = series_table.reindex(natsorted(series_table.columns), axis=1)
   # print("series table after sorting=\n",series_table,series_table.shape)

    
 #   print("\ngraph a series, table.T=\n",series_table,series_table.shape)
 #   print("3series_table.shape",series_table.shape,len(dates))
#    series_table=np.num_to_nan(series_table,0)
 #   series_table[column_names] = series_table[column_names].replace({0:np.nan})
 #   print("graph a series - series_table",series_table)
  #  series_table=series_table.T 
  #  print("4series_table.shape=\n",series_table,series_table.shape,len(dates))
#
    ax = plt.gca()

  #  del cols[-1]  # delete reference to period column
#    print("cols=",cols,len(cols))
    col_count=0
  #  print("Column names",column_names)
    for col in list(series_table.columns):
        if col=='period':
            pass
        else:    
      #      print("series_table,[col]col=",col)
            series_table[col] = series_table[col].replace({0:np.nan})
      #      print("\ngraph a series - series_table",col,"\n",series_table[col])
      
          #  print("find series type",col,"=",find_series_type(col))  
            series_suffix= str(find_series_type(col)) 
     #       print("series suffix=",series_suffix)
          #  series_type=str(series_dict[series_suffix])   # name, type of plot, colour
       #     print("series type=\n",series_type,">",series_type)   # name, type of plot, colour
            if (series_suffix=="mt_pred_mc"): # | (series_suffix=="mt_yerr_mc")):
                pred_plot=col
          #      print("pred_polt=",pred_plot)
    
            if col in column_names:
         #       print("add a series",col,series_suffix)
    #            series_table.plot(kind=series_dict_elem[1],x='period',y=col,color=series_dict_elem[2],ax=ax,fontsize=8)
                 #    plt.errorbar('period', series_table[col], yerr=series_table.iloc[col_count+1], data=series_table)
                if series_suffix=="mt_yerr_mc":
           #         print("\nplotting error bar\n")
                    plt.errorbar('period', pred_plot, yerr=col, fmt="r.",ms=3,data=series_table,ecolor="magenta",errorevery=1)
                   # plt.errorbar(series_table['period'], pred_plot, yerr=col, fmt="k.",ms=2,data=series_table,ecolor="magenta",errorevery=2)
               #     plt.errorbar(series_table.iloc[start_point:, series_table.columns.get_loc('period')], pred_plot, yerr=col, fmt="k.",ms=2,data=series_table,ecolor="magenta",errorevery=2)

      
                else:   
                    if series_suffix=="mt_":
                         plt.plot(series_table['period'],series_table[col],"b-",markersize=3,label=col)    #,range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
                    #     plt.plot(series_table['period'],series_table[col],"b-",markersize=3,label=col)    #,range(start_point,original_steps), ys[0,:(original_steps-start_point),p],"g.", markersize=5, label="validation")
    
                    elif series_suffix=="mt_pred_mc":        
                   #     plt.plot(series_table['period'],series_table[col],"g.",markersize=3,label=col) 
                        plt.plot(series_table['period'],series_table[col],"g.",markersize=3,label=col) 
    
                    else: 
                      #  pass
                        plt.plot(series_table['period'],series_table[col],"k.",markersize=3,label=col) 
                 #   series_table.plot(kind='scatter',x='period',y=col,color=series_type,ax=ax,fontsize=8,s=2,legend=False)
                  #      series_table.plot(kind='line',x='period',y=col,color=series_type,ax=ax,fontsize=8)
    
        col_count+=1    
        
    return 
    
     
  
# def add_a_new_series(table,arr_names,arr,start_point,predict_ahead_steps,periods_len):
  
#   #  print("ans input table shape",table,table.shape)
#   #  print("add a new series first date=",table.index[0])
#   #  print("ans arr_names",arr_names)
#   #  print("ans arr[0]",arr[0].shape)
#     first_date=(table.T.index[0]+timedelta(days=int(start_point+1))).strftime('%Y-%m-%d')
#   #  print("ans first_date",first_date)
#     pidx = pd.period_range(table.T.index[0]+timedelta(days=int(start_point+1)), periods=periods_len-1-start_point)   # 2000 days  
#   #  print("befofre dates=pidx",pidx,periods_len)
    
#     pad_before_arr = np.empty((1,start_point,arr.shape[2]))
#     pad_before_arr[:] = np.NaN
#   #  print("pad befor arr=\n",pad_before_arr.shape,"arr[0]=\n",arr[:,start_point:,:].shape)
#     y_values= np.concatenate((pad_before_arr,arr[:,start_point:,:]),axis=1)
#    # print("aaseries y_values.shapoe",y_values[0].shape)
#   #  print("pidx=",len(pidx))
# #    new_cols=pd.DataFrame(arr[0,start_point:,:],columns=arr_names,index=pidx[start_point:]).T
#  #   new_cols=pd.DataFrame(y_values,columns=arr_names,index=pidx[start_point:]).T
#     new_cols=pd.DataFrame(y_values[0],columns=arr_names,index=pidx).T

#   #  print("ans input new cols",new_cols,new_cols.shape)
#   #  print("Table=\n",table,table.shape)
#  #   table=table.T 
#     table2=pd.concat((table,new_cols),join='outer',axis=0)   
#     new_product_names=list(table2.T.columns)
#   #  print("ans output table2 shape",table2,table2.shape)
# #    print("extended dates",list(series_table.index))
#     extended_dates=list(table2.columns.to_timestamp(freq="D",how="S"))
#   #  print("extended dates=",extended_dates)
#     return table2,new_product_names,extended_dates
  


# def build_mini_batches(mat_sales_orig,no_of_batches,batch_length,start_point,end_point):
#     if batch_length>(end_point-start_point):
#      #   print("\nonly one \n")
#         repeats_needed=1
# #        gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,end_point-start_point-start_point-batch_length+1))
#      #  gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))

#         gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))
#    #     print("raandom",gridtest)
#     else:    
#         repeats_needed=no_of_batches/int(((end_point-start_point)/batch_length)+1)  #      repeats_needed=int(no_of_batches/(end_point-start_point-start_point-batch_length))

#         gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.arange(0,int(((end_point-start_point)/batch_length)+1)))   #int((end_point-start_point)/batch_length)+1))
#   #  print("repeats needed=",repeats_needed)
#     #gridtest=np.meshgrid(np.arange(np.random.random_integers(0,total_steps,n_steps))), np.arange(0,n_steps))
#   #  print(gridtest,len(gridtest) ) #.shape)
#     start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
#   #  print("start index.shape",start_index,start_index.shape)
#     np.random.shuffle(start_index)
#    # print("start index",start_index,start_index.shape)
 
#     new_batches=mat_sales_orig[0,start_index]
#     np.random.shuffle(new_batches)
#   #  print("new batches",new_batches)
#     #if repeats_needed==1:
#       #  print(" one only - batches complete. batches shape:",new_batches.shape)
    
#     return new_batches   #,new_batches[:,1:batch_length+1,:]














    
    
def actual_days(series_table):
 #   print("ad=",series_table.index[0])
    first_date=series_table.index[0].to_timestamp(freq="D",how="S")
    last_date=series_table.index[-1].to_timestamp(freq="D",how="S")
    return (last_date - first_date).days +1    #.timedelta_series.dt.days    



       

def find_series_type(series_name):
    return series_name[series_name.find(':')+1:]




  
    
# def load_series(start_point,end_point):    
#     with open("batch_dict.pkl", "rb") as f:
#          seriesbatches = pickle.load(f)
#     #mat_sales_x =seriesbatches[0][7]
#     series=seriesbatches[0][9]
#     print("full series shape=",series.shape)
#     mat_sales_x=series.to_numpy().astype(np.int32)
#    # print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))
#   #  mat_sales_x=mat_sales_x.astype(np.int32)

#     #Wmat_sales_x=mat_sales_x[...,np.newaxis].astype(np.int32)
#     print("mat sales x.shape",mat_sales_x.shape)
#     print("mat_sales_x size=",mat_sales_x.nbytes,type(mat_sales_x))

#     print("loaded mat_sales x shape",mat_sales_x.shape)
#     print("start point=",start_point)
#     print("end_point=",end_point)
#     shortened_series=series.iloc[:,start_point:]
#     dates=shortened_series.T.index.astype(str).tolist()  #.astype(str)) #strftime("%Y-%m-%d"))
#     shortened_series=series.to_numpy()

#     mat_sales_x=mat_sales_x[:,start_point:end_point+1][...,np.newaxis]
#     print("trimmed mat_sales x shape",mat_sales_x.shape)
#     print("batch len=",batch_length)
#  #   print("shortened_series=",shortened_series.shape)
#    # series=series[:,start_point:end_point]
#     #print("series trimmed=",series,series.shape)
#     shortened_series=shortened_series[...,np.newaxis].astype(np.int32)
#     print("shortened series=",shortened_series.shape)
#     return shortened_series,mat_sales_x,dates   #[..., np.newaxis].astype(np.float32)    
  

# def create_batches(no_of_batches,batch_length,mat_sales_x,start_point,end_point):    
#  #   print("mat_sales_x=\n",mat_sales_x[0])
#  #   print("nob=",no_of_batches)
#     if no_of_batches==1:
#      #   print("\nonly one \n")
#         repeats_needed=1
# #        gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,end_point-start_point-start_point-batch_length+1))
#      #  gridtest=np.meshgrid(np.arange(start_point,start_point+batch_length),np.random.randint(0,int(((end_point-start_point)/batch_length)+1)))

#         gridtest=np.meshgrid(np.arange(0,batch_length),np.random.randint(0,end_point-start_point-batch_length+1))
#    #     print("raandom",gridtest)
#     else:    
#         repeats_needed=int(no_of_batches/(end_point-batch_length-start_point)+1)  #      repeats_needed=int(no_of_batches/(end_point-start_point-start_point-batch_length))

#         gridtest=np.meshgrid(np.arange(0,batch_length),np.arange(0,end_point-start_point-batch_length+1))  #int((end_point-start_point)/batch_length)+1))
#  #   print("gi=\n",gridtest)
#     start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
#  #   print("start index=",start_index,start_index.shape)
#     np.random.shuffle(start_index)
# #    print("start index min/max=",np.min(start_index),np.max(start_index),start_index.shape) 

#     X=mat_sales_x[0,start_index,:]
#     np.random.shuffle(X)
#  #   print("X.shape=\n",X.shape)
#     gc.collect()
#     return X   #,new_batches[:,1:batch_length+1,:]





def create_plot_df(plot_dict,dates,date_len):
    start_points=[]
   
    for series_number in range(1,9):        
        subdict = {k: v for k, v in plot_dict.items() if str(k).startswith("("+str(series_number))}
        if subdict: 
            for series_type in range(1,9):
                   subdict2 = {k: v for k, v in subdict.items() if str(k).startswith("("+str(series_number)+", "+str(series_type))}
                   if subdict2:
                       key_value=list(subdict2.keys())[0]
                       dict_value=list(subdict2.values())[0]
                       dict_name=key_value[2]
                       if series_number==1:   # start_point
                          # sp=dict_value
                           start_points.append(dict_value)   
                        #   series_names.append(dict_name)
                       elif series_number==2:   # data length

                           sp=start_points.pop(0)
                           filler_array=np.zeros(sp)
                           filler_array[:] = np.nan
                           back_filler=date_len-(sp+dict_value.shape[0])
                           
 
                           if back_filler<=0:
                               back_filler_array=np.zeros(0)
                           else:    
                               back_filler_array=np.zeros(back_filler)
                           back_filler_array[:] = np.nan
    
                               
                               
                           subdict3 = {k: v for k, v in plot_dict.items() if str(k).startswith(("("+str(series_number)+", "+str(series_type)))}
    
                           kv=list(subdict3.keys())[0] 
                       #    print("key v=",kv)
                           uk={kv:np.concatenate((filler_array,plot_dict[kv],back_filler_array),axis=0)[:date_len]}
      
                           plot_dict.update(uk)
   
 
# plot only array data
    series_number=2        
    subdict = {k[2]: v for k, v in plot_dict.items() if str(k).startswith("("+str(series_number))}
 
    plot_df=pd.DataFrame.from_dict(subdict,orient='columns',dtype=np.int32)
    plot_df.index=dates

 #   print("plot_df=\n",plot_df)
    

    return plot_df


 
 


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




def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


 

def save_fig(fig_id, images_path, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

   
 
    
 
def plot_learning_curves(loss, val_loss,epochs,title):
    ax = plt.gca()
    ax.set_yscale('log')

    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
#    plt.axis([1, epochs+1, 0, np.max(loss[1:])])
    plt.axis([1, epochs+1, 0, np.max(loss)])

    plt.legend(fontsize=14)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)






###########################################3





def main():
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

    st=sales_trans_lib_v1.salestrans()   # instantiate the sales trans class

    
    print("\n\nTF2 Salestrans predict - By Anthony Paech 25/5/20")
    print("==================================================\n")       
    
    print("Python version:",sys.version)
    print("\ntensorflow:",tf.__version__)
    print("keras:",keras.__version__)
    print("numpy:",np.__version__)
    print("pandas:",pd.__version__)
    print("matplotlib:",mpl.__version__)
    print("salestranslib:",st.__version__)
 #   print("sklearn:",sklearn.__version__)
   
    print("\nnumber of cpus : ", multiprocessing.cpu_count())

    visible_devices = tf.config.get_visible_devices('GPU') 

    print("tf.config.get_visible_devices('GPU'):",visible_devices)


    print("\n==================================================\n")       


    answer="y"
    answer=input("Load salestrans?")
    if answer=="y":
        sales_df=st.load_sales(st.filenames)  # filenames is a list of xlsx files to load and sort by date
        sales_df=st.preprocess_sales(sales_df)
     
       # print("start sales dataframe=\n",sales_df)    # pandas dataframe
       
    # =============================================================================
    #  
    # 
    # # the plot dictionary is the holding area of all data
    # # it has a 3-tuple for a key
    # 
    # first is query name
    # second is 0 = don't plot, 1 = plot actual, 2 = plot prediction, 3 = plot prediction with error bar
    # third is the start point
    #
    # the value is a 1D Tensor except at the start where sales_df is a pandas dataframe
    #     
    # =============================================================================
    
    
    
        plot_dict=dict({('loaded_dataframe',0,0) : sales_df})
        st.save_plot_dict(plot_dict,st.plot_dict_filename)
    
    else:
        
        plot_dict=st.load_plot_dict(st.plot_dict_filename)
        sales_df=[plot_dict[k] for k in plot_dict.keys()][0] 


    plot_dict=st.query_sales(sales_df,st.queryfilename,plot_dict)  
    
    st.save_plot_dict(plot_dict,st.plot_dict_filename)
    plot_dict=st.remove_key_from_dict(plot_dict,('loaded_dataframe',0,0))   # to save memory  


    for key in plot_dict.keys():
        print("plot_dict key=",key)
    #    print("plot dict values",plot_dict[key])
        X=st.build_mini_batches(plot_dict[key],st.no_of_batches,st.batch_length)
        print("X",X.shape)
        Y=st.create_Y(plot_dict[key],X.shape[0],X.shape[1])
        print("Y",Y.shape)
            
        X_train,X_valid,X_test = st.train_test_split(X)
        Y_train,Y_valid,Y_test = st.train_test_split(Y)
        
        print("X train, valid & test",X_train.shape,X_valid.shape,X_test.shape)
        print("Y train, valid & test",Y_train.shape,Y_valid.shape,Y_test.shape)

    
 #####################################3
 # model goes here
 #
 
     
        np.random.seed(42)
        tf.random.set_seed(42)
       
        model = keras.models.Sequential([
           keras.layers.GRU(st.neurons, return_sequences=True, input_shape=[None, 1]),
           keras.layers.BatchNormalization(),
           keras.layers.GRU(st.neurons, return_sequences=True),
           keras.layers.AlphaDropout(rate=st.dropout_rate),
           keras.layers.BatchNormalization(),
           keras.layers.TimeDistributed(keras.layers.Dense(st.batch_length))
        ])
    
        model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
       
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=st.patience),MyCustomCallback()]
       
        history = model.fit(X_train, Y_train, epochs=st.epochs,
                           validation_data=(X_valid, Y_valid))
       
       
        print("\nsave model\n")
        model.save("GRU_Dropout_sales_predict_model.h5", include_optimizer=True)
          
        model.summary()
       
        plot_learning_curves(history.history["loss"], history.history["val_loss"],st.epochs,"GRU and dropout")
        save_fig("GRU and dropout learning curve",st.images_path)
    
        plt.show()
    
    
        
     ###################################
       
    ########################################3
    #  send predictions to plot_dict and then excel    
        
       
        
    # =============================================================================
    #     
    # 
    #    #print(batch_dict)
    #     print("\n\nPredict from models")  
    #     
    #     
    #     
    #     print("\n\nunpickling model filenames")  
    #     with open("model_filenames.pkl", "rb") as f:
    #          model_filenames_list = pickle.load(f)
    #     #   #  testout2 = pickle.load(f)
    #     # qnames=[testout1[k][0] for k in testout1.keys()]    
    #     print("unpickled model filenames list",model_filenames_list)
    #     
    #     with open("batch_dict.pkl", "rb") as f:
    #         batches = pickle.load(f)
    #     
    #     with open("tables_dict.pkl", "rb") as f:
    #         all_tables = pickle.load(f)
    #       #  testout2 = pickle.load(f)
    #     qnames=[all_tables[k][0] for k in all_tables.keys()]    
    #     print("unpickled",len(all_tables),"tables (",qnames,")")
    #       
    #     
    # =============================================================================
        #required_starting_length=c.required_starting_length  #731+np.max(mats)+batch_length   # 2 years plus the MAT data lost at the start + batchlength
        
         
        
        ########################################
    # =============================================================================
    #     
    #     model_number=0
    #     for model_name in model_filenames_list:
    #         model=keras.models.load_model(model_name,custom_objects={"last_time_step_mse": last_time_step_mse})
    #     
    #         print("\nmodel=",model_name,"loaded.\n\n")
    #         model.summary
    #        
    #         with open("product_names_"+str(qnames[model_number])+".pkl","rb") as g:
    #              original_product_names=pickle.load(g)
    #      
    #      
    #         
    #         print("\nSingle Series Predicting",predict_ahead_length,"steps ahead.")
    #      #   print("series=\n",list(series_table.index),"\n")
    #         print("Loading mat_sales_x")
    #       #  mat_sales_x=np.load("mat_sales_x.npy")
    #       #  with open("batch_dict.pkl", "rb") as f:
    #       #      batches = pickle.load(f)
    #         mat_sales_x =batches[model_number][1]
    #     #    X=batches[model_number][10]
    #         
    #         
    #         mat_sales_orig=mat_sales_x
    #         mat_sales_pred=mat_sales_x
    #         
    #         product_names=batches[model_number][1]
    #         #series_table=all_tables[model_number][1]
    #         series_table=batches[model_number][3]
    #     
    #         print("unpickled series table shape",series_table.shape)
    #     
    #         print("mat_sales_x shape",mat_sales_x.shape)
    #         print("series table shape",series_table.shape)
    #     
    #         actual_days_in_series_table=actual_days(series_table.T)
    #             
    #         #series_table=series_table.T    
    #         print("actual days in series table=",actual_days_in_series_table)
    #         print("required minimum starting days for 2 year series analysis:",required_starting_length)
    #     
    #     
    #         periods_len=actual_days_in_series_table  #+predict_ahead_steps # np.max(mats)
    #         print("total periods=",periods_len)
    #     
    #     
    # =============================================================================
        #    original_steps=series_table.shape[1]
            
    #####################################################################3        
    
    # Now let's create an RNN that predicts the next 10 steps at each time step. That is, 
    # instead of just forecasting time steps 50 to 59 based on time steps 0 to 49, 
    # it will forecast time steps 1 to 10 at time step 0, then time steps 2 to 11 at time step 1, 
    # and so on, and finally it will forecast time steps 50 to 59 at the last time step. 
    # Notice that the model is causal: when it makes predictions at any time step, it can 
    # only see past time steps.    
    
     
    ###################################3
          #  gc.collect()
         
        print("\nPredicting....")
        #predict = np.empty((X.shape[0], batch_length, pred_length),dtype=np.int32)
        mat_sales_x=mat_sales_x.astype(np.float32)   #[0,:,0]     #]
        series_length=mat_sales_x.shape[1]
        predict_ahead=mat_sales_x[:,start_point:end_point,:]   #.astype(np.float32)   #[0,:,0]     #]
      #  predict_ahead_end_point=end_point-start_point
       # Y_probas = np.empty((1,batch_length),dtype=np.int32)  #predict_ahead_steps))
    #    predict_ahead=model(predict_ahead[:,-batch_length:predict_ahead.shape[1],:],training=True)[0,:,-1]
     
        new_prediction=np.empty((1,0,1))
        for batch_ahead in range(0,int(round(predict_ahead_length/batch_length,0)+1)):        
            # multiple more accurate prediction
    
    #            Y_probs=np.stack([model(predict_ahead[:,-batch_length:,:],training=True)[0,:,-1] for sample in range(pred_error_sample_size)])         
                           
            Y_probs=np.stack([model(predict_ahead[:,-batch_length:,:])[0,:,-1] for sample in range(pred_error_sample_size)])         
            print("Y_probs=\n",Y_probs.shape)
            Y_mean=Y_probs.mean(axis=0)[np.newaxis,...]
            Y_mean=Y_mean[...,np.newaxis]
            Y_mean=Y_mean[:,:batch_length,:]
      #  Y_stddev=Y_probs.std(axis=0)#[np.newaxis]
          
            print("Y_mean=",Y_mean.shape)
            print("before new prediction=",new_prediction.shape)
            print("before predict ahead=",predict_ahead.shape)
    
            new_prediction=np.concatenate((new_prediction,Y_mean),axis=1) 
            predict_ahead=np.concatenate((predict_ahead,Y_mean),axis=1)
            print("mafter new prediction=",new_prediction.shape)
            print("mafter predict ahead=",predict_ahead.shape)
    
       # print("Y_stddev=",Y_stddev.shape)
        print("pal=",predict_ahead_length)
        print("start point + pereiods len=",start_point+periods_len)
        print("new prediction shape=",new_prediction.shape)
        print("end poinbt=",end_point)
        
        #predict_values=np.concatenate([predict_values,Y_diag],axis=1) 
        
       # Y_mean=Y_diag
              
        
        plot_dict=dict({(1,1,"Actual_start") : start_point,
        #                    (2,1,"Actual_data") : predict_values[0,:,0],
                        (2,1,"Actual:"+str(qnames[model_number])) : mat_sales_orig[0,:,0],
        
                        (1,2,"MC_predict_mean_start") : end_point-start_point,     #+mat_sales_orig.shape[1],
                        (2,2,"Predicted:"+str(qnames[model_number])) : new_prediction[0,:,0],
                  #      (1,3,"MC_predict_stddev_start") : end_point,
                  #      (2,3,"MC_predict_stddev_data") : Y_stddev,
                    #    (1,4,"Full_actual_start") : start_point,
                    #    (2,4,"Full_actual") : shortened_series[0,:,0],
        
                    #    (0,0,"dates") : dates
                        })
         
        
        
               
        #print("plot_dict=",plot_dict) 
        plot_df=create_plot_df(plot_dict,dates,date_len)
     #   plot_df.plot()
        
    #    plt.errorbar(range(end_point-start_point-7,end_point-start_point-7+Y_mean.shape[0]), Y_mean, yerr=Y_stddev*2,errorevery=1,ecolor='magenta',color='red',linestyle='dotted')   #, label="dropout mean pred 95% conf")
        plot_df.plot()
    
        
        plt.legend(fontsize=10)
        plt.title(str(qnames[model_number])+":Unit sales/day prediction")
         #   plt.xlabel("Days")
        plt.ylabel("units")
        plt.grid(True)
        
        save_fig("actual_vs_predict_"+str(qnames[model_number]),images_path)
        plt.show()     


    
    gc.collect()
    tf.keras.backend.clear_session()
    #cuda.select_device(0)
#cuda.close()

         
    
    
       
    print("\nwrite predictions to sales_prediction(.....).CSV file....")
   
    print("Saving pickled final table - final_series_table.pkl",plot_df.shape)
    
        #    series_table=series_table.T       
    pd.to_pickle(plot_df,"final_series_table.pkl")
       
    forecast_table = plot_df.resample('M', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
    forecast_table.index=forecast_table.index.strftime("%Y-%m-%d")

    print("forecast_table=\n",forecast_table)
 #    s=str(qnames[model_number])
    s = s.replace(',', '_')
    s = s.replace("'", "")
    s = s.replace(" ", "")
    s = "xx"
    forecast_table.to_excel(c.output_dir+"SCBS_"+s+".xlsx") 
        
             
    print("\n\npredict module finish\n\n")
 
 #   print("\n\nFinished.")
    gc.collect()      
        
        
    return


if __name__ == '__main__':
    main()

        
          
          
          

