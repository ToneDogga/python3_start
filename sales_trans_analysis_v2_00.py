#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:52:36 2020

@author: tonedogga
"""
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

 



import os
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import timedelta
import pickle
import multiprocessing

from collections import defaultdict
from datetime import datetime
from pandas.plotting import scatter_matrix

import matplotlib.dates as mdates
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


colesjamsxls="Coles_jams_scan_data_300520.xlsx"
latestscannedsalescoles="Coles_IRI_portal_scan_data_170620.xlsx"

mats=7

#         self.start_point=0
        
  

def save_fig(fig_id, images_path, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

  
   

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./SCBS2_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

   
class MCDropout(keras.layers.Dropout):
     def call(inputs):
        return super().call(inputs,training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(inputs):
        return super().call(inputs,training=True)


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


def plot_learning_curves(loss, val_loss,epochs,title):
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


 

@tf.function
def sequential_indices(start_points,length_of_indices): 
     grid_indices=tf.meshgrid(tf.range(0,length_of_indices),start_points)   #int((end_point-start_point)/batch_length)+1)) #   print("gt=",gridtest)
     return tf.add(grid_indices[0],grid_indices[1])   #[:,:,np.newaxis
    
 
  
 # print("new Y shape",Y.shape)
 # for step_ahead in range(1, predict_ahead_length + 1):
 #     Y[...,step_ahead - 1] = series[..., step_ahead:step_ahead+batch_length-predict_ahead_length, 0]  #+1

@tf.function
def create_X_batches(series,batch_length,no_of_batches,start_point,end_point):
     start_points=tf.random.uniform(shape=[no_of_batches],minval=start_point,
                  maxval=end_point-batch_length,dtype=tf.int32)
     return sequential_indices(start_points,batch_length)[...,tf.newaxis]
 
 

@tf.function
def create_X_and_y_batches(X_set,y_set,batch_length,no_of_batches):
     indices=create_X_batches(X_set,batch_length,no_of_batches,0,X_set.shape[0])
  #   print("X indices shape",X_indices.shape)
  #   print("full indices shape",full_indices.shape)
                
     X=tf.cast(tf.gather(X_set,indices[:,:-1,:],axis=0),tf.int32)
     y=tf.cast(tf.gather(y_set,indices[:,-1:,:],axis=0),tf.int32)

     tf.print("2X[1]=",X[1],X.shape,"\n")
     tf.print("2y[1]=",y[1],y.shape,"\n")

     return X,y


     




def add_dates_back(df,all_dates):
    return pd.concat((df,all_dates_df),axis=1)   #,on='ww_scan_week',how='right')


def plot_query(query_df_passed,plot_col,query_name):
    query_df=query_df_passed.copy(deep=True)
  #  query_details=str(query_df.columns[0])
  #  query_df.columns = query_df.columns.get_level_values(0)
    print("query df shae",query_df.shape)
 #   query_df=self.mat_add_1d(query_df.to_numpy().swapaxes(0,1),self.mats[0])
    query_df['weeks']=query_df.index.copy(deep=True)  #.to_timestamp(freq="D",how='s') #
#        query_df['qdate']=pd.to_datetime(pd.Series(query_list).to_timestamp(freq="D",how='s'), format='%Y/%m/%d')
   # print("query list",query_list)
  #  query_df['qdate'].apply(lambda x : x.to_timestamp())
#    query_df['qdate']=query_list.to_timestamp(freq="D",how='s')
    query_list=query_df['weeks'].tolist()
  #  print("qudf=\n",query_df,query_df.columns[1][0])
    print("f",len(query_list))
    #   query_df['qdate'] = query_df.qdate.tolist()

    for col in plot_col:
        new_query_df=query_df[col].copy(deep=True)
     #   print("1query_df=",new_query_df)
        new_query_df=new_query_df.rolling(mats,axis=0).mean()
      #  print("1query_df=",new_query_df,new_query_df.shape,"col=",col)

      #  new_query_df['week_count']=np.arange(len(new_query_df.shape[0]))  

 #       print("2query_df=",new_query_df,new_query_df.shape,"col=",col)
     #   fill_val=new_query_df.iloc[mats+1]  #.to_numpy()
    #    print("fill val",fill_val)
      #  print("3query_df=",new_query_df,new_query_df.shape)

     #   new_query_df=new_query_df.fillna(fill_val)
       # query_df.reset_index()   #['qdate']).sort_index()
     #   query_df.reset_index(level='specialpricecat')
   #     query_df.reset_index(drop=True, inplace=True)
     #   new_query_df['weeks']=query_list   #.set_index(['qdate',''])
      #  print("3query df=\n",new_query_df,col)
      #  query_df=query_df.replace(0, np.nan)
       #     ax=query_df.plot(y=query_df.columns[0],style="b-")   # actual
     #   ax=query_df.plot(x=query_df.columns[1][0],style="b-")   #,use_index=False)   # actual
        ax=new_query_df.plot(x='week_count',y=col)   #,style="b-")   #,use_index=False)   # actual

  #  col_no=1
 #   query.plot(style='b-')
 #   ax.axvline(pd.to_datetime(start_date), color='k', linestyle='--')
 #   ax.axvline(pd.to_datetime(end_date), color='k', linestyle='--')

    plt.title("Unit scanned sales (000):"+query_name,fontsize=10)   #str(new_plot_df.columns.get_level_values(0)))
    plt.legend(fontsize=8)
    plt.ylabel("(000) units scanned/week")
    plt.grid(True)
 #   self.save_fig("actual_"+query_name+query_details,self.images_path)
    plt.show()

  


#############################
  
    
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors



 #   print("\n\nSales Crystal Ball Stack2 : TF2 Salestrans predict - By Anthony Paech 25/5/20")
 #   print("=============================================================================\n")       

print("Python version:",sys.version)
print("\ntensorflow:",tf.__version__)
#    print("eager exec:",tf.executing_eagerly())

print("keras:",keras.__version__)
print("numpy:",np.__version__)
print("pandas:",pd.__version__)
print("matplotlib:",mpl.__version__)

print("sklearn:",sklearn.__version__)
   
print("\nnumber of cpus : ", multiprocessing.cpu_count())

visible_devices = tf.config.get_visible_devices('GPU') 

print("tf.config.get_visible_devices('GPU'):",visible_devices)

 
print("\n============================================================================\n")       


   
np.random.seed(42)
tf.random.set_seed(42)
        
#column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
 
# col_dict=dict({0:"Coles_scan_week",
#                1:"BB_off_promo_sales",
#                2:"BB_on_promo_sales",
#                3:"SD_off_promo_sales",
#                4:"SD_on_promo_sales",
#                5:"BM_off_promo_sales",
#                6:"BM_on_promo_sales"})


# rename_dict=dict({"10":"BB_scanned_sales"})
       
# df=pd.read_excel(colesjamsxls,-1,skiprows=[0,1,2]).T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
# #print("before",df)
# df = df.rename(col_dict,axis='index').T

# df['Coles_scan_week']=pd.to_datetime(df['Coles_scan_week'],format="%d/%m/%y")
# #df['coles_scan_week']=df["date"] #.strftime("%Y-%m-%d")
# df.fillna(0.0,inplace=True)
# df.drop_duplicates(keep='first', inplace=True)
# #df.replace(0.0, np.nan, inplace=True)
# #print("after",df)

# df=df.sort_values(by=['Coles_scan_week'], ascending=True)
# df=df.set_index('Coles_scan_week') 
# df=df.astype(np.float32)  #,inplace=True)
# df['weekno']= np.arange(len(df))
# #print("final",df,df.T)

# df['BB_on_promo']=(df['BB_on_promo_sales']>0.0)
# df['SD_on_promo']=(df['SD_on_promo_sales']>0.0)
# df['BM_on_promo']=(df['BM_on_promo_sales']>0.0)

# df['BB_total_sales']=df['BB_off_promo_sales']+df['BB_on_promo_sales']
# df['SD_total_sales']=df['SD_off_promo_sales']+df['SD_on_promo_sales']
# df['BM_total_sales']=df['BM_off_promo_sales']+df['BM_on_promo_sales']

    


# df.replace(0.0, np.nan, inplace=True)

# print("df=",df,df.index)
# plot_query(df,['BB_total_sales','SD_total_sales','BM_total_sales'],'BB total scanned Coles jam units per week')



# #sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')  #,fit_reg=True,robust=True,legend=True) 
# #sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
# sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BB_on_promo')  #,fit_reg=True,robust=True,legend=True) 
# sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='BB_on_promo')

# sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
# sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')


# # convert weekly scan data to daily sales
# print(df)
# df=df*1000
# print(df)

# output_dir = log_dir("SCBS2")
# os.makedirs(output_dir, exist_ok=True)

# images_path = os.path.join(output_dir, "images/")
# os.makedirs(images_path, exist_ok=True)




# loadpkl="All_jams_coles.pkl"

# savepkl="All_jams_coles_plus_scanned_sales.pkl"

# print("Loading query dataframe:",loadpkl)
# data_df=pd.read_pickle(loadpkl)

# #print("daya df=\n",data_df)
# #print("pd",data_df.index) 

# #df.sort_values(by=['date'], inplace=True, ascending=False)
          
#      #   print(df.head(5))
#      #   print(df.tail(5))
   
# data_df.columns = data_df.columns.get_level_values(0)
# #print("dATA df",data_df)
# #data_df.index=data_df.index.astyoe('period')  #.to_period("W")
# data_df.columns = list(data_df.columns)
# data_df_list=data_df.index  
# data_df.reset_index(drop=True,inplace=True)  
# data_df['new_dates'] = data_df_list   #pd.Period(data_df_list, freq = "W")
# #print(data_df,data_df.Period(freq="W"))
# #print("DD",data_df,data_df.index)
# data_df['new_date2'] = data_df.new_dates.values.astype('datetime64[W]')
# #print("data_df",data_df)
# new_data_df=data_df.set_index('new_date2')
# #print("fdata",new_data_df)
# #data_df['new_dates2'] = pd.to_datetime(data_df['new_dates'])
# #df['T'] = pd.to_datetime('T')
# #df = df.set_index('T')
# #data_df=data_df.index.to_datetime()
# forecast_df = new_data_df.resample('W-SAT', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
# #print("fd=",forecast_df)
# #print("df",df)  #,forecast_df.index)

# joined_df=pd.concat([df,forecast_df],axis=1)
# #joined_df['weekno']= np.arange(len(joined_df)).astype(np.int32)
# joined_df=joined_df.rename(columns=rename_dict)
                           
# #print("joined=",joined_df,joined_df.T,joined_df.columns,joined_df.index)
# #data_df=data_df.index
# #data_df=data_df.new_dates.dt.to_period('W')
# #data_df['new_period'] = data_df['new_period'].astype('category')
#     #    df["period"]=df.date.dt.to_period('D')
#     #    df['period'] = df['period'].astype('category')
 
# #print("2datat df",data_df)
# #joined_df.replace(0.0, np.nan, inplace=True)

# #print("df=",df)
# plt.grid(True)
# joined_df[['BB_total_sales','BB_scanned_sales']].plot(title="Coles jam",grid=True)   #),'BB total scanned vs purchased Coles jam units per week')
# mat=4
# joined_df=joined_df.rolling(mat,axis=0).mean()

# #joined_df['BB_scanned_sales']=joined_df['BB_scanned_sales'].rolling(mat,axis=0).mean()
# joined_df[['BB_total_sales','BB_scanned_sales']].plot(grid=True,title="Coles Jam units Moving total "+str(mat)+" weeks")   #),'BB total scanned vs purchased Coles jam units per week')
# #plt.grid(True)
# plt.show()
# plt.close("all")


# print("saving query dataframe:",savepkl)
# pd.to_pickle(joined_df,savepkl)


# # use Coles scan data from IRI weekly to predict Coles orders
# #  X is the BB_total_sales
# # the Target y is scanned sales 4 weeks ahead
# # 
# df=joined_df[['BB_total_sales','BB_scanned_sales']]
# print(df)

#      # 
#     # 
#     # no of weeks
# target_offset=3
# batch_length=4
# no_of_batches=1000
# no_of_repeats=4
# epochs=20
# start_point=7
# end_point=123
# df.fillna(0,inplace=True)
# # use a simple model
# X_set=df['BB_total_sales'].iloc[target_offset:].to_numpy()
# X_set=np.concatenate((X_set,np.zeros(target_offset)),axis=0).astype(np.int32)
# y_set=df['BB_scanned_sales'].iloc[start_point:end_point].to_numpy().astype(np.int32)

# dates=df[start_point:end_point].index.tolist()
# pred_dates=df[end_point-1:].index.tolist()

# print("1Xset=",X_set,X_set.shape)
# print("1yset=",y_set,y_set.shape)


# X_set=X_set[start_point:end_point]
# y_pred=df['BB_scanned_sales'].iloc[end_point-1:].to_numpy().astype(np.int32)
# print("2Xset=",X_set,X_set.shape)
# print("2yset=",y_set,y_set.shape)

# print("y_pred=",y_pred,y_pred.shape)


# ###############################
# # batches of X shape (no of batches,batch length, 1)
# # batches of Y shape (no of batches,batch length, 1)

# answer=input("Train a model?")
# if answer=="y":
    
    
#     X,y=create_X_and_y_batches(X_set,y_set,batch_length,no_of_batches)
    
    
    
#     ##########################3
    
#     dataset=tf.data.Dataset.from_tensor_slices((X,y)).cache().repeat(no_of_repeats)
#     dataset=dataset.shuffle(buffer_size=1000,seed=42)
    
#     train_set=dataset.batch(1).prefetch(1)
#     valid_set=dataset.batch(1).prefetch(1)
       
     
    
#     ##########################
#     print("\nTraining with GRU and dropout")
#     model = keras.models.Sequential([
#     #     keras.layers.Conv1D(filters=st.batch_length,kernel_size=4, strides=1, padding='same', input_shape=[None, 1]),  #st.batch_length]), 
#       #   keras.layers.BatchNormalization(),
#          keras.layers.GRU(200, return_sequences=True, input_shape=[None, 1]), #st.batch_length]),
#         # keras.layers.BatchNormalization(),
#          keras.layers.GRU(200, return_sequences=True),
#        #  keras.layers.AlphaDropout(rate=0.2),
#        #  keras.layers.BatchNormalization(),
#          keras.layers.TimeDistributed(keras.layers.Dense(1))
#     ])
      
#     model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
     
#     model.summary() 
     
#     #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),self.MyCustomCallback()]
     
#     history = model.fit(train_set ,epochs=epochs,
#                          validation_data=(valid_set))  #, callbacks=callbacks)
          
#     print("\nsave model :GRU_Dropout_coles jam_sales_predict_model.h5\n")
#     model.save("GRU_Dropout_coles_jam_sales_predict_model.h5", include_optimizer=True)
           
#     plot_learning_curves(history.history["loss"], history.history["val_loss"],epochs,"GRU and dropout:")
#     save_fig("GRU and dropout learning curve",images_path)
      
#     plt.show()

# else:
   
#     print("\nload model :GRU_Dropout_coles jam_sales_predict_model.h5\n")

#     model=keras.models.load_model("GRU_Dropout_coles_jam_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})




# #######################################################333
# scanned_sales=y_pred    #np.array([13400, 12132, 12846, 9522, 11858 ,13846 ,13492, 12310, 13584 ,13324, 15656 ,15878 ,13566, 10104 , 7704  ,7704])
# scanned_sales=scanned_sales.reshape(-1,1)[np.newaxis,...]

# print("scanned sales",scanned_sales,scanned_sales.shape,"[:,3,:]",scanned_sales[:,3,:])
# for t in range(scanned_sales.shape[1]-1):
#     print("predict",t,scanned_sales[:,t,:],"=",model(scanned_sales[:,t,:]))
# Y_pred=[np.stack(model(scanned_sales[:,r,:]).numpy(),axis=2) for r in range(scanned_sales.shape[1])]
# #print("Y_pred",Y_pred)
# Y_pred=np.concatenate((X_set[-1][np.newaxis,np.newaxis],np.array(Y_pred)[:,0,0]),axis=0)[:-1,0]
# print("Y_pred=",Y_pred)
# #########################################333
# fig, ax = plt.subplots()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
# locator = mdates.MonthLocator(interval=3)
# ax.xaxis.set_major_locator(locator)
# plt.xticks(rotation=0)
# ax.grid(axis='x')
# ax.plot_date(dates, X_set,"b-")
# ax.plot_date(dates,y_set,"r-")
# plt.title("X (DC purchases-blue) and y (scanned sales-red) series aligned at "+str(target_offset)+" weeks offset.")
# plt.legend(fontsize=10,loc='best')
# plt.show()
# ################################################
# #plt.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")

# fig, ax = plt.subplots()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
# locator = mdates.MonthLocator(interval=3)
# ax.xaxis.set_major_locator(locator)
# plt.xticks(rotation=0)
# ax.grid(axis='x')
# ax.plot_date(dates, X_set,"b-")
# ax.plot_date(dates,y_set,"r-")
# #ax.plot_date(dates,y_pred,"r:")

# # plt.plot(range(X_set.shape[0]),X_set,"b-")
# # plt.plot(range(y_set.shape[0]),y_set,"r-")
# #plt.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")
# #ax.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")
# ax.plot_date(pred_dates,y_pred,"r:")

# #plt.plot(range(X_set.shape[0]-2,X_set.shape[0]-2+Y_pred.shape[0]),Y_pred,"b-")

# plt.legend(fontsize=10,loc='best')
# #plt.title("X (DC purchases-blue) and y (scanned sales-red) series aligned at "+str(target_offset)+" weeks offset.")
# plt.title("X (DC purchases-blue) and y (scanned sales-red) offset with new scanned sales")
# plt.show()
 


 
# ################################
# print("Y_pred",Y_pred,Y_pred.shape)

# fig, ax = plt.subplots()
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
# locator = mdates.MonthLocator(interval=3)
# ax.xaxis.set_major_locator(locator)
# plt.xticks(rotation=0)
# ax.grid(axis='x')
# ax.plot_date(dates, X_set,"b-")
# #ax.plot_date(dates,y_set,"r-")
# ax.plot_date(pred_dates,y_pred,"r:")
# ax.plot_date(pred_dates,Y_pred,"b:")

# # plt.plot(range(X_set.shape[0]),X_set,"b-")
# # #plt.plot(range(y_set.shape[0]),y_set,"r-")
# # plt.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")
# # plt.plot(range(X_set.shape[0]-1,X_set.shape[0]-1+Y_pred.shape[0]),Y_pred,"b-")

# plt.legend(fontsize=10,loc='best')
# plt.title("Coles jam predicted purchases")
# plt.show()


# print("3Xset=",X_set,X_set.shape)
# print("3yset=",y_set,y_set.shape)
 
# print("scanned_sales=",scanned_sales)
# print("Y_pred=",Y_pred)      



############################################################
# load dataframe of sales_trans created by sales predict





def load_sales(filenames):  # filenames is a list of xlsx files to load and sort by date
    print("load:",filenames[0])
    df=pd.read_excel(filenames[0],sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   
    print("df size=",df.shape,df.columns)
    for filename in filenames[1:]:
        print("load:",filename)
        new_df=pd.read_excel(filename,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows  
        new_df['date'] = pd.to_datetime(new_df.date)  #, format='%d%m%Y')
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
    
    
 
    return df           
 


def plot_stacked_line_pivot(pivot_df,title,stacked=True,number=6):    
    pivot_df[pivot_df < 0] = np.nan
    

    pivot_df.drop('All',axis=0,inplace=True)
    pivot_df=pivot_df.sort_values(by=["All"],ascending=[True]).tail(number)
        
 #   print("before plot",pivot_df) 
    pivot_df.drop('All',axis=1,inplace=True)
        # remove last month as it will always be incomplete

    pivot_df.drop(pivot_df.columns[-1],axis=1,inplace=True)
    #dates=pivot_df.T.index.tolist()
    dates=pivot_df.columns.tolist()
    pivot_df=pivot_df.T
    pivot_df['dates']=dates
  #  print("plot pivot",pivot_df)
    #plt.legend(loc='best', fontsize=8)   #prop={'size': 6})
    return pivot_df.plot(rot=45,grid=True,logy=False,use_index=True,fontsize=8,kind='line',stacked=stacked,title=title+", stacked="+str(stacked))

    




pd.options.display.float_format = '{:.2f}'.format
sales_df_savename="sales_trans_df.pkl"
filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]


with open(sales_df_savename,"rb") as f:
    sales_df=pickle.load(f)

print("sales df=\n",sales_df,sales_df.shape)

first_date=sales_df['date'].iloc[-1]
last_date=sales_df['date'].iloc[0]

print("\nAttache sales trans analysis.  Current save is:")


print("\nData available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")
#print("\n\n")   

answer="y"
answer=input("Refresh salestrans?")
if answer=="y":
    sales_df=load_sales(filenames)  # filenames is a list of xlsx files to load and sort by date
  
    


   # start_date = pd.to_datetime(st.data_start_date) + pd.DateOffset(days=st.start_point)
   # end_date = pd.to_datetime(st.data_start_date) + pd.DateOffset(days=st.end_point)
#sales_df['period'].remove_categories(inplace=True)      
sales_df.sort_values(by=['date'],ascending=True,inplace=True)

dds=sales_df.groupby(['period'])['salesval'].sum().to_frame() 
datelen=dds.shape[0]-365
#print(dds)
# 365 day Moving total $ GSV
#daily_dollar_sales_df=sales_df['salesval'].groupby.index.sum()


def glset_GSV(dds,title):
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['diff1']=dds.mat.diff(periods=1)
    dds['diff7']=dds.mat.diff(periods=7)
    dds['diff30']=dds.mat.diff(periods=30)
    dds['diff365']=dds.mat.diff(periods=365)
    
    dds['7_day%']=round(dds['diff7']/dds['mat']*100,2)
    dds['30_day%']=round(dds['diff30']/dds['mat']*100,2)
    dds['365_day%']=round(dds['diff365']/dds['mat']*100,2)
    
    dds['date']=dds.index.tolist()
    dds.reset_index(inplace=True)
    #print(dds)
    dds.drop(['period'],axis=1,inplace=True)
    #print(dds)
    #dds=dds.tail(365)
    dds.tail(365)[['date','mat']].plot(x='date',y='mat',grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    print(dds[['date','salesval','diff7','30_day%','365_day%','mat']].tail(7)) 
    dds.tail(dds.shape[0]-731)[['date','30_day%','365_day%']].plot(x='date',y=['30_day%','365_day%'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')


print("\n$GSV sales progress")
glset_GSV(dds,"Beerenberg GSV Annual growth rate")

#########################################

print("\nshop sales $")
shop_df=sales_df[(sales_df['glset']=="SHP")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
glset_GSV(dds,"Shop GSV")

############################################

print("\nONL sales $")
shop_df=sales_df[(sales_df['glset']=="ONL")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
glset_GSV(dds,"ONL GSV")

############################################

print("\nExport sales $")
shop_df=sales_df[(sales_df['glset']=="EXS")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
glset_GSV(dds,"Export GSV")

############################################

print("\nNAT sales $")
shop_df=sales_df[(sales_df['glset']=="NAT")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
glset_GSV(dds,"NAT GSV")

############################################

print("\nWW sales $")
shop_df=sales_df[(sales_df['specialpricecat']==10)]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
glset_GSV(dds,"WW (010) GSV")

############################################

print("\nColes sales $")
shop_df=sales_df[(sales_df['specialpricecat']==12)]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
glset_GSV(dds,"Coles (012) GSV")

############################################


print("\nDFS sales $")
shop_df=sales_df[(sales_df['glset']=="DFS")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
glset_GSV(dds,"DFS GSV")

plt.show()
plt.close('all')
############################################

# create a sales_trans pivot table
print("\nSales summary (units or ctns)")
#print(sales_df.columns)
sales_df = sales_df.drop(sales_df[(sales_df['productgroup']==0)].index)

sales_df["month"] = pd.to_datetime(sales_df["date"]).dt.strftime('%m-%b')
sales_df["year"] = pd.to_datetime(sales_df["date"]).dt.strftime('%Y')
#sales_df=sales_df.sort_values(by=['year','month'],ascending=False)  #,inplace=True)

#sales_df=sales_df.drop((sales_df['productgroup']=='0'),axis=0)
#sales_df['col_date'] = sales_df['date'].format("%b-%Y")       #pd.to_datetime(sales_df['date'], infer_datetime_format=False, format='%b-%Y')
#print(sales_df)
pivot_df=pd.pivot_table(sales_df, values='qty', index=['productgroup','product'],columns=['year','month'], aggfunc=np.sum, margins=False,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
print(pivot_df) 
pivot_df.to_excel("pivot_table_units.xlsx") 

pivot_df=pd.pivot_table(sales_df, values='salesval', index=['productgroup','product'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
print(pivot_df) 
#pivot_df.plot(kind='line',stacked=True,title="Unit sales per month by productgroup")

pivot_df.to_excel("pivot_table_dollars.xlsx") 


#jam_sales_df=sales_df[sales_df['productgroup']==10]
#print("jsdf=\n",jam_sales_df)

pivot_df=pd.pivot_table(sales_df[sales_df['productgroup']<"40"], values='qty', index=['productgroup'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])

ax=plot_stacked_line_pivot(pivot_df,"Unit sales per month by productgroup",False)   #,number=6)

pivot_df.to_excel("pivot_table_dollars_product_group.xlsx") 



pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset','specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])

print(pivot_df) 
pivot_df.to_excel("pivot_table_customers_x_glset_x_spc.xlsx") 

pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])

print(pivot_df) 
pivot_df.to_excel("pivot_table_customers_x_glset.xlsx") 


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
ax=plot_stacked_line_pivot(pivot_df,"Dollar sales per month by spc",False)   #,number=6)

print(pivot_df) 
pivot_df.to_excel("pivot_table_customers_spc_nocodes.xlsx") 


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#ax=plot_stacked_line_pivot(pivot_df,"Unit sales per month by productgroup",False)   #,number=6)

print(pivot_df) 
pivot_df.to_excel("pivot_table_customers_x_spc.xlsx") 


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['cat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
print(pivot_df) 
pivot_df.to_excel("pivot_table_customers.xlsx") 

##############################################################33
# rank top customers and products
#
end_date=sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
#print(end_date)
#print("ysdf=",sales_df)
year_sales_df=sales_df[sales_df['date']>end_date]
#print("ysdf=",year_sales_df)
year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["code"]).transform(sum)
pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
#print("pv=",pivot_df)
unique_code_pivot_df=pivot_df.drop_duplicates('code',keep='first')
#unique_code_pivot_df=pd.unique(pivot_df['code'])
print("\nTop 50 customers by $purchases in the last 30 days.")
print(unique_code_pivot_df[['code','total_dollars']].head(50))
#pivot_df=pd.pivot_table(sales_df, values='tot3', index=['code'],columns=['year','month'], margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#print(pivot_df) 
#pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 

year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["specialpricecat"]).transform(sum)
pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
#print("pv=",pivot_df)
unique_code_pivot_df=pivot_df.drop_duplicates('specialpricecat',keep='first')
#unique_code_pivot_df=pd.unique(pivot_df['code'])
print("\nTop 50 customers special price category by $purchases in the last 30 days.")
print(unique_code_pivot_df[['specialpricecat','total_dollars']].head(50))







year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["product"]).transform(sum)
year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["product"]).transform(sum)

pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
#print("pv=",pivot_df)
unique_code_pivot_df=pivot_df.drop_duplicates('product',keep='first')

#unique_code_pivot_df=pd.unique(pivot_df['code'])
print("\nTop 50 products by $sales in the last 30 days.")
print(unique_code_pivot_df[['product','total_units','total_dollars']].head(50))
#pivot_df=pd.pivot_table(sales_df, values='tot3', index=['code'],columns=['year','month'], margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#print(pivot_df) 
#pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 




year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["productgroup"]).transform(sum)
year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["productgroup"]).transform(sum)
pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
unique_pg_pivot_df=pivot_df.drop_duplicates('productgroup',keep='first')

print("\nTop productgroups by $sales in the last 30 days.")
print(unique_pg_pivot_df[['productgroup','total_units','total_dollars']].head(20))
#pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 



print("\nTop 50 Credits in past 30 days")
end_date=sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
#print(end_date)
#print("ysdf=",sales_df)
month_sales_df=sales_df[sales_df['date']>end_date]
#print("msdf=",month_sales_df)
credit_df=month_sales_df[(month_sales_df['salesval']<-100) | (month_sales_df['qty']<-10)]
#print(credit_df.columns)
credit_df=credit_df.sort_values(by=["salesval"],ascending=[True])

print(credit_df.tail(50)[['date','code','glset','qty','salesval']])



print("finished\n")




