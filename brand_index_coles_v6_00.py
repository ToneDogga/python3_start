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

import warnings

from collections import namedtuple
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


colesscan="Coles_scan_data_300620.xlsx"
#latestscannedsalescoles="Coles_IRI_portal_scan_data_170620.xlsx"

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
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format

report = namedtuple("report", ["name", "report_type","cust","prod"])



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
 
col_dict=  {0:"Coles_scan_week",
            1:"Total_jam_curd_marm_off_promo_scanned",
            2:"Total_jam_curd_marm_on_promo_scanned",
            3:"BB_off_promo_scanned",
            4:"BB_on_promo_scanned",
            5:"SD_off_promo_scanned",
            6:"SD_on_promo_scanned",
            7:"BM_off_promo_scanned",
            8:"BM_on_promo_scanned",
            9:"BB_SJ300_off_promo_scanned",
            10:"BB_SJ300_on_promo_scanned",
            11:"BB_RJ300_off_promo_scanned",
            12:"BB_RJ300_on_promo_scanned",
            13:"BB_OM300_off_promo_scanned",
            14:"BB_OM300_on_promo_scanned",
            15:"BB_AJ300_off_promo_scanned",
            16:"BB_AJ300_on_promo_scanned",
            17:"BB_TC260_off_promo_scanned",
            18:"BB_TC260_on_promo_scanned",
            19:"BB_HTC260_off_promo_scanned",
            20:"BB_HTC260_on_promo_scanned",
            21:"BB_CAR280_off_promo_scanned",
            22:"BB_CAR280_on_promo_scanned",
            23:"BB_BBR280_off_promo_scanned",
            24:"BB_BBR280_on_promo_scanned",
            25:"BB_TS300_off_promo_scanned",
            26:"BB_TS300_on_promo_scanned",
            27:"BB_PCD300_off_promo_scanned",
            28:"BB_PCD300_on_promo_scanned",
            29:"BB_BLU300_off_promo_scanned",
            30:"BB_BLU300_on_promo_scanned",
            31:"BB_RAN300_off_promo_scanned",
            32:"BB_RAN300_on_promo_scanned"}
   
report_savename="sales_trans_report_dict.pkl"


# =============================================================================
# 
# pkl_dict={"all_coles_jams.pkl":("12","10","*"),   # special price cat, productgroup,productcode
#           "coles_SJ300.pkl":(12,10,"SJ300"),
#           "coles_AJ300.pkl":(12,10,"AJ300"),
#           "coles_OM300.pkl":(12,10,"OM300"),
#           "coles_RJ300.pkl":(12,10,"RJ300"),
#           "coles_TS300.pkl":(12,11,"TS300"),
#           "coles_CAR280.pkl":(12,13,"CAR280"),
#           "coles_BBR280.pkl":(12,13,"BBR280"),
#           "coles_TC260.pkl":(12,13,"TC260"),
#           "coles_HTC260.pkl":(12,13,"HTC260"),
#           "coles_PCD300.pkl":(12,14,"PCD300"),
#           "coles_BLU300.pkl":(12,14,"BLU300"),
#           "coles_RAN300.pkl":(12,14,"RAN300")}
# 
# =============================================================================


#rename_dict=dict({"qty":"BB_total_invoiced_sales"})
       
df=pd.read_excel(colesscan,-1,skiprows=[0,1,2]).T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
#print("before",df)

df.fillna(0.0,inplace=True)

df = df.rename(col_dict,axis='index').T
#print("df",df)
df['Coles_scan_week']=pd.to_datetime(df['Coles_scan_week'],format="%d/%m/%y")
#df['coles_scan_week']=df["date"] #.strftime("%Y-%m-%d")

df.drop_duplicates(keep='first', inplace=True)
#df.replace(0.0, np.nan, inplace=True)
#print("after",df)

df=df.sort_values(by=['Coles_scan_week'], ascending=True)
df=df.set_index('Coles_scan_week') 
df=df.astype(np.float32)  #,inplace=True)
df['weekno']= np.arange(len(df))
#print("final",df,df.T)

df['BB_on_promo']=(df['BB_on_promo_scanned']>0.0)
df['SD_on_promo']=(df['SD_on_promo_scanned']>0.0)
df['BM_on_promo']=(df['BM_on_promo_scanned']>0.0)

df['BB_total_scanned']=df['BB_off_promo_scanned']+df['BB_on_promo_scanned']
df['SD_total_scanned']=df['SD_off_promo_scanned']+df['SD_on_promo_scanned']
df['BM_total_scanned']=df['BM_off_promo_scanned']+df['BM_on_promo_scanned']

col_dict_list=list(col_dict.keys())
for key in col_dict_list[1::2]:
      #  print("key=",col_dict[key])
        df[col_dict[key][:9]+"_scan_total"]=df[col_dict[key]]+df[col_dict[key+1]]

print("df=\n",df.columns)
df.replace(0.0, np.nan, inplace=True)

print("df=",df,df.index)
plot_query(df,['BB_total_scanned','SD_total_scanned','BM_total_scanned'],'BB total scanned Coles jam units per week')



#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')  #,fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_scanned',data=df,col='SD_on_promo',hue='BB_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_scanned',data=df,col='BM_on_promo',hue='BB_on_promo')

sns.lmplot(x='weekno',y='BB_total_scanned',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='BB_total_scanned',data=df,col='SD_on_promo',hue='BM_on_promo')


# convert weekly scan data to daily sales

df.replace(np.nan, 0.0, inplace=True)



print(df)
df=df*1000
print(df)

output_dir = log_dir("SCBS2")
os.makedirs(output_dir, exist_ok=True)

images_path = os.path.join(output_dir, "images/")
os.makedirs(images_path, exist_ok=True)

####################################
# pkl_dict which is save in a dictionary of report_dict as a pickle
# pkl_dict contains a list of files names as keys to run as the actual sales in the prediction vs actual df
#

with open(report_savename,"rb") as f:
    report_dict=pickle.load(f)

#print("report dict=",report_dict.keys())
pkl_dict=report_dict[report('pkl_dict',0,"","")]
print("pkl dict=",pkl_dict)

###########################################3


joined_df=df.copy(deep=True)
for key in pkl_dict.keys():
   # savepkl="scanned_sales_plus_"+key
    print("Loading query dataframe:",key)
    with open(key,"rb") as f:
        actual_sales=pickle.load(f)

    actual_sales.reset_index(drop=True,inplace=True)  
    actual_sales.index=actual_sales.date
    actual_sales=actual_sales[['qty']]
    forecast_df = actual_sales.resample('W-SAT', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
    joined_df=pd.concat([joined_df,forecast_df],axis=1)
    joined_df=joined_df.rename(columns={"qty":key.rsplit( ".", 1 )[ 0 ]+"_invoiced"})
 
#print("df=",df)
print("joined_df=\n",joined_df)
print("joined_df.T=\n",joined_df.T)

plt.grid(True)
joined_df[['BB_total_invoiced','BB_total_scanned']].plot(title="Coles jam",grid=True)   #),'BB total scanned vs purchased Coles jam units per week')
mat=4
joined_df=joined_df.rolling(mat,axis=0).mean()

#joined_df['BB_scanned_sales']=joined_df['BB_scanned_sales'].rolling(mat,axis=0).mean()
joined_df[['BB_total_invoiced','BB_total_scanned']].plot(grid=True,title="Coles Jam units Moving total "+str(mat)+" weeks")   #),'BB total scanned vs purchased Coles jam units per week')
#plt.grid(True)
plt.show()
plt.close("all")


print("saving query dataframe:",savepkl)
pd.to_pickle(joined_df,savepkl)


# use Coles scan data from IRI weekly to predict Coles orders
#  X is the BB_total_sales
# the Target y is scanned sales 4 weeks ahead
# 
df=joined_df[['BB_total_invoiced','BB_total_scanned']]
print(df)

     # 
    # 
    # no of weeks
target_offset=3
batch_length=4
no_of_batches=1000
no_of_repeats=4
epochs=20
start_point=7
end_point=123
df.fillna(0,inplace=True)
# use a simple model
X_set=df['BB_total_invoiced'].iloc[target_offset:].to_numpy()
X_set=np.concatenate((X_set,np.zeros(target_offset)),axis=0).astype(np.int32)
y_set=df['BB_total_scanned'].iloc[start_point:end_point].to_numpy().astype(np.int32)

dates=df[start_point:end_point].index.tolist()
pred_dates=df[end_point-1:].index.tolist()

print("1Xset=",X_set,X_set.shape)
print("1yset=",y_set,y_set.shape)


X_set=X_set[start_point:end_point]
y_pred=df['BB_total_scanned'].iloc[end_point-1:].to_numpy().astype(np.int32)
print("2Xset=",X_set,X_set.shape)
print("2yset=",y_set,y_set.shape)

print("y_pred=",y_pred,y_pred.shape)


###############################
# batches of X shape (no of batches,batch length, 1)
# batches of Y shape (no of batches,batch length, 1)

answer=input("Train a model?")
if answer=="y":
    
    
    X,y=create_X_and_y_batches(X_set,y_set,batch_length,no_of_batches)
    
    
    
    ##########################3
    
    dataset=tf.data.Dataset.from_tensor_slices((X,y)).cache().repeat(no_of_repeats)
    dataset=dataset.shuffle(buffer_size=1000,seed=42)
    
    train_set=dataset.batch(1).prefetch(1)
    valid_set=dataset.batch(1).prefetch(1)
       
     
    
    ##########################
    print("\nTraining with GRU and dropout")
    model = keras.models.Sequential([
    #     keras.layers.Conv1D(filters=st.batch_length,kernel_size=4, strides=1, padding='same', input_shape=[None, 1]),  #st.batch_length]), 
      #   keras.layers.BatchNormalization(),
         keras.layers.GRU(200, return_sequences=True, input_shape=[None, 1]), #st.batch_length]),
        # keras.layers.BatchNormalization(),
         keras.layers.GRU(200, return_sequences=True),
       #  keras.layers.AlphaDropout(rate=0.2),
       #  keras.layers.BatchNormalization(),
         keras.layers.TimeDistributed(keras.layers.Dense(1))
    ])
      
    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
     
    model.summary() 
     
    #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),self.MyCustomCallback()]
     
    history = model.fit(train_set ,epochs=epochs,
                         validation_data=(valid_set))  #, callbacks=callbacks)
          
    print("\nsave model :GRU_Dropout_coles jam_sales_predict_model.h5\n")
    model.save("GRU_Dropout_coles_jam_sales_predict_model.h5", include_optimizer=True)
           
    plot_learning_curves(history.history["loss"], history.history["val_loss"],epochs,"GRU and dropout:")
    save_fig("GRU and dropout learning curve",images_path)
      
    plt.show()

else:
   
    print("\nload model :GRU_Dropout_coles jam_sales_predict_model.h5\n")

    model=keras.models.load_model("GRU_Dropout_coles_jam_sales_predict_model.h5",custom_objects={"last_time_step_mse": last_time_step_mse})




#######################################################333
scanned_sales=y_pred    #np.array([13400, 12132, 12846, 9522, 11858 ,13846 ,13492, 12310, 13584 ,13324, 15656 ,15878 ,13566, 10104 , 7704  ,7704])
scanned_sales=scanned_sales.reshape(-1,1)[np.newaxis,...]

print("scanned sales",scanned_sales,scanned_sales.shape,"[:,3,:]",scanned_sales[:,3,:])
for t in range(scanned_sales.shape[1]-1):
    print("predict",t,scanned_sales[:,t,:],"=",model(scanned_sales[:,t,:]))
Y_pred=[np.stack(model(scanned_sales[:,r,:]).numpy(),axis=2) for r in range(scanned_sales.shape[1])]
#print("Y_pred",Y_pred)
Y_pred=np.concatenate((X_set[-1][np.newaxis,np.newaxis],np.array(Y_pred)[:,0,0]),axis=0)[:-1,0]
print("Y_pred=",Y_pred)
#########################################333
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
locator = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(locator)
plt.xticks(rotation=0)
ax.grid(axis='x')
ax.plot_date(dates, X_set,"b-")
ax.plot_date(dates,y_set,"r-")
plt.title("X (DC purchases-blue) and y (scanned sales-red) series aligned at "+str(target_offset)+" weeks offset.")
plt.legend(fontsize=10,loc='best')
plt.show()
################################################
#plt.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
locator = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(locator)
plt.xticks(rotation=0)
ax.grid(axis='x')
ax.plot_date(dates, X_set,"b-")
ax.plot_date(dates,y_set,"r-")
#ax.plot_date(dates,y_pred,"r:")

# plt.plot(range(X_set.shape[0]),X_set,"b-")
# plt.plot(range(y_set.shape[0]),y_set,"r-")
#plt.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")
#ax.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")
ax.plot_date(pred_dates,y_pred,"r:")

#plt.plot(range(X_set.shape[0]-2,X_set.shape[0]-2+Y_pred.shape[0]),Y_pred,"b-")

plt.legend(fontsize=10,loc='best')
#plt.title("X (DC purchases-blue) and y (scanned sales-red) series aligned at "+str(target_offset)+" weeks offset.")
plt.title("X (DC purchases-blue) and y (scanned sales-red) offset with new scanned sales")
plt.show()
 


 
################################
print("Y_pred",Y_pred,Y_pred.shape)

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
locator = mdates.MonthLocator(interval=3)
ax.xaxis.set_major_locator(locator)
plt.xticks(rotation=0)
ax.grid(axis='x')
ax.plot_date(dates, X_set,"b-")
#ax.plot_date(dates,y_set,"r-")
ax.plot_date(pred_dates,y_pred,"r:")
ax.plot_date(pred_dates,Y_pred,"b:")

# plt.plot(range(X_set.shape[0]),X_set,"b-")
# #plt.plot(range(y_set.shape[0]),y_set,"r-")
# plt.plot(range(y_set.shape[0]-1,y_set.shape[0]-1+y_pred.shape[0]),y_pred,"r:")
# plt.plot(range(X_set.shape[0]-1,X_set.shape[0]-1+Y_pred.shape[0]),Y_pred,"b-")

plt.legend(fontsize=10,loc='best')
plt.title("Coles jam predicted purchases")
plt.show()


print("3Xset=",X_set,X_set.shape)
print("3yset=",y_set,y_set.shape)
 
print("scanned_sales=",scanned_sales)
print("Y_pred=",Y_pred)      


