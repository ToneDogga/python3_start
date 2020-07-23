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


from pathlib import Path,WindowsPath


import pickle
import multiprocessing

import warnings

from collections import namedtuple
from collections import defaultdict
from datetime import datetime
from pandas.plotting import scatter_matrix

import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
#plt.ion() # enables interactive mode

#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


#colesjamsxls="Coles_scan_data_300620.xlsx"
#latestscannedsalescoles="Coles_IRI_portal_scan_data_170620.xlsx"
stock_level_query='stock_level_query.xlsx'
production_made_query='Production Schedule.xlsx'
production_planned_query='#B Stock & Schedule Forecast.xlsx'
report_savename="sales_trans_report_dict.pkl"
colesscan="Coles_scan_data_300620.xlsx"
woolscan="Ww_scan_data_300620.xlsx"
  #  report_savename="sales_trans_report_dict.pkl"
savepkl="coles_invoiced_and_scanned_sales.pkl"


sales_df_savename="sales_trans_df.pkl"
filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]
   
# output_dir = log_dir("scandata")
# os.makedirs(output_dir, exist_ok=True)

# images_path = os.path.join(output_dir, "images/")
# os.makedirs(images_path, exist_ok=True)

product_groups_only=["10","11","12","13","14","15","18"]
spc_only=["088"]   #,"038","048","028","080","020","030","040"]

mats=7

scan_data_files=["jam_scan_data_2020.xlsx","cond_scan_data_2020.xlsx","sauce_scan_data_2020.xlsx"]
scan_dict_savename="scan_dict.pkl"


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./dashboard2_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)



output_dir = log_dir("dashboard2")
os.makedirs(output_dir, exist_ok=True)

#images_path = os.path.join(output_dir, "images/")
#os.makedirs(images_path, exist_ok=True)

 
# mats=7

# #         self.start_point=0
        
  

# def save_fig(fig_id, images_path, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(images_path, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)

  
   
   
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

    #  tf.print("2X[1]=",X[1],X.shape,"\n")
    #  tf.print("2y[1]=",y[1],y.shape,"\n")

      return X,y




def get_xs_name(df,filter_tuple):
    #  returns a slice of the multiindex df with a tuple (column value,index_level) 
    # col_value itselfcan be a tuple, col_level can be a list
    # levels are (brand,specialpricecat, productgroup, product,name) 
    #
  #  print("get_xs_name df index",df.columns,df.columns.nlevels)
    if df.columns.nlevels>=2:

        df=df.xs(filter_tuple[0],level=filter_tuple[1],drop_level=False,axis=1)
    #df=df.T
   #     print("2get_xs_name df index",df.columns,df.columns.nlevels)
        if df.columns.nlevels>=2:
            for _ in range(df.columns.nlevels-1):
                df=df.droplevel(level=0,axis=1)
    
    else:
        print("not a multi index df columns=",df,df.columns)    
    return df






def predict_order(hdf,title,model):
    hdf['lastdate'] = pd.to_datetime(hdf.index,format="%Y-%m-%d",exact=False)

    latest_date = hdf['lastdate'].max()

  
    y_set=hdf.iloc[:,2].to_numpy().astype(np.int32)[7:-1]
    scanned_sales=hdf.iloc[:,0].to_numpy().astype(np.int32)[7:-1]     #np.array([13400, 12132, 12846, 9522, 11858 ,13846 ,13492, 12310, 13584 ,13324, 15656 ,15878 ,13566, 10104 , 7704  ,7704])
    scanned_sales=scanned_sales.reshape(-1,1)[np.newaxis,...]
    
    #print("scanned sales",scanned_sales,scanned_sales.shape,"#[:,-2:,:]",scanned_sales[:,-2:,:])
    #for t in range(scanned_sales.shape[1]-1):
    #    print("predict",t,scanned_sales[:,t,:],"=",model(scanned_sales[:,t,:]))
    Y_pred=np.stack(model(scanned_sales[:,-1,:]).numpy(),axis=2) #for r in range(scanned_sales.shape[1])]
    #print("Y_pred",Y_pred,Y_pred.shape)
    Y_pred=np.concatenate((y_set,Y_pred[0,:,0]),axis=0)
    #print("Y_pred=",Y_pred,Y_pred.shape)
    #Y_pred=np.roll(Y_pred,-3)
    #Y_pred[-3:]=0
    #print("Y_pred=",Y_pred,Y_pred.shape)
    
    #########################################333
    X_pred=hdf[title+'_total_scanned'].to_numpy().astype(np.int32)[7:]
    y_invoiced=hdf[title+'_invoiced_shifted_3wks'].to_numpy().astype(np.int32)[7:]
    
    #print("X_pred=\n",X_pred,X_pred.shape)
    #dates=hdf.shift(1,freq="W").index.tolist()[7:]
    dates=hdf.index.tolist()[7:]
    #print("dates:",dates,len(dates))
    df=pd.DataFrame({title+'_total_scanned':X_pred,title+'_ordered_prediction':Y_pred,title+'_total_invoiced_shifted_3wks':y_invoiced},index=dates)
   # df=pd.DataFrame({title+'_total_scanned':X_pred,title+'_ordered_prediction':Y_pred},index=dates)
 
    #shifted_df=df.shift(1, freq='W')   #[:-3]   # 3 weeks
    
    #df=gdf[['coles_BB_jams_total_scanned','all_BB_coles_jams_predicted']].rolling(mat,axis=0).mean()
    
  #  styles1 = ['b-','r:']
    styles1 = ['b-','g:','r:']
           # styles1 = ['bs-','ro:','y^-']
    linewidths = 1  # [2, 1, 4]
   # print("df=\n",df,df.shape)
    df.iloc[-26:].plot(grid=True,title=title+" w/c:("+str(latest_date)+")",style=styles1, lw=linewidths)
    #plt.pause(0.001)
    
    #df.iloc[-6:].plot(grid=True,title=title,style=styles1, lw=linewidths)
    #plt.pause(0.001)
    #ax.legend(title="")
    #plt.ax.show()
    
    #df=df.rolling(mat,axis=0).mean()
    #df=df[100:]
    
    #ax=df.plot(grid=True,title="Coles units moving total "+str(mat)+" weeks",style=styles1, lw=linewidths)
    #ax.legend(title="")
    #plt.show()
    
    
    save_fig(title+"_order_predictions")   #,images_path)
      
   # plt.show()

    #print(df)
    plt.close("all")
    return df




def train_model(name,X_set,y_set,batch_length,no_of_batches):
   
    X,y=create_X_and_y_batches(X_set,y_set,batch_length,no_of_batches)
    
    
    
    ##########################3
    
    dataset=tf.data.Dataset.from_tensor_slices((X,y)).cache().repeat(no_of_repeats)
    dataset=dataset.shuffle(buffer_size=1000,seed=42)
    
    train_set=dataset.batch(1).prefetch(1)
    valid_set=dataset.batch(1).prefetch(1)
       
     
    
    ##########################
    print("\nTraining with GRU :",name)
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
          
    print("\nsave model :"+name+"_predict_model.h5\n")
    model.save(output_dir+name+"_sales_predict_model.h5", include_optimizer=True)
           
    plot_learning_curves(history.history["loss"], history.history["val_loss"],epochs,"GRU :"+name)
    save_fig(name+"GRU learning curve")  #,images_path)
      
  #  plt.show()
    plt.close("all")
    return model






# def get_xs_name(df,filter_tuple):
#     #  returns a slice of the multiindex df with a tuple (column value,index_level) 
#     # col_value itselfcan be a tuple, col_level can be a list
#     # levels are (brand,specialpricecat, productgroup, product,name) 
#     #
#   #  print("get_xs_name df index",df.columns,df.columns.nlevels)
#     if df.columns.nlevels>=2:

#         df=df.xs(filter_tuple[0],level=filter_tuple[1],drop_level=False,axis=1)
#     #df=df.T
#    #     print("2get_xs_name df index",df.columns,df.columns.nlevels)
#         if df.columns.nlevels>=2:
#             for _ in range(df.columns.nlevels-1):
#                 df=df.droplevel(level=0,axis=1)
    
#     else:
#         print("not a multi index df columns=",df,df.columns)    
#     return df



def get_xs_name2(df,f,l):
    #  returns a slice of the multiindex df with a tuple (column value,index_level) 
    # col_value itselfcan be a tuple, col_level can be a list
    # levels are (brand,specialpricecat, productgroup, product,name) 
    #
  #  print("get_xs_name df index",df.columns,df.columns.nlevels)
    if df.columns.nlevels>=2:

        df=df.xs(f,level=l,drop_level=False,axis=1)
    #df=df.T
   #     print("2get_xs_name df index",df.columns,df.columns.nlevels)
        if df.columns.nlevels>=2:
            for _ in range(df.columns.nlevels-1):
                df=df.droplevel(level=0,axis=1)
    
    else:
        print("not a multi index df columns=",df,df.columns)    
    return df





def add_dates_back(df,all_dates):
    return pd.concat((df,all_dates_df),axis=1)   #,on='ww_scan_week',how='right')






def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(output_dir, fig_id + "." + fig_extension)
  #  print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    return







def add_dates_back(df,all_dates):
    return pd.concat((df,all_dates_df),axis=1)   #,on='ww_scan_week',how='right')


def plot_query2(query_df_passed,plot_col,query_name):
    query_df=query_df_passed.copy(deep=True)
  #  query_details=str(query_df.columns[0])
  #  query_df.columns = query_df.columns.get_level_values(0)
    print("query df shape",query_df.shape)
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
  #  plt.draw()
  #  plt.pause(0.001)
  #  plt.show(block=False)

  




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
   # +" w/c:("+str(latest_date)+")"
    
    df.fillna(0,inplace=True)
    
    #print(df)
    print("drop duplicates")
    df.drop_duplicates(keep='first', inplace=True)
    print("after drop duplicates df size=",df.shape)
    print("sort by date",df.shape[0],"records.\n")
    df.sort_values(by=['date'], inplace=True, ascending=False)
      
    print(df.head(3))
    print(df.tail(3))
   
 
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
    figname="fig_3_"+title
    fig=pivot_df.plot(rot=45,grid=True,logy=False,use_index=True,fontsize=8,kind='line',stacked=stacked,title=title)
    save_fig(figname)
#    pickle.dump(fig,open(output_dir+figname+".pkl", 'wb'))
 #   plt.draw()
#    plt.pause(0.001)
#    plt.show(block=False)
    return figname
    



#def annualise_growth_rate(days,rate):
#    return (((1+rate)**(365/days))-1.0)




def plot_trend(s,title,slope,latest_date):
   #  ax=s[['days_since_last_order','units']].iloc[-1].plot(x='days_since_last_order', linestyle='None', color="green", marker='.')

     fig=s[['days_since_last_order','units']].iloc[:-1].plot(x='days_since_last_order', linestyle='None', color="red", marker='o')

     s[['days_since_last_order','bestfit']].plot(x='days_since_last_order',kind="line",ax=fig)

     plt.title(title+" (slope="+str(round(slope,3))+") w/c:("+str(latest_date)+")")  #str(new_plot_df.columns.get_level_values(0)))
     fig.legend(fontsize=8)
     plt.ylabel("unit sales")
     plt.grid(True)
#     self.save_fig("actual_v_prediction_"+str(plot_number_df.columns[0]),self.images_path)
     figname="fig_2_"+title

     save_fig(figname)


  #   pickle.dump(fig,open(output_dir+figname+".pkl", 'wb'))
   #  plt.draw()
   #  plt.pause(0.001)
   #  plt.show(block=False)
     return figname





def calculate_first_derivative(s,cust,prod,latest_date):

    s=s.sort_values('date',ascending=True)
    lastone=s.iloc[-1]
    newgap=pd.Timedelta(pd.to_datetime('today') -lastone['date']).days
    s=s.append(lastone)
  #  s['daysdiff'].iloc[-1]=newgap
  #  s['days'].iloc[-1]=newgap
    s['qty'].iloc[-1]=0.0
    s['date'].iloc[-1]=pd.to_datetime('today')

    s=s.sort_values('date',ascending=False)

    s['daysdiff']=s['date'].diff(periods=1).dt.days
    s['daysdiff'].fillna(0,inplace=True)

    s['days']=s['daysdiff'].cumsum()

 
    s.index =  pd.to_datetime(s['date'], format='%Y%m%d') 
    X=s[['days']].to_numpy()
    X=X[::-1,0]
  #  X=X[:,0]

    s['days_since_last_order']=X
    y=s[['qty']].to_numpy()   
    y=y[::-1,0]
    s['units']=y
   
    p = np.polyfit(X[:-1], y[:-1], 1)  # linear regression 1 degree
    
    s['bestfit']=np.polyval(p, X)
    figname=""
    title=""
    slope=round(p[0],6)
    if ((slope>0.06) | (slope<-0.1)):
        title=cust+"_"+prod
        figname= plot_trend(s,title,slope,latest_date)
    return slope,figname,title



def glset_GSV(dds,title):
#    dds.index =  pd.to_datetime(dds['date'], format='%Y%m%d') 

    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['mat7']=dds['salesval'].rolling(7,axis=0).sum()
    dds['mat14']=dds['salesval'].rolling(14,axis=0).sum()
    dds['mat30']=dds['salesval'].rolling(30,axis=0).sum()
    dds['mat90']=dds['salesval'].rolling(90,axis=0).sum()

  #  dds['diff1']=dds.mat.diff(periods=1)
    dds['diff7']=dds.mat7.diff(periods=7)
    dds['diff14']=dds.mat14.diff(periods=14)
    dds['diff30']=dds.mat30.diff(periods=30)
    dds['diff90']=dds.mat90.diff(periods=90)
    dds['diff365']=dds.mat.diff(periods=365)
    
    dds['7_day%']=round(dds['diff7']/dds['mat7']*100,2)
    dds['14_day%']=round(dds['diff14']/dds['mat14']*100,2)
    dds['30_day%']=round(dds['diff30']/dds['mat30']*100,2)
    dds['90_day%']=round(dds['diff90']/dds['mat90']*100,2)
    dds['365_day%']=round(dds['diff365']/dds['mat']*100,2)
    
    dds['date']=dds.index.tolist()
    latest_date=dds['date'].max()
    dds.reset_index(inplace=True)
    #print(dds)
    dds.drop(['period'],axis=1,inplace=True)
    #print(dds)
    #dds=dds.tail(365)
    dds.tail(365)[['date','mat']].plot(x='date',y='mat',grid=True,title=title+" w/c:("+str(latest_date)+")")   #),'BB total scanned vs purchased Coles jam units per week')
    print(dds[['date','mat7','diff7','30_day%','90_day%','365_day%','mat']].tail(8)) 
    fig=dds.tail(dds.shape[0]-731)[['date','30_day%','90_day%','365_day%']].plot(x='date',y=['30_day%','90_day%','365_day%'],grid=True,title=title+" w/c:("+str(latest_date)+")")   #),'BB total scanned vs purchased Coles jam units per week')
    figname="Afig_"+title
    save_fig(figname)
 #   pickle.dump(fig,open(output_dir+figname+".pkl", 'wb'))
    return dds[['date','mat7','diff7','30_day%','90_day%','365_day%','mat']].tail(18),figname
    



 

    
    
#def main():   
#if True:          
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format
  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors


print("\n\n\nDashboard2 : TF2 Salestrans analyse/predict - By Anthony Paech 25/5/20")
print("==============================================================================\n")       

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

 
print("\n=============================================================================\n")       


   
np.random.seed(42)
tf.random.set_seed(42)
  
##############################################################################
   #report = namedtuple("report", ["name", "report_type","cust","prod"])
 
report = namedtuple("report", ["name", "report_type","cust","prod"])

report_type_dict={0:"dictionary",
                       3:"dataframe",
                       5:"spreadsheet",
                       6:"pivottable",
                       8:"chart_filename"}



# value is (brand,specialpricecat, productgroup, product,type,name)
# type is 0 off promo. 1 is on promo, 2 is total of both, 3 is invoiced total

product_type = namedtuple("product_type", ["brandno","customercat", "productgroup","product","type","name"])

coles_pkl_dict={"coles_BB_jams_invoiced.pkl":(1,12,10,"_T",3,"coles_BB_jams_invoiced"),   # special price cat, productgroup,productcode,product, on_promo, name)
          "coles_BB_SJ300_invoiced.pkl":(1,12,10,"SJ300",3,"coles_BB_SJ300_invoiced"),
          "coles_BB_AJ300_invoiced.pkl":(1,12,10,"AJ300",3,"coles_BB_AJ300_invoiced"),
          "coles_BB_OM300_invoiced.pkl":(1,12,10,"OM300",3,"coles_BB_OM300_invoiced"),
          "coles_BB_RJ300_invoiced.pkl":(1,12,10,"RJ300",3,"coles_BB_RJ300_invoiced"),
          "coles_BB_TS300_invoiced.pkl":(1,12,11,"TS300",3,"coles_BB_TS300_invoiced"),
          "coles_BB_CAR280_invoiced.pkl":(1,12,13,"CAR280",3,"coles_BB_CAR280_invoiced"),
          "coles_BB_BBR280_invoiced.pkl":(1,12,13,"BBR280",3,"coles_BB_BBR280_invoiced"),
          "coles_BB_TC260_invoiced.pkl":(1,12,13,"TC260",3,"coles_BB_TC260_invoiced"),
          "coles_BB_HTC260_invoiced.pkl":(1,12,13,"HTC260",3,"coles_BB_HTC260_invoiced"),
          "coles_BB_PCD300_invoiced.pkl":(1,12,14,"PCD300",3,"coles_BB_PCD300_invoiced"),
          "coles_BB_BLU300_invoiced.pkl":(1,12,14,"BLU300",3,"coles_BB_BLU300_invoiced"),
          "coles_BB_RAN300_invoiced.pkl":(1,12,14,"RAN300",3,"coles_BB_RAN300_invoiced")}
          

   
# value is (brand,specialpricecat, productgroup, product,name)
product_type = namedtuple("product_type", ["brandno","customercat", "productgroup","product","on_promo","name"])

coles_col_dict=  {"scan_week":(0,12,0,'_*',0,'scan_week'),
            1:(0,12,10,"_*",0,"coles_total_jam_curd_marm_off_promo_scanned"),
            2:(0,12,10,"_*",1,"coles_total_jam_curd_marm_on_promo_scanned"),
            3:(1,12,10,"_*",0,"coles_BB_jams_off_promo_scanned"),
            4:(1,12,10,"_*",1,"coles_BB_jams_on_promo_scanned"),
            5:(2,12,10,"_*",0,"coles_SD_jams_off_promo_scanned"),
            6:(2,12,10,"_*",1,"coles_SD_jams_on_promo_scanned"),
            7:(3,12,10,"_*",0,"coles_BM_jams_off_promo_scanned"),
            8:(3,12,10,"_*",1,"coles_BM_jams_on_promo_scanned"),
            9:(1,12,10,"SJ300",0,"coles_BB_SJ300_off_promo_scanned"),
            10:(1,12,10,"SJ300",1,"coles_BB_SJ300_on_promo_scanned"),
            11:(1,12,10,"RJ300",0,"coles_BB_RJ300_off_promo_scanned"),
            12:(1,12,10,"RJ300",1,"coles_BB_RJ300_on_promo_scanned"),
            13:(1,12,10,"OM300",0,"coles_BB_OM300_off_promo_scanned"),
            14:(1,12,10,"OM300",1,"coles_BB_OM300_on_promo_scanned"),
            15:(1,12,10,"AJ300",0,"coles_BB_AJ300_off_promo_scanned"),
            16:(1,12,10,"AJ300",1,"coles_BB_AJ300_on_promo_scanned"),
            17:(1,12,13,"TC260",0,"coles_BB_TC260_off_promo_scanned"),
            18:(1,12,13,"TC260",1,"coles_BB_TC260_on_promo_scanned"),
            19:(1,12,13,"HTC260",0,"coles_BB_HTC260_off_promo_scanned"),
            20:(1,12,13,"HTC260",1,"coles_BB_HTC260_on_promo_scanned"),
            21:(1,12,13,"CAR280",0,"coles_BB_CAR280_off_promo_scanned"),
            22:(1,12,13,"CAR280",1,"coles_BB_CAR280_on_promo_scanned"),
            23:(1,12,13,"BBR280",0,"coles_BB_BBR280_off_promo_scanned"),
            24:(1,12,13,"BBR280",1,"coles_BB_BBR280_on_promo_scanned"),
            25:(1,12,11,"TS300",0,"coles_BB_TS300_off_promo_scanned"),
            26:(1,12,11,"TS300",1,"coles_BB_TS300_on_promo_scanned"),
            27:(1,12,14,"PCD300",0,"coles_BB_PCD300_off_promo_scanned"),
            28:(1,12,14,"PCD300",1,"coles_BB_PCD300_on_promo_scanned"),
            29:(1,12,14,"BLU300",0,"coles_BB_BLU300_off_promo_scanned"),
            30:(1,12,14,"BLU300",1,"coles_BB_BLU300_on_promo_scanned"),
            31:(1,12,14,"RAN300",0,"coles_BB_RAN300_off_promo_scanned"),
            32:(1,12,14,"RAN300",1,"coles_BB_RAN300_on_promo_scanned")}
 

coles_convert_dict = {'scan_week': np.datetime64, 
                1: np.float64,
                2: np.float64,
                3: np.float64,
                4: np.float64,
                5: np.float64,
                6: np.float64,
                7: np.float64,
                8: np.float64,
                9: np.float64,
                10: np.float64,
                11: np.float64,
                12: np.float64,
                13: np.float64,
                14: np.float64,
                15: np.float64,
                16: np.float64,
                17: np.float64,
                18: np.float64,
                19: np.float64,
                20: np.float64,
                21: np.float64,
                22: np.float64,
                23: np.float64,
                24: np.float64,
                25: np.float64,
                26: np.float64,
                27: np.float64,
                28: np.float64,
                29: np.float64,
                30: np.float64,
                31: np.float64,
                32: np.float64
                } 

  
ww_pkl_dict={}
# "ww_BB_jams_invoiced.pkl":(1,12,10,"_T",3,"ww_BB_jams_invoiced"),   # special price cat, productgroup,productcode,product, on_promo, name)
#           "ww_BB_SJ300_invoiced.pkl":(1,10,10,"SJ300",3,"ww_BB_SJ300_invoiced"),
#           "w_BB_AJ300_invoiced.pkl":(1,10,10,"AJ300",3,"ww_BB_AJ300_invoiced"),
#           "ww_BB_OM300_invoiced.pkl":(1,10,10,"OM300",3,"ww_BB_OM300_invoiced"),
#           "ww_BB_RJ300_invoiced.pkl":(1,10,10,"RJ300",3,"ww_BB_RJ300_invoiced"),
#           "ww_BB_TS300_invoiced.pkl":(1,10,11,"TS300",3,"ww_BB_TS300_invoiced"),
#           "ww_BB_CAR280_invoiced.pkl":(1,10,13,"CAR280",3,"ww_BB_CAR280_invoiced"),
#           "ww_BB_BBR280_invoiced.pkl":(1,10,13,"BBR280",3,"ww_BB_BBR280_invoiced"),
#           "ww_BB_TC260_invoiced.pkl":(1,10,13,"TC260",3,"ww_BB_TC260_invoiced"),
#           "ww_BB_HTC260_invoiced.pkl":(1,10,13,"HTC260",3,"ww_BB_HTC260_invoiced"),
#           "ww_BB_PCD300_invoiced.pkl":(1,10,14,"PCD300",3,"ww_BB_PCD300_invoiced"),
#           "ww_BB_BLU300_invoiced.pkl":(1,10,14,"BLU300",3,"ww_BB_BLU300_invoiced"),
#           "ww_BB_RAN300_invoiced.pkl":(1,10,14,"RAN300",3,"ww_BB_RAN300_invoiced")}
   


ww_col_dict={"scan_week":(0,10,0,'_*',0,'scan_week'),
                1:(1,10,10,"_*",0,"WW_BB_jams_off_promo_scanned"),
                2:(1,10,10,"_*",1,"WW_BB_jams_on_promo_scanned"),
                3:(2,10,10,"_*",0,"WW_SD_jams_off_promo_scanned"),
                4:(2,10,10,"_*",1,"WW_SD_jams_on_promo_scanned"),
                5:(3,10,10,"_*",0,"WW_BM_jams_off_promo_scanned"),
                6:(3,10,10,"_*",1,"WW_BM_jams_on_promo_scanned")}

 #           1:(0,10,10,"_*",0,"WW_total_jam_curd_marm_off_promo_scanned"),
 #           2:(0,10,10,"_*",1,"WW_total_jam_curd_marm_on_promo_scanned"),
  


ww_convert_dict = {'scan_week': np.datetime64, 
                1: np.float64,
                2: np.float64,
                3: np.float64,
                4: np.float64,
                5: np.float64,
                6: np.float64}



  




#  Report_dict is a dictionary of all the reports created plus the report_type_dict to decode it
# at the end it is picked so it can be loaded

report_dict={report("report_type_dict",0,"",""):report_type_dict,
             report("coles_pkl_dict",0,"",""):coles_pkl_dict,
             report("ww_pkl_dict",0,"",""):ww_pkl_dict}



# prediction hyper-parameters
batch_length=4
no_of_batches=1000
no_of_repeats=4
epochs=8







try:
    with open("stock_level_query.pkl","rb") as f:
       stock_df=pickle.load(f)
except:  
    print("load:",stock_level_query)
    stock_df=pd.read_excel(stock_level_query)    # -1 means all rows   
    with open("stock_level_query.pkl","wb") as f:
        pickle.dump(stock_df, f,protocol=-1)
#except:
#    pass        
#print("stock df size=",stock_df.shape,stock_df.columns)
#
    
stock_df['end_date']=stock_df['lastsalesdate']+ pd.Timedelta(90, unit='d')
#print(end_date)
#print("ysdf=",sales_df)
stock_df['recent']=stock_df['end_date']>pd.Timestamp('today')



#print("stock_df=\n",stock_df)
stock_df=stock_df[(stock_df['qtyinstock']<=100) & (stock_df['recent']==True)]
                
stock_report_df=stock_df[['code','lastsalesdate','qtyinstock']].sort_values('lastsalesdate',ascending=True)


print("Out of stock report\n",stock_report_df)

#####################################

try:
    with open("production_made.pkl","rb") as f:
       production_made_df=pickle.load(f)
except:  
    print("load:",production_made_query)
    production_made_df=pd.read_excel(production_made_query)    # -1 means all rows   
    with open("production_made.pkl","wb") as f:
        pickle.dump(production_made_df, f,protocol=-1)
#print("stock df size=",stock_df.shape,stock_df.columns)
#
    
#stock_report_df=stock_df[['code','lastsalesdate','qtyinstock']].sort_values('lastsalesdate',ascending=True)
#print(production_schedule_df.columns)
production_made_df=production_made_df[['to_date','jobid','code','qtybatches','qtyunits']].sort_values('to_date',ascending=True)
print("\nProduction recently made:\n",production_made_df.tail(30))





###############################

try:
    with open("production_planned.pkl","rb") as f:
       production_planned_df=pickle.load(f)
except:  
    print("load:",production_planned_query)
    production_planned_df=pd.read_excel(production_planned_query)    # -1 means all rows   
    with open("production_planned.pkl","wb") as f:
        pickle.dump(production_planned_df, f,protocol=-1)
#print("stock df size=",stock_df.shape,stock_df.columns)
#
production_planned_df['future']=production_planned_df['to_date']>=pd.Timestamp('today')
production_planned_df=production_planned_df[(production_planned_df['future']==True)]
                
#stock_report_df=stock_df[['code','lastsalesdate','qtyinstock']].sort_values('lastsalesdate',ascending=True)
  
#stock_report_df=stock_df[['code','lastsalesdate','qtyinstock']].sort_values('lastsalesdate',ascending=True)
#print(production_schedule_df.columns)
production_planned_df=production_planned_df[['to_date','jobid','code','qtybatches','qtyunits']].sort_values('to_date',ascending=True)
print("\nProduction planned:\n",production_planned_df.head(30))


#######################################
  
 
 
np.random.seed(42)
tf.random.set_seed(42)


print("\n\nLoad scan data spreadsheets...\n")
scan_data_files=["jam_scan_data_2020.xlsx","cond_scan_data_2020.xlsx","sauce_scan_data_2020.xlsx"]
#total_columns_count=1619+797
scan_dict_savename="scan_dict.pkl"

#output_dir = log_dir("scandata")
#os.makedirs(output_dir, exist_ok=True)

    
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format
  
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors


# print("\n\n\nIRI scan data reader - By Anthony Paech 25/5/20")
# print("=================================================\n")       

# print("Python version:",sys.version)
# print("\ntensorflow:",tf.__version__)
# #    print("eager exec:",tf.executing_eagerly())

# print("keras:",keras.__version__)
# print("numpy:",np.__version__)
# print("pandas:",pd.__version__)
# print("matplotlib:",mpl.__version__)

# print("sklearn:",sklearn.__version__)
   
# print("\nnumber of cpus : ", multiprocessing.cpu_count())

# visible_devices = tf.config.get_visible_devices('GPU') 

# print("tf.config.get_visible_devices('GPU'):",visible_devices)

 
# print("\n==================================================\n")       


   
np.random.seed(42)
tf.random.set_seed(42)
  
##############################################################################
   #report = namedtuple("report", ["name", "report_type","cust","prod"])
 
#scandata = namedtuple("scandata", ["market", "product","measure"])

#market_dict={0:"Woolworths",
#              1:"Coles"}

market_dict={0:"Woolworths",
              1:"coles"}
           #   2:"indies",
           #   3:"ritches",
           #   4:"drakes"}



# measure_dict={1:"units_sold_total",
#               2:"units_sold_off_promotion",
#               3:"units_sold_on_promotion",
#               4:"dollars_sold_total",
#               5:"dollars_sold_off_promotion",
#               6:"dollars_sold_on_promotion",
#               7:"full_price",
#               8:"promo_price",
#               9:"weighted_depth_of_dist"}


category_dict={0:"unknown",
               1:"jams",
               2:"condiments",
               3:"sauces",
               4:"dressings"}


plot_type_dict={0:"Units report",
                   1:"Dollars report",
                   2:"Units vs dist report",
                   3:"Dollars vs price report"}
              #     4:"Distribution report",
              #     5:"Price report"}


#Used to convert the meaure column names to a measure type to match the dict above
# the first digit is the plot report to group them in, the second is whether to stack the fields when plotting
measure_conversion_dict={0:2,
                         1:1,
                         2:3,
                         3:3,
                         4:3,
                      #   5:0,
                         6:0,
                         7:2,
                         8:2}


#  measure order
# {'Depth Of Distribution Wtd': 0,
# 'Dollars (000)': 1, 
# 'Dollars (000) Sold off Promotion >= 5 % 6 wks': 2,
#  'Dollars (000) Sold on Promotion >= 5 % 6 wks': 3, 
#  'Price ($/Unit)': 4, 
#  'Promoted Price >= 5 % 6 wks': 5, 
#  'Units (000)': 6, 
#   'Units (000) Sold off Promotion >= 5 % 6 wks': 7, 
# 'Units (000) Sold on Promotion >= 5 % 6 wks': 8}


#Stack the value when plotting?
stacked_conversion_dict={0:True,
                         1:False,
                         2:True,
                         3:True,
                         4:True,
                     #    5:False,
                         6:False,
                         7:True,
                         8:True}


# #Used to convert the meaure column names back to a measure type to match the dict above
# measure_unconversion_dict={4:0,
#                          2:1,
#                          2:2,
#                          2:3,
#                          3:4,
#                          3:5,
#                          1:6,
#                          1:7,
#                          1:8}




variety_type_dict={0:"unknown",
                   1:"total",
                   2:"apricot",
                   3:"blackberry",
                   4:"blueberry",
                   5:"orange marm",
                   6:"blood orange marm",
                   7:"plum",
                   8:"strawberry",
                   9:"raspberry",
                   10:"other",
                   11:"fig almond",
                   12:"tomato chutney",
                   13:"worcestershire",
                   14:"tomato sce",
                   15:"tomato sce hot",
                   16:"fruit of forest",
                   17:"roadhouse",
                   18:"lime lemon",
                   19:"peri peri",
                   20:"pear",
                   21:"quince",
                   22:"mustard pickles",
                   23:"mango",
                   24:"red currant",
                   25:"rose petal",
                   26:"ajvar",
                   27:"4 fruit",
                   28:"peach",
                   29:"cherry",
                   30:"chutney",
                   31:"blackcurrant",
                   32:"fig royal",
                   33:"tomato&red"
                   }

brand_dict={0:"unknown",
            1:"beerenberg",
            2:"st dalfour",
            3:"bonne maman",
            4:"cottees",
            5:"anathoth",
            6:"roses",
            10:"baxters",
            11:"whitlocks",
            12:"barker",
            13:"three threes",
            14:"spring gully",
            15:"masterfoods",
            16:"yackandandah",
            17:"goan",
            18:"podravka",
            20:"heinz",
            21:"leggos",
            22:"mrs h.s.ball",
            23:"branston",
            24:"maggie beer",
            25:"country cuisine",
            26:"fletchers",
            27:"jamie oliver",
            28:"regimental",
            29:"maleny",
            7:"ixl",
            30:"red kellys",
            100:"other"}


#############################
 

# def find_brands(brand_dict,product_list,df):
#     #  search through products and extract brand names per column
#   #  print(find_in_dict(brand_dict,"oran"))  
#     #r=[k for k, v in brand_dict.items() if v in product_list]  
#  #   r=[brand_dict.keys() for v in product_list if v in brand_dict.items()]   
    
#     m = [k for k,v in brand_dict.items() if any(v in p.lower() for p in product_list)]
#  #   print(m)
#     #    print(p,m)
#    #     if p.lower() in list(brand_dict.values()):
#    #         print("p",p)
    

#     return brand_dict[m[0]]


# def recreate_full_index(df):
#     print("\nstart recreate=\n",df,category_dict)
#     cats=list(set(list(df.columns.levels[0])))
#     print(cats)
#     full_index_df=df.copy(deep=True)
#     print(full_index_df)
#     full_index_df.rename(columns=category_dict,level='category',inplace=True)
#     print("\n",full_index_df)
#     full_index_df.rename(columns=market_dict,level='market',inplace=True)
#     full_index_df.rename(columns=brand_dict,level='brand',inplace=True)
#     full_index_df.rename(columns=variety_type_dict,level='variety',inplace=True)
#     full_index_df.rename(columns=measure_unconversion_dict,level='measure',inplace=True)
#     full_index_df.rename(columns=measure_type_dict,level='measure',inplace=True)
#     return full_index_df
    




def find_in_dict(dictname,name):
    m= [k for k, v in dictname.items() if v in name.lower()]
    if len(m)>1:
        m=m[-1]
    elif len(m)==1:
        m=m[0]
    else:
        m=0
    return m    

    
    
    
np.random.seed(42)
tf.random.set_seed(42)
        
#column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
#rename_dict=dict({"qty":"BB_total_invoiced_sales"})
#df=df.astype(convert_dict)    
    

count=1
for scan_file in scan_data_files:
    column_count=pd.read_excel(scan_file,-1).shape[1]   #count(axis='columns')
    print("Loading...",scan_file,"->",column_count,"columns")
    convert_dict={col: np.float64 for col in range(1,column_count-1)}   #1619
    convert_dict['index']=np.datetime64

    if count==1:
        df=pd.read_excel(scan_file,-1,dtype=convert_dict,index_col=0,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
        df=df.T
        df['category']=[category_dict[count]]*(column_count-1)
        df = df.set_index('category', append=True)
        df=df.T

    else:
   #     print(convert_dict)
     #   del df2
        df2=pd.read_excel(scan_file,-1,index_col=0,header=[0,1,2])
     #   print(df2)
        df2=df2.T
        df2['category']=[category_dict[count]]*(column_count-1)
        df2 = df2.set_index('category', append=True)
        df2=df2.T
   #     print(df2)
        df=pd.concat([df,df2],axis=1)   #,keys=[df['index']])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
      #  del df2
   # print(df)
    count+=1 
    
    


print("\n")
df=df.reorder_levels([3,0,1,2],axis=1)

df=df.T
df.index.set_names('market', level=1,inplace=True)

df.index.set_names('product', level=2,inplace=True)
df.index.set_names('measure', level=3,inplace=True)
plot_type=df.index.get_level_values(3)
market_name=df.index.get_level_values(1)

df['plot_type']=plot_type
df['market_name']=market_name
df['stacked']=plot_type
#print(df)
#df=df.T
df=df.set_index('market_name', append=True)

df=df.set_index('plot_type', append=True)
df=df.set_index('stacked', append=True)

#df=df.rename_levels(['category','market','product','measure'],axis=1)

df=df.T
#print(df)
#df = df.set_index('category', append=True)

#print("dc=\n",df,df.columns,df.shape)
#convert_dict={col: np.float64 for col in range(1,sheet_cols)}
#convert_dict['index']=np.datetime64
#df=df.astype(convert_dict)    
    
df.fillna(0.0,inplace=True)
#print("convert dict",convert_dict.items())
#df = df.astype(convert_dict) 

market_list=list(set(list(df.columns.levels[1])))
#print(market_list)
market_dict={k:market_list[k] for k in range(len(market_list))}
market_rename_dict={market_list[k]:k for k in range(len(market_list))}

#print("\nmd=",market_dict)
#market_dict{0:}
#product_list=list(set(list(df.columns.levels[1])))
#print(product_list)

#product_dict={k:product_list[k] for k in range(len(product_list))}
#product_rename_dict={product_list[k]:k for k in range(len(product_list))}

#print("\npd=",product_dict)


measure_list=list(df.columns.levels[3])
#stacked_list=list(df.columns.levels[3])
#print(measure_list)

#measure_dict={k:measure_list[k] for k in range(len(measure_list))}
measure_rename_dict={measure_list[k]:k for k in range(len(measure_list))}

#print("\nmsd=",measure_dict)
#print("\nm rename d=",measure_rename_dict)
#print("\nm conversion d=",measure_conversion_dict)


df=df.T
#df.index.set_names('market', level=0,inplace=True)
#df.index.set_names('product', level=1,inplace=True)
#df.index.set_names('measure', level=2,inplace=True)
df.index.set_names('market_name', level=4,inplace=True)

df.index.set_names('plot_type', level=5,inplace=True)
df.index.set_names('stacked', level=6,inplace=True)

#df = df.set_index('category', append=True)
         
#print(df)
df=df.T
#print(df)
#product_columns=list(df.columns.levels[1])

original_df=df.copy(deep=True)
#print("orig df=\n",original_df)








#s=scan_data(market_list,product_list,measure_list)
#print("s=",s)    
# call a x-section of the database out with a tuple (type,y)


#df=df.xs(product_list[2],level=1,drop_level=False,axis=1)
#print(df)




df.rename(columns=market_rename_dict,level='market',inplace=True)
#print("1",df.T)
#df.rename(columns=product_rename_dict,level='product',inplace=True)
df.rename(columns=measure_rename_dict,level='plot_type',inplace=True)
#print("2",df.T)
df.rename(columns=measure_conversion_dict,level='plot_type',inplace=True)

df.rename(columns=measure_rename_dict,level='stacked',inplace=True)

df.rename(columns=stacked_conversion_dict,level='stacked',inplace=True)


#print("3",df.T)


#######################################################
# add brand, variety and catoery to multiindex index
  #  print(c,"=",variety_type_dict[c])
#    print(c,find_in_dict(brand_dict,c),find_in_dict(variety_type_dict,c))  
 #   print(df.loc[find_in_dict(brand_dict,c)])

brand_values=[find_in_dict(brand_dict,c) for c in original_df.columns.get_level_values('product')]
#print("brands:",brand_values)
product_values=[find_in_dict(variety_type_dict,c) for c in original_df.columns.get_level_values('product')]
#print("products:",product_values)
  #  print(c,"=",variety_type_dict[c])
   # print(c,find_in_dict(brand_dict,c),find_in_dict(variety_type_dict,c))  
#print(brand_values,product_values)

df=df.T
df['brand']=brand_values
df = df.set_index('brand', append=True)
#df['category']=['c']*df.shape[0]
#df = df.set_index('category', append=True)
df['variety']=product_values
df = df.set_index('variety', append=True)
#df=df.reorder_levels([4,0,3,5,2,1,6],axis=0)
df=df.reorder_levels(['category','market','brand','variety','plot_type','stacked','market_name','product','measure'],axis=0).T

# new_level_name = "brand"
# new_level_labels = ['p']
# df1 = pd.DataFrame(data=1,index=df.index, columns=new_level_labels).stack()
# df1.index.names = [new_level_name,'market','product','measure']
# #df=df.T.index.names=['brand','market','product','measure']

#print(df)
#print("\n",df.T)

#full_index_df=recreate_full_index(df)
#print(full_index_df)

scan_dict={"original_df":original_df,
           "final_df":df,
 #          "full_index_df":full_index_df,
           "market_dict":market_dict,
        #   "product_dict":product_dict,
           "measure_conversion_dict":measure_conversion_dict,
           "stacked_conversion_dict":stacked_conversion_dict,
           'plot_type_dict':plot_type_dict,
           'brand_dict':brand_dict,
           'category_dict':category_dict,
           'variety_type_dict':variety_type_dict}


with open(scan_dict_savename,"wb") as f:
    pickle.dump(scan_dict,f,protocol=-1)
    
##############################################################    

with open(scan_dict_savename, 'rb') as g:
    scan_data_dict = pickle.load(g)



print("final_df shape:",scan_data_dict['final_df'].shape)
print("\n\n********************************************\n")
print("unknown brands=")
try:
    print(df.xs(0,level='brand',drop_level=False,axis=1))
except:
    print("no unknown brands\n")
#print("unknown variety")
#try:
#    print(df.xs(0,level='variety',drop_level=False,axis=1))
#except:
#    print("no unknown varieis\n")    
print("unknown meausre type=")
try:
    print(df.xs(0,level='measure',drop_level=False,axis=1))
except:
    print("no unknown measures")


#
print("\n\n")
print("All scandata dataframe saved to",scan_dict_savename,":\n",scan_data_dict['final_df'])


























    
 
    
 
    
 
###################################################    
 

with open(sales_df_savename,"rb") as f:
    sales_df=pickle.load(f)

#print("sales shape df=\n",sales_df.shape)

first_date=sales_df['date'].iloc[-1]
last_date=sales_df['date'].iloc[0]

print("\nAttache sales trans analysis.  Current save is:")


print("\nData available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")
#print("\n\n")   

answer="y"
answer=input("Refresh salestrans?")
if answer=="y":
    sales_df=load_sales(filenames)  # filenames is a list of xlsx files to load and sort by date
    with open(sales_df_savename,"wb") as f:
          pickle.dump(sales_df, f,protocol=-1)

    
sales_df.sort_values(by=['date'],ascending=True,inplace=True)
last_date=sales_df['date'].iloc[-1]
first_date=sales_df['date'].iloc[0]

print("\nAttache sales trans analysis up to date.  New save is:",sales_df_savename)
print("\nData available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")

dds=sales_df.groupby(['period'])['salesval'].sum().to_frame() 
datelen=dds.shape[0]-365







################################################
name="Beerenberg GSV MAT"
print("\n",name)
dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
dds['date']=dds.index.tolist()
latest_date=dds['date'].max()
title=name+" w/c:("+str(latest_date)+")"

dds.reset_index(inplace=True)
 #print(dds)
#dds.drop(['period'],axis=1,inplace=True)
 
fig=dds.tail(dds.shape[0]-731)[['date','mat']].plot(x='date',y=['mat'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
figname="Afig_"+name
save_fig(figname)
dds[['date','mat']].to_excel(output_dir+name+".xlsx") 

#dds=sales_df.groupby(['period'])['salesval'].sum().to_frame() 

name="Beerenberg GSV Annual growth rate"
print("\n",name)
title=name+" w/c:("+str(latest_date)+")"

result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"_*","_*")]=result
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

result.to_excel(output_dir+name+".xlsx") 


#########################################
name="shop GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['glset']=="SHP")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"CASHSHOP","_*")]=result
report_dict[report(name,8,"CASHSHOP","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

result.to_excel(output_dir+name+".xlsx") 

############################################

name="ONL GSV sales $"
print("\n",name)
shop_df=sales_df[(sales_df['glset']=="ONL")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"_*","_*")]=result
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

result.to_excel(output_dir+name+".xlsx") 

############################################
name="Export GSV sales $"
print("\n",name)
shop_df=sales_df[(sales_df['glset']=="EXS")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"_*","_*")]=result
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

result.to_excel(output_dir+name+".xlsx") 

############################################
name="NAT sales GSV$"

print("\n",name)
shop_df=sales_df[(sales_df['glset']=="NAT")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"_*","_*")]=result
report_dict[report(name,8,"_*","_*")]=figname
result.to_excel(output_dir+name+".xlsx") 

############################################
name="WW (010) GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['specialpricecat']==10)]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"_*","_*")]=result
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

result.to_excel(output_dir+name+".xlsx") 

############################################
name="Coles (012) GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['specialpricecat']==12)]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"_*","_*")]=result
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

result.to_excel(output_dir+name+".xlsx") 

############################################
name="DFS GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['glset']=="DFS")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3,"_*","_*")]=result
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

result.to_excel(output_dir+name+".xlsx") 

#plt.draw()
#plt.pause(0.001)
#plt.show(block=False)

#plt.ion()
#plt.show()
plt.close('all')
############################################

# create a sales_trans pivot table
name="Sales summary (units or ctns)"

print("\n",name)
#print(sales_df.columns)
sales_df = sales_df.drop(sales_df[(sales_df['productgroup']==0)].index)

sales_df["month"] = pd.to_datetime(sales_df["date"]).dt.strftime('%m-%b')
sales_df["year"] = pd.to_datetime(sales_df["date"]).dt.strftime('%Y')
pivot_df=pd.pivot_table(sales_df, values='qty', index=['productgroup','product'],columns=['year','month'], aggfunc=np.sum, margins=False,dropna=True,observed=True)
name="pivot_table_units"
pivot_df.to_excel(output_dir+name+".xlsx") 
report_dict[report(name,6,"_*","_*")]=pivot_df
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"



pivot_df=pd.pivot_table(sales_df, values='salesval', index=['productgroup','product'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)

name="pivot_table_dollars"
pivot_df.to_excel(output_dir+name+".xlsx") 
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"


pivot_df=pd.pivot_table(sales_df[sales_df['productgroup']<"40"], values='qty', index=['productgroup'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
name="pivot_table_units_product_group"
figname=plot_stacked_line_pivot(pivot_df,name,False)   #,number=6)


pivot_df.to_excel(output_dir+name+".xlsx") 
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"




pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset','specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)

name="pivot_table_customers_x_glset_x_spc"
pivot_df.to_excel(output_dir+name+".xlsx")
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])

#print(pivot_df)  
name="pivot_table_customers_x_glset"
pivot_df.to_excel(output_dir+name+".xlsx")
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
name="Dollar sales per month by spc"
figname=plot_stacked_line_pivot(pivot_df,name,False)   #,number=6)
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

pivot_df.to_excel(output_dir+name+".xlsx")


#print(pivot_df) 
pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
name="pivot_table_customers_spc_nocodes"
pivot_df.to_excel(output_dir+name+".xlsx")
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,8,"_*","_*")]=figname
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#ax=plot_stacked_line_pivot(pivot_df,"Unit sales per month by productgroup",False)   #,number=6)

#print(pivot_df) 
name="pivot_table_customers_x_spc"
pivot_df.to_excel(output_dir+name+".xlsx") 
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['cat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#print(pivot_df) 
name="pivot_table_customers"
pivot_df.to_excel(output_dir+name+".xlsx") 
report_dict[report(name,6,"_*","_*")]=pivot_df
#report_dict[report(name,5,"*","*")]=name+".xlsx"
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"


##################################################################3
# update reports and save them as pickles for the brand_index.py program

print("\nUpdate and save the qty reports from the coles_pkl_dict\n")



for key in coles_pkl_dict.keys():
    brand=coles_pkl_dict[key][0]
    spc=coles_pkl_dict[key][1]
    pg=str(coles_pkl_dict[key][2])
    pc=coles_pkl_dict[key][3]
    if (pc=="_*") | (pc=="_t") | (pc=="_T"):
    #    print("pc=",pc,pg,spc)
        v=sales_df.query('specialpricecat==@spc & productgroup==@pg')[['date','qty']]
    else: 
        v=sales_df.query('specialpricecat==@spc & product==@pc')[['date','qty']]
    print("saving",key)  #,"=\n",v)      
    #print(v)
    with open(key,"wb") as f:
          pickle.dump(v, f,protocol=-1)



##############################################################33
# rank top customers and products
#
latest_date=sales_df['date'].max()

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
name="Top 50 customers by $purchases in the last 30 days"
print("\n",name)
print(unique_code_pivot_df[['code','total_dollars']].head(50))
report_dict[report(name,3,"_*","_*")]=unique_code_pivot_df
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

unique_code_pivot_df[['code','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 


year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["specialpricecat"]).transform(sum)
pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
#print("pv=",pivot_df)
name="Top 50 customers special price category by $purchases in the last 30 days"
unique_code_pivot_df=pivot_df.drop_duplicates('specialpricecat',keep='first')
#unique_code_pivot_df=pd.unique(pivot_df['code'])
print("\n",name)
print(unique_code_pivot_df[['specialpricecat','total_dollars']].head(50))
report_dict[report(name,3,"_*","_*")]=unique_code_pivot_df
report_dict[report(name,5,"_*","-*")]=output_dir+name+".xlsx"

unique_code_pivot_df[['specialpricecat','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 






year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["product"]).transform(sum)
year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["product"]).transform(sum)

pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
#print("pv=",pivot_df)
unique_code_pivot_df=pivot_df.drop_duplicates('product',keep='first')

name="Top 50 products by $sales in the last 30 days"
#unique_code_pivot_df=pd.unique(pivot_df['code'])
print("\n",name)
print(unique_code_pivot_df[['product','total_units','total_dollars']].head(50))
#pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 
report_dict[report(name,3,"_*","_*")]=unique_code_pivot_df
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

unique_code_pivot_df[['product','total_units','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 




year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["productgroup"]).transform(sum)
year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["productgroup"]).transform(sum)
pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
unique_pg_pivot_df=pivot_df.drop_duplicates('productgroup',keep='first')

name="Top productgroups by $sales in the last 30 days"
print("\n",name)
print(unique_pg_pivot_df[['productgroup','total_units','total_dollars']].head(20))
#pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 
report_dict[report(name,3,"_*","_*")]=unique_code_pivot_df
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

unique_pg_pivot_df[['productgroup','total_units','total_dollars']].head(20).to_excel(output_dir+name+".xlsx") 


name="Top 50 Credits in past 30 days"
print("\n",name)
end_date=sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
#print(end_date)
#print("ysdf=",sales_df)
month_sales_df=sales_df[sales_df['date']>end_date]
#print("msdf=",month_sales_df)
credit_df=month_sales_df[(month_sales_df['salesval']<-100) | (month_sales_df['qty']<-10)]
#print(credit_df.columns)
credit_df=credit_df.sort_values(by=["salesval"],ascending=[True])

print(credit_df.tail(50)[['date','code','glset','qty','salesval']])
report_dict[report(name,3,"_*","_*")]=credit_df
report_dict[report(name,5,"_*","_*")]=output_dir+name+".xlsx"

credit_df[['date','code','glset','qty','salesval']].tail(50).to_excel(output_dir+name+".xlsx") 






#################################################################################################

print("\nChecking sales trends by customers and products of past year.")

# find all the good performing and poor performing outliers in retail sales
#  limit product groups
#product_groups_only=["10","11","12","13","14","15","18"]
#spc_only=["088"]

# for each spc
# colect all the customer that have bought more than 3 products over $1000 in total over more them 3 trnsactions in the past year
#
# for each customer code, rank the sales growth of each product bought and the total sales
# with the products belonging product_groups_only
# append to a list
# sort the whole list
# highlight the top 20 growers and botomlatest_date=sales_df['date'].max()
# 20 losers
#
#print("\nSales performace start=\n",sales_df)

end_date=sales_df['date'].iloc[-1]- pd.Timedelta(366, unit='d')
#print(end_date)
year_sales_df=sales_df[sales_df['date']>end_date]
#print("ysdf1=",year_sales_df)
year_sales_df=year_sales_df[year_sales_df['productgroup'].isin(product_groups_only) & year_sales_df['specialpricecat'].isin(spc_only)]   
#print("ysdf2=",year_sales_df[['date','code','product']])
  
cust_list=year_sales_df.code.unique()
cust_list = cust_list[cust_list != 'OFFINV']
#cust_licust_list.remove('OFFINV')
cust_list.sort()
prod_list=year_sales_df['product'].unique()
prod_list.sort()

#print("c=",cust_list,len(cust_list))
#print("p=",prod_list,len(prod_list))

print("\nunique customers=",len(cust_list))
print("unique products=",len(prod_list))


year_sales_df['counter']=0
new_sales_df=year_sales_df.copy(deep=True)
new_sales_df=new_sales_df.iloc[0:0]
#print(new_sales_df)

#figure_list=[]
t=0
total=len(cust_list)*len(prod_list)
print("total combinations=",total,"\n")
for cust in cust_list:
    for prod in prod_list:
        s=year_sales_df[(year_sales_df['code']==cust) & (year_sales_df['product']==prod) & (year_sales_df['salesval']>0.0) & (year_sales_df['qty']>0.0)].copy(deep=True)
        s['counter']=s.shape[0]
    #    print("s=\n",s[['code','product','counter']],s.shape)
        s=s.sort_values('date',ascending=False)
      #  s.index=s.date
        t+=1
        if t%10==0:
            print("\r",cust,prod,"+",s.shape[0],"=",new_sales_df.shape[0],int(round(t/total*100,0)),"%               ",end='\r',flush=True)

        if s.shape[0]>7: 
            s['slope'],figname,name=calculate_first_derivative(s,cust,prod,latest_date)  
           # s['figure']=figure
          #  figure_list.append(figure)
            new_sales_df=new_sales_df.append(s)
            if (figname!="") & (name!=""):
                report_dict[report(name,8,cust,prod)]=figname
 
print("\n\n")
#print("\nysdf3=",new_sales_df[['date','code','product','counter','slope']],new_sales_df.shape)
new_sales_df.drop_duplicates(['code','product'],keep='first',inplace=True)
#new_sales_df=new_sales_df[new_sales_df['slope']>0.02]
new_sales_df.sort_values(['slope'],ascending=[False],inplace=True)
name="growth rankings"
print("\nbest growth=",new_sales_df[['code','product','slope']].head(100).to_string())
print("\nworst growth=",new_sales_df[['code','product','slope']].tail(50).to_string())
print(new_sales_df.shape)
report_dict[report(name,3,"_*","_*")]=new_sales_df
new_sales_df[['code','product','slope']].to_excel(output_dir+name+".xlsx") 


#print("\n\nreport dict=\n",report_dict.keys())
print("reports being pickled and saved to",report_savename)
with open(report_savename,"wb") as f:
    pickle.dump(report_dict, f,protocol=-1)
  
#plt.pause(0.001) 
#plt.show()
plt.close("all")





#############################
 

   
np.random.seed(42)
tf.random.set_seed(42)
        
#column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
 
 

#rename_dict=dict({"qty":"BB_total_invoiced_sales"})
df=pd.read_excel(colesscan,-1,skiprows=[0,1,2],dtype=object).T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)



df.fillna(0.0,inplace=True)


df = df.rename({0:"scan_week"})
df=df.T
#print("after",df)
df['scan_week']=pd.to_datetime(df['scan_week'],format="%d/%m/%Y")
df.drop_duplicates(keep='first', inplace=True)

# using dictionary to convert specific columns 

df = df.astype(coles_convert_dict) 

df=df.T
df = df.rename(coles_col_dict,axis='index')
#print("df=\n",df)
df.index = pd.MultiIndex.from_tuples(df.index,names=["brand","specialpricecat","productgroup","product","on_promo","names"])
df=df.T

df.drop_duplicates(keep='first', inplace=True)
df=df.set_index(list(df.columns[[0]]))   #.dt.strftime('%d/%m/%Y')
df.index.name = 'scan_week'
df.index = pd.to_datetime(df.index, format = '%d/%m/%Y',infer_datetime_format=True)
#df=df.astype(np.float32)  #,inplace=True)

#print("after6=\n",df)
#print(df.columns)
df.replace(np.nan, 0.0,inplace=True)

#test=get_xs_name(df,3,0)
#print("test=\n",test)


df[0,12,10,"_T",0,'coles_jams_total_scanned']=df[0,12,10,"_*",0,'coles_total_jam_curd_marm_off_promo_scanned']+df[0,12,10,"_*",1,'coles_total_jam_curd_marm_on_promo_scanned']
df[1,12,10,"_t",0,'coles_BB_jams_total_scanned']=df[1,12,10,"_*",0,'coles_BB_jams_off_promo_scanned']+df[1,12,10,"_*",1,'coles_BB_jams_on_promo_scanned']
df[2,12,10,"_t",0,'coles_SD_jams_total_scanned']=df[2,12,10,"_*",0,'coles_SD_jams_off_promo_scanned']+df[2,12,10,"_*",1,'coles_SD_jams_on_promo_scanned']
df[3,12,10,"_t",0,'coles_BM_jams_total_scanned']=df[3,12,10,"_*",0,'coles_BM_jams_off_promo_scanned']+df[3,12,10,"_*",1,'coles_BM_jams_on_promo_scanned']
#df=df*1000

df[1,12,10,"_t",0,'coles_BB_jams_on_promo']=(df[1,12,10,"_*",1,'coles_BB_jams_on_promo_scanned']>0.0)
df[2,12,10,"_t",0,'coles_SD_jams_on_promo']=(df[2,12,10,"_*",1,'coles_SD_jams_on_promo_scanned']>0.0)
df[3,12,10,"_t",0,'coles_BM_jams_on_promo']=(df[3,12,10,"_*",1,'coles_BM_jams_on_promo_scanned']>0.0)



df["","","","_t","",'weekno']= np.arange(df.shape[0])

#print("after7=\n",df)
############################################
# total all other on_promo and off_promo on matching productcodes (level 3)
# and level 4 is either 0 or 1
# get a list of products
#products=list(set(list(df.columns.get_level_values(3))))
#print("products=",products)

for col_count in range(0,len(df.columns),2):
    cc=list(df.iloc[:,col_count].name)
    if cc[0]==1:
        brand="BB"
    else:
        brand="other"
    if cc[1]==10:
        cust="WW"
    elif cc[1]==12:
        cust="coles"
    else:
        cust="other"
    
    #customer=cc[1]
    p=cc[3]    # product name
    cc[4]=2   # type is total                 
    cc[5]=str(cust)+"_"+str(brand)+"_"+str(p)+"_total_scanned"
    cc=tuple(cc)
    df[cc]=df.T.xs(p,level=3).sum()



tdf=get_xs_name(df,("_t",3))
plot_query2(tdf,['coles_BB_jams_total_scanned','coles_SD_jams_total_scanned','coles_BM_jams_total_scanned'],'BB total scanned Coles jam units per week')
#tdf['coles_scan_week']=tdf.index
tdf.reset_index('scan_week',drop=True,inplace=True)
tdf=tdf.astype(np.float64)
#print("tdf=\n",tdf)
#print("tdf.T=\n",tdf.T)

#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')  #,fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
sns.lmplot(x='weekno',y='coles_BB_jams_total_scanned',data=tdf,col='coles_BB_jams_on_promo',hue='coles_SD_jams_on_promo')  #,fit_reg=True,robust=True,legend=True) 
save_fig("coles1")   #),images_path)
sns.lmplot(x='weekno',y='coles_BB_jams_total_scanned',data=tdf,col='coles_BB_jams_on_promo',hue='coles_BM_jams_on_promo')
save_fig("coles2")  #,images_path)
sns.lmplot(x='weekno',y='coles_BB_jams_total_scanned',data=tdf,col='coles_SD_jams_on_promo',hue='coles_BB_jams_on_promo')  #,fit_reg=True,robust=True,legend=True) 
save_fig("coles3")   #),images_path)
sns.lmplot(x='weekno',y='coles_BB_jams_total_scanned',data=tdf,col='coles_BM_jams_on_promo',hue='coles_BB_jams_on_promo')
save_fig("coles4")  #,images_path)
sns.lmplot(x='weekno',y='coles_BB_jams_total_scanned',data=tdf,col='coles_BM_jams_on_promo',hue='coles_SD_jams_on_promo')  #,fit_reg=True,robust=True,legend=True) 
save_fig("coles5")   #,images_path)
sns.lmplot(x='weekno',y='coles_BB_jams_total_scanned',data=tdf,col='coles_SD_jams_on_promo',hue='coles_BM_jams_on_promo')
save_fig("coles6")  #,images_path)
################################################################333
# convert weekly scan data to daily sales

df.replace(np.nan, 0.0, inplace=True)



#print(df)
df=df*1000
#print(df)

####################################
# coles_pkl_dict which is save in a dictionary of report_dict as a pickle
# coles_pkl_dict contains a list of files names as keys to run as the actual sales in the prediction vs actual df
#

with open(report_savename,"rb") as f:
    report_dict=pickle.load(f)

#print("report dict=",report_dict.keys())
coles_pkl_dict=report_dict[report('coles_pkl_dict',0,"","")]
#print("cole_pkl dict=",coles_pkl_dict)

###########################################3


joined_df=df.copy(deep=True)
for key in coles_pkl_dict.keys():
   # savepkl="scanned_sales_plus_"+key
    print("Loading query dataframe:",key)
    with open(key,"rb") as f:
        actual_sales=pickle.load(f)
 #   print("key=",key,"coles_pkl_dict]key]=",pkl_dict[key],"\n",actual_sales)    
    actual_sales.reset_index(drop=True,inplace=True)  
    actual_sales.index=actual_sales.date
    actual_sales=actual_sales[['qty']]
  #  print(actual_sales)
    forecast_df = actual_sales.resample('W-SAT', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
 #   print(key,"fdf=\n",forecast_df,"pdk=",pkl_dict[key])
    joined_df=pd.concat([joined_df,forecast_df],axis=1)   #.sort_index(axis=1)
#    joined_df=joined_df.rename(columns={"qty":key.rsplit(".", 1)[0]})
    joined_df=joined_df.rename(columns={"qty":coles_pkl_dict[key]})
 
    shifted_key=list(coles_pkl_dict[key])
    
    #  create another query with the invoiced sales shifted left 3 week to align with scanned sales
    shifted_df=forecast_df.shift(3, freq='W')[:-3]   # 3 weeks
  #  print("shufted key=",shifted_key)
    shifted_key[4]=4
    shifted_key[5]=shifted_key[5]+"_shifted_3wks"
    joined_df=pd.concat([joined_df,shifted_df],axis=1)   #.sort_index(axis=1)
#    joined_df=joined_df.rename(columns={"qty":key.rsplit(".", 1)[0]})
    joined_df=joined_df.rename(columns={"qty":tuple(shifted_key)})

    
 
print("\n")    
#print("df=",df)
joined_df=joined_df.T
#print("\njoined_df before=\n",joined_df)


joined_df.index = pd.MultiIndex.from_tuples(joined_df.index,names=["brand","specialpricecat","productgroup","product","on_promo","names"])
joined_df=joined_df.T


products=list(set(list(joined_df.columns.get_level_values(3))))
#print("products=",products)

#graph_list=[]
#print("jdf=\n",joined_df)

#joined_df=joined_df.T

joined_df['lastdate'] = pd.to_datetime(joined_df.index,format="%Y-%m-%d",exact=False)

latest_date = joined_df['lastdate'].max()

  
for p in products:
    if (p=="_t") | (p=="_*") | (p=="_T"):
        pass
    else:
        test=joined_df.T.xs(p,level=3,drop_level=True)
        test=test.droplevel(level=0,axis=0)
        test=test.droplevel(level=0,axis=0)
        test=test.droplevel(level=0,axis=0).T
        mat=4
        rdf=test[[2,3,4]].rolling(mat,axis=0).mean()
        
        rdf=rdf.droplevel(level=0,axis=1)
        

        styles1 = ['b-','g:','r-']
       # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]

        #styles2 = ['rs-','go-','b^-']
       # fig, ax = plt.subplots()

        ax2=rdf.plot(grid=True,title="Coles units moving total "+str(p)+":"+str(mat)+" weeks w/c:("+str(latest_date)+")",style=styles1, lw=linewidths)   #),'BB total scanned vs purchased Coles jam units per week')
        ax2.legend(title="")
        save_fig(p+"_moving_total")   #,images_path)
#plt.grid(True)
      #  plt.show()
plt.close("all")

#savepkl="invoiced_and_scanned_sales.pkl"

print("saving query dataframe:",savepkl)
pd.to_pickle(rdf,savepkl)

#print(joined_df.columns)


# use Coles scan data from IRI weekly to predict Coles orders
#  X is the BB_total_sales
# the Target y is scanned sales 4 weeks ahead
# 
#print("jdf=\n",joined_df)
hdf=joined_df.copy(deep=True)
if hdf.columns.nlevels>=2:
    for _ in range(hdf.columns.nlevels-1):
        hdf=hdf.droplevel(level=0,axis=1)

#hdf=get_xs_name2(joined_df,"",5)
#print("hdf=\n",hdf.columns)
df=hdf[['coles_BB_jams_total_scanned','coles_BB_jams_invoiced','coles_BB_jams_invoiced_shifted_3wks']].rolling(mat,axis=0).mean()
styles1 = ['b-','g:','r-']
       # styles1 = ['bs-','ro:','y^-']
linewidths = 1  # [2, 1, 4]

  
ax=df.plot(grid=True,title="Coles units moving total "+str(mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
ax.legend(title="")
save_fig("Coles_total_jams_units_moving_total")
#plt.show()
#print(df)
plt.close("all")

############################################################33
# load previous runs coles_predictions

previous_df=pd.read_pickle("coles_order_predict_results.pkl")
#prev_cols=list(previous_df.columns).contains("prediction")
#print("prev_cols",prev_cols)
pred_cols = [col for col in previous_df.columns if 'prediction' in col]
#print(list(previous_df.columns))
#print(pred_cols)
previous_df=previous_df[pred_cols]
previous_df.columns=previous_df.columns+"_old"
#print("previous df=\n",previous_df)
#print("joined_df=\n",joined_df)
#joined_df=pd.concat((joined_df,previous_df),axis=1)
#print("new joined_df=\n",joined_df)

#############################################################
     # 
    # 
    # no of weeks
#target_offset=3

#answer="n"
#answer=input("\nPredict next weeks Coles orders? (y/n)\n")
#if answer=="y":
pfx="coles_BB_"
print("Orders to predict:",pfx,products)
count=0
for p in products:
    
    if (p=="_t") | (p=="_*") | (p=="_T"):
        pass
    else:
        test=joined_df.T.xs(p,level=3,drop_level=True)
        test=test.droplevel(level=0)
        test=test.droplevel(level=0)
        test=test.droplevel(level=0).T
        mdf=test[[2,3,4]]   #.rolling(mat,axis=0).mean()
        
        mdf=mdf.droplevel(level=0,axis=1)
        
        
        mdf.fillna(0,inplace=True)
        # X_set=mdf['coles_BB_jams_total_scanned'].to_numpy().astype(np.int32)[7:-1]
        X_set=mdf.iloc[:,0].to_numpy().astype(np.int32)[7:-1]     #np.array([13400, 12132, 12846, 9522, 11858 ,13846 ,13492, 12310, 13584 ,13324, 15656 ,15878 ,13566, 10104 , 7704  ,7704])
 
 
      #  y_set=mdf['coles_BB_jams_invoiced_shifted_3wks'].to_numpy()[7:-1]   #iloc[target_offset:].to_numpy()
        y_set=mdf.iloc[:,2].to_numpy().astype(np.int32)[7:-1]
      #   X_new=mdf['coles_BB_jams_total_scanned'].to_numpy().astype(np.int32)[-2:]   #iloc[target_offset:].to_numpy()

        dates=mdf.index.tolist()[7:-1]

    #    print(p,mdf.T,X_set,y_set)
        model=train_model(pfx+p,X_set,y_set,batch_length,no_of_batches)
        if count==0:
            results=predict_order(mdf,pfx+p,model).iloc[:,:]
        else:    
            results=pd.concat((results,predict_order(mdf,pfx+p,model).iloc[:,:]),axis=1)
    #    print(count,"results:=\n",results,results.shape)    
        count+=1    
    
    
 #   print("results=\n",results)
 #   print("results.T=\n",results.T)
#results.index = pd.to_datetime(df.index, format = '%d-%m-%Y',infer_datetime_format=True)
results=pd.concat((results,previous_df),axis=1)
results.sort_index(axis=1,inplace=True)
print("results=\n",results.tail(5))

results.to_pickle(output_dir+"coles_order_predict_results.pkl")
results.to_pickle("coles_order_predict_results.pkl")

results.to_excel(output_dir+"coles_order_predict_results.xlsx")


##########################################################################3

##  Woolworths

        
#column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
 
 


       
df=pd.read_excel(woolscan,-1,skiprows=[0,1,2]).T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
#print("before",df)

df.fillna(0.0,inplace=True)

#.df=df.T.set_index(['specialpricecat','productgroup'])
#print("after",df)
#df=df.T
#print("before",df)

df = df.rename({0:"scan_week"})
df=df.T
#print("after",df)
#df['scan_week']=pd.to_datetime(df['scan_week'],format="%Y-%m-%d",infer_datetime_format=True)
df.drop_duplicates(keep='first', inplace=True)

# using dictionary to convert specific columns 



df = df.astype(ww_convert_dict) 

df=df.T
df = df.rename(ww_col_dict,axis='index')
#print("df=\n",df)
df.index = pd.MultiIndex.from_tuples(df.index,names=["brand","specialpricecat","productgroup","product","on_promo","names"])
df=df.T

df.drop_duplicates(keep='first', inplace=True)
df=df.set_index(list(df.columns[[0]]))   #.dt.strftime('%d/%m/%Y')
df.index.name = 'scan_week'
df.index = pd.to_datetime(df.index, format = "%Y-%m-%d",infer_datetime_format=True)

#df.replace(np.nan, 0.0, inplace=True)
#df=df.astype(np.float32)  #,inplace=True)

#print("after6=\n",df)
#print(df.columns)

#test=get_xs_name(df,3,0)
#print("test=\n",test)

#df[0,10,10,"_T",0,'WW_jams_total_scanned']=df[0,10,10,"_*",0,'WW_total_jam_curd_marm_off_promo_scanned']+df[0,10,10,"_*",1,'WW_total_jam_curd_marm_on_promo_scanned']
df[1,10,10,"_t",0,'WW_BB_jams_total_scanned']=df[1,10,10,"_*",0,'WW_BB_jams_off_promo_scanned']+df[1,10,10,"_*",1,'WW_BB_jams_on_promo_scanned']
df[2,10,10,"_t",0,'WW_SD_jams_total_scanned']=df[2,10,10,"_*",0,'WW_SD_jams_off_promo_scanned']+df[2,10,10,"_*",1,'WW_SD_jams_on_promo_scanned']
df[3,10,10,"_t",0,'WW_BM_jams_total_scanned']=df[3,10,10,"_*",0,'WW_BM_jams_off_promo_scanned']+df[3,10,10,"_*",1,'WW_BM_jams_on_promo_scanned']
 
df=df*1000



df[1,10,10,"_t",0,'WW_BB_jams_on_promo']=(df[1,10,10,"_*",1,'WW_BB_jams_on_promo_scanned']>0.0)
df[2,10,10,"_t",0,'WW_SD_jams_on_promo']=(df[2,10,10,"_*",1,'WW_SD_jams_on_promo_scanned']>0.0)
df[3,10,10,"_t",0,'WW_BM_jams_on_promo']=(df[3,10,10,"_*",1,'WW_BM_jams_on_promo_scanned']>0.0)



df.replace(np.nan, 0.0, inplace=True)
df["","","","_t","",'weekno']= np.arange(df.shape[0])


  

#plot_query2(df,['WW_BB_jams_total_scanned'],'BB total scanned WW jam units per week')


#    test=joined_df.T.xs(p,level=3,drop_level=True)
ndf=df.droplevel(level=0,axis=1)
ndf=ndf.droplevel(level=0,axis=1)
ndf=ndf.droplevel(level=0,axis=1)
ndf=ndf.droplevel(level=0,axis=1)
ndf=ndf.droplevel(level=0,axis=1)

#print("df=\n",df)
#print("df.t=\n",df.T)

#print("ndf=\n",ndf)
#plot_query2(ndf,['WW_BB_jams_total_scanned','WW_SD_jams_total_scanned','WW_BM_jams_total_scanned'],'BB total scanned WW jam units per week')


ndf.replace(np.nan, 0.0, inplace=True)
#
ndf.reset_index('scan_week',drop=True,inplace=True)
ndf=ndf.astype(np.float64)
#print("ndf=\n",ndf)
#print("ndf.t=\n",ndf.T)

#ndf.index = pd.to_datetime(ndf.index, format = "%Y-%m-%d",infer_datetime_format=True)

#print(ndf.index)
#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='SD_on_promo',hue='BM_on_promo')  #,fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x='weekno',y='BB_total_sales',data=df,col='BM_on_promo',hue='SD_on_promo')  #,fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_SD_jams_on_promo',hue='WW_BB_jams_on_promo')  #,fit_reg=True,robust=True,legend=True) 
#save_fig("ww1",images_path)
#sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_BM_jams_on_promo',hue='WW_BB_jams_on_promo')
#save_fig("ww2",images_path)
sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_BB_jams_on_promo',hue='WW_SD_jams_on_promo')  #,fit_reg=True,robust=True,legend=True) 
save_fig("ww1")  #,images_path)
sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_BB_jams_on_promo',hue='WW_BM_jams_on_promo')
save_fig("ww2")   #,images_path)
sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_BM_jams_on_promo',hue='WW_BB_jams_on_promo')  #,fit_reg=True,robust=True,legend=True) 
save_fig("ww3")  #,images_path)
sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_SD_jams_on_promo',hue='WW_BB_jams_on_promo')
save_fig("ww4")   #,images_path)    
sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_BM_jams_on_promo',hue='WW_SD_jams_on_promo')  #,fit_reg=True,robust=True,legend=True) 
save_fig("ww5")  #,images_path)
sns.lmplot(x='weekno',y='WW_BB_jams_total_scanned',data=ndf,col='WW_SD_jams_on_promo',hue='WW_BM_jams_on_promo')
save_fig("ww6")   #,images_path)


plt.close("all")

