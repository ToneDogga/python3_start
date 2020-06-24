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


colesjamsxls="Coles_jams_scan_data_300520.xlsx"
latestscannedsalescoles="Coles_IRI_portal_scan_data_170620.xlsx"

report_savename="sales_trans_report_dict.pkl"

mats=7



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

   
# class MCDropout(keras.layers.Dropout):
#      def call(inputs):
#         return super().call(inputs,training=True)


# class MCAlphaDropout(keras.layers.AlphaDropout):
#     def call(inputs):
#         return super().call(inputs,training=True)


# def last_time_step_mse(Y_true, Y_pred):
#     return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


# def plot_learning_curves(loss, val_loss,epochs,title):
#     if ((np.min(loss)<=0) or (np.max(loss)==np.inf)):
#         return
#     if ((np.min(val_loss)<=0) or (np.max(val_loss)==np.inf)):
#         return
#     if np.min(loss)>10:
#         lift=10
#     else:
#         lift=1
#     ax = plt.gca()
#     ax.set_yscale('log')
  
#     plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
#     plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
#     plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
# #    plt.axis([1, epochs+1, 0, np.max(loss[1:])])
#   #  plt.axis([1, epochs+1, np.min(loss), np.max(loss)])
#     plt.axis([1, epochs+1, np.min(loss)-lift, np.max(loss)])

#     plt.legend(fontsize=14)
#     plt.title(title)
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.grid(True)


 

# @tf.function
# def sequential_indices(start_points,length_of_indices): 
#      grid_indices=tf.meshgrid(tf.range(0,length_of_indices),start_points)   #int((end_point-start_point)/batch_length)+1)) #   print("gt=",gridtest)
#      return tf.add(grid_indices[0],grid_indices[1])   #[:,:,np.newaxis
    
 
  
#  # print("new Y shape",Y.shape)
#  # for step_ahead in range(1, predict_ahead_length + 1):
#  #     Y[...,step_ahead - 1] = series[..., step_ahead:step_ahead+batch_length-predict_ahead_length, 0]  #+1

# @tf.function
# def create_X_batches(series,batch_length,no_of_batches,start_point,end_point):
#      start_points=tf.random.uniform(shape=[no_of_batches],minval=start_point,
#                   maxval=end_point-batch_length,dtype=tf.int32)
#      return sequential_indices(start_points,batch_length)[...,tf.newaxis]
 
 

# @tf.function
# def create_X_and_y_batches(X_set,y_set,batch_length,no_of_batches):
#      indices=create_X_batches(X_set,batch_length,no_of_batches,0,X_set.shape[0])
#   #   print("X indices shape",X_indices.shape)
#   #   print("full indices shape",full_indices.shape)
                
#      X=tf.cast(tf.gather(X_set,indices[:,:-1,:],axis=0),tf.int32)
#      y=tf.cast(tf.gather(y_set,indices[:,-1:,:],axis=0),tf.int32)

#      tf.print("2X[1]=",X[1],X.shape,"\n")
#      tf.print("2y[1]=",y[1],y.shape,"\n")

#      return X,y


     




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
    figname="fig_3_"+title+".pkl"
    fig=pivot_df.plot(rot=45,grid=True,logy=False,use_index=True,fontsize=8,kind='line',stacked=stacked,title=title)
    
    pickle.dump(fig,open(figname, 'wb'))
    plt.show()
    return figname
    



def annualise_growth_rate(days,rate):
    return (((1+rate)**(365/days))-1.0)




def plot_trend(s,title):
   #  ax=s[['days_since_last_order','units']].iloc[-1].plot(x='days_since_last_order', linestyle='None', color="green", marker='.')

     fig=s[['days_since_last_order','units']].iloc[:-1].plot(x='days_since_last_order', linestyle='None', color="red", marker='o')

     s[['days_since_last_order','bestfit']].plot(x='days_since_last_order',kind="line",ax=fig)

     plt.title(title)   #str(new_plot_df.columns.get_level_values(0)))
     fig.legend(fontsize=8)
     plt.ylabel("unit sales")
     plt.grid(True)
#     self.save_fig("actual_v_prediction_"+str(plot_number_df.columns[0]),self.images_path)
     figname="fig_2_"+title+".pkl"

     pickle.dump(fig,open(figname, 'wb'))


     plt.show()
     return figname






def calculate_first_derivative(s,cust,prod):
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
        title=cust+"_"+prod+"="+str(slope)
        figname= plot_trend(s,title)
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
    dds.reset_index(inplace=True)
    #print(dds)
    dds.drop(['period'],axis=1,inplace=True)
    #print(dds)
    #dds=dds.tail(365)
    dds.tail(365)[['date','mat']].plot(x='date',y='mat',grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    print(dds[['date','mat7','diff7','30_day%','90_day%','365_day%','mat']].tail(8)) 
    fig=dds.tail(dds.shape[0]-731)[['date','30_day%','90_day%','365_day%']].plot(x='date',y=['30_day%','90_day%','365_day%'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    figname="fig_"+title+".pkl"
    
    pickle.dump(fig,open(figname, 'wb'))
    return dds,figname
    








warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format
sales_df_savename="sales_trans_df.pkl"
filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]


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

dds=sales_df.groupby(['period'])['salesval'].sum().to_frame() 
datelen=dds.shape[0]-365





################################################333

report = namedtuple("report", ["name", "report_type"])

report_type_dict=dict({0:"dictionary",
                       3:"dataframe",
                       5:"spreadsheet",
                       6:"pivottable",
                       8:"chart_filename"})


#  Report_dict is a dictionary of all the reports created plus the report_type_dict to decode it
# at the end it is picked so it can be loaded

report_dict={report("report_type_dict",0):report_type_dict}


################################################
name="Beerenberg $ GSV Annual growth rate"
print("\n",name)
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname


#########################################
name="shop GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['glset']=="SHP")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname

############################################

name="ONL GSV sales $"
print("\n",name)
shop_df=sales_df[(sales_df['glset']=="ONL")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname

############################################
name="Export GSV sales $"
print("\n",name)
shop_df=sales_df[(sales_df['glset']=="EXS")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname

############################################
name="NAT sales GSV$"

print("\n",name)
shop_df=sales_df[(sales_df['glset']=="NAT")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname

############################################
name="WW (010) GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['specialpricecat']==10)]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname

############################################
name="Coles GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['specialpricecat']==12)]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname

############################################
name="DFS GSV sales $"

print("\n",name)
shop_df=sales_df[(sales_df['glset']=="DFS")]
dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
result,figname=glset_GSV(dds,name)
report_dict[report(name,3)]=result
report_dict[report(name,8)]=figname

plt.show()
plt.close('all')
############################################

# create a sales_trans pivot table
name="Sales summary (units or ctns)"

print("\n",name)
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
#print(pivot_df) 
name="pivot_table_units.xlsx"
pivot_df.to_excel(name) 
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name



pivot_df=pd.pivot_table(sales_df, values='salesval', index=['productgroup','product'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#print(pivot_df) 
#pivot_df.plot(kind='line',stacked=True,title="Unit sales per month by productgroup")

name="pivot_table_dollars.xlsx"
pivot_df.to_excel(name) 
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name



#jam_sales_df=sales_df[sales_df['productgroup']==10]
#print("jsdf=\n",jam_sales_df)

pivot_df=pd.pivot_table(sales_df[sales_df['productgroup']<"40"], values='qty', index=['productgroup'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
name="pivot_table_units_product_group.xlsx"
figname=plot_stacked_line_pivot(pivot_df,name,False)   #,number=6)


pivot_df.to_excel(name) 
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name
report_dict[report(name,8)]=figname




pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset','specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])

#print(pivot_df) 

name="pivot_table_customers_x_glset_x_spc.xlsx"
pivot_df.to_excel(name)
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name

pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])

#print(pivot_df)  
name="pivot_table_customers_x_glset.xlsx"
pivot_df.to_excel(name)
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
name="Dollar sales per month by spc.xlsx"
figname=plot_stacked_line_pivot(pivot_df,name,False)   #,number=6)
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name
report_dict[report(name,8)]=figname


#print(pivot_df) 
pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
name="pivot_table_customers_spc_nocodes.xlsx"
pivot_df.to_excel(name)
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name
report_dict[report(name,8)]=figname


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#ax=plot_stacked_line_pivot(pivot_df,"Unit sales per month by productgroup",False)   #,number=6)

#print(pivot_df) 
name="pivot_table_customers_x_spc.xlsx"
pivot_df.to_excel(name) 
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name


pivot_df=pd.pivot_table(sales_df, values='salesval', index=['cat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
#pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
#print(pivot_df) 
name="pivot_table_customers.xlsx"
pivot_df.to_excel(name) 
report_dict[report(name,6)]=pivot_df
report_dict[report(name,5)]=name

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
report_dict[report("Top 50 customers by $purchases in the last 30 days",3)]=unique_code_pivot_df

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
report_dict[report("Top 50 customers special price category by $purchases in the last 30 days",3)]=unique_code_pivot_df






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
report_dict[report("Top 50 products by $sales in the last 30 days",3)]=unique_code_pivot_df




year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["productgroup"]).transform(sum)
year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["productgroup"]).transform(sum)
pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
unique_pg_pivot_df=pivot_df.drop_duplicates('productgroup',keep='first')

print("\nTop productgroups by $sales in the last 30 days.")
print(unique_pg_pivot_df[['productgroup','total_units','total_dollars']].head(20))
#pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 
report_dict[report("Top productgroups by $sales in the last 30 days",3)]=unique_code_pivot_df



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
report_dict[report("Top 50 Credits in past 30 days",3)]=credit_df

# find all the good performing and poor performing outliers in retail sales
#  limit product groups
product_groups_only=["10","11","12","13","14","15","18"]
spc_only=["088"]

# for each spc
# colect all the customer that have bought more than 3 products over $1000 in total over more them 3 trnsactions in the past year
#
# for each customer code, rank the sales growth of each product bought and the total sales
# with the products belonging product_groups_only
# append to a list
# sort the whole list
# highlight the top 20 growers and botom 20 losers
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
            s['slope'],figname,name=calculate_first_derivative(s,cust,prod)  
           # s['figure']=figure
          #  figure_list.append(figure)
            new_sales_df=new_sales_df.append(s)
            if (figname!="") & (name!=""):
                report_dict[report(name,8)]=figname
 
print("\n\n")
#print("\nysdf3=",new_sales_df[['date','code','product','counter','slope']],new_sales_df.shape)
new_sales_df.drop_duplicates(['code','product'],keep='first',inplace=True)
#new_sales_df=new_sales_df[new_sales_df['slope']>0.02]
new_sales_df.sort_values(['slope'],ascending=[False],inplace=True)

print("\nbest growth=",new_sales_df[['code','product','slope']].head(100).to_string())
print("\nworst growth=",new_sales_df[['code','product','slope']].tail(50).to_string())
print(new_sales_df.shape)
report_dict[report("growth rankings",3)]=new_sales_df

print("reports being pickled and saved to",report_savename)
with open(report_savename,"wb") as f:
    pickle.dump(report_dict, f,protocol=-1)
  
   

print("finished\n")




