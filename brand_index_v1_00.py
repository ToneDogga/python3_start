#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:52:36 2020

@author: tonedogga
"""

import pandas as pd
import datetime as dt
from pandas.plotting import scatter_matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# logfile="IRI_reader_logfile.txt"
# resultsfile="IRI_reader_results.txt"
# pklsave="IRI_savenames.pkl"
# colnamespklsave="IRI_savecoldetails.pkl"
# fullcolnamespklsave="IRI_saveallcoldetails.pkl"
# dfdictpklsave="IRI_savedfdict.pkl"
# dfpklsave="IRI_fullspreadsheetsave.pkl"

wwjamsxls="IRI_ww_jams_v5.xlsx"




def plot_df(df,title,col_focus_no):    

     # print("Corr matrix1",title,"\n",df1.corr(),"\n\n")  
     # scatter_matrix(df1,alpha=1,figsize=(12,9))
     # print("Corr matrix2",title,"\n",df1.corr(),"\n\n")  
     # scatter_matrix(df2,alpha=1,figsize=(12,9))
     # print("Corr matrix3",title,"\n",df1.corr(),"\n\n")  
     # scatter_matrix(df3,alpha=1,figsize=(12,9))

 #    ax=df[df.columns[:3]].plot(x=df.columns[0],y=df.columns[1],style=["b:","r:","k:"],kind='scatter')   # actual
     #plot_numbdf[plot_number_df.columns[col_no]].plot(yerr=plot_number_df[plot_number_df.columns[col_no+1]],style='r', ecolor=['red'],errorevery=10)
  #   ax.axvline(pd.to_datetime(start_date), color='k', linestyle='--')
  #   ax.axvline(pd.to_datetime(end_date), color='k', linestyle='--')
    
  #    plt.title(title)   #str(new_plot_df.columns.get_level_values(0)))
  #    plt.legend(fontsize=11)
  #    plt.ylabel("units/week sales")
  #    plt.grid(True)
  # #   self.save_fig("actual_v_prediction_"+query_names[query_number],self.images_path)
    
  #    plt.show()
   #   if col_focus_no==0: 
   #       print("BB promos")
   #       df1=df.iloc[:,:3]
   # #  df2=df.iloc[:,1:3]
   #       df2=df.iloc[:,0::2]
   #   elif col_focus_no==1: 
   #       print("sd promos")
  #  df1=df.iloc[:,:5]
#    df2=df.iloc[:,1:3]
#    df3=df.iloc[:,0::2]
     # elif col_focus_no==2: 
     #     print("bm promos")
     #  #   df1=df.iloc[:,:2]
     #     df1=df.iloc[:,1:4]
     #     df2=df.iloc[:,0::2]
   
   #  print("df",df)
  #   print("df1",df1)
  #   print("df2",df2)
  #   print("df3",df3)
 #    df1.plot.scatter(x=df1.columns[0],y=df1.columns[1],title=title)
 #    df2.plot.scatter(x=df2.columns[0],y=df2.columns[1],title=title)
 #    df3.plot.scatter(x=df3.columns[0],y=df3.columns[1],title=title)


    sns.lmplot(x=df.columns[0],y=df.columns[1],col=df.columns[3],data=df,hue=df.columns[3],fit_reg=True,robust=True,legend=True) 
 #    sns.lmplot(x=df2.columns[0],y=df2.columns[1],data=df2,hue=df1.columns[2],fit_reg=True,robust=True,legend=True) 
  #   sns.lmplot(x=df3.columns[0],y=df3.columns[1],data=df3,fit_reg=True,robust=True) 



# def save_df(df,filename):
#     df.to_pickle(filename)
#     return


# def load_df(filename):
#     return pd.read_pickle(filename)



# class salestrans:
#     def __init__(self):   
#         self.epochs=8
#     #    self.steps_per_epoch=100 
#         self.no_of_batches=1000
#         self.no_of_repeats=2
        
#         self.dropout_rate=0.2
#         self.start_point=0
        
 
def add_dates_back(df,all_dates):
    return pd.concat((df,all_dates_df),axis=1)   #,on='ww_scan_week',how='right')

       
        
        
column_list=list(["ww_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
        
df=pd.read_excel(wwjamsxls,-1,header=0)[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
df['ww_scan_week']=pd.to_datetime(df['ww_scan_week'])
df.fillna(0,inplace=True)
df.drop_duplicates(keep='first', inplace=True)
df.sort_values(by=['ww_scan_week'], inplace=True, ascending=True)
df.set_index('ww_scan_week',inplace=True)
#first_date=df.index[0]
#last_date=df.index[-1]
#print("first dats",first_date,last_date)
#print("1df.shape",df,df.shape)   
#all_dates = pd.period_range(first_date, periods=df.shape[0],freq="W")   # 2000 days
#all_dates = pd.to_datetime(pd.date_range(first_date, periods=150,freq="W-THU"))   # 2000 days
#all_dates_df=pd.DataFrame(all_dates,columns=['ww_scan_week'])  #,index=['ww_scan_week'])
#all_dates_df['extra']=True
#all_dates_df.set_index('ww_scan_week',inplace=True)
#print("newdf.shape",new_df,new_df.shape)   

#print("all sdtae",all_dates,len(all_dates))
#newdf=df[column_list]
#print("df size=",newdf,newdf.shape,newdf.columns)
#df=df[df.columns=[bb_total_units,bb_promo_disc]]
#print(df)



df["bb_on_promo"]=(df['bb_promo_disc']>4.9)
df["sd_on_promo"]=(df['sd_promo_disc']>4.9)
df["bm_on_promo"]=(df['bm_promo_disc']>4.9)

df.drop(["bb_promo_disc","sd_promo_disc","bm_promo_disc"],axis=1,inplace=True)

no_promo_df=df.loc[(df['bm_on_promo']==False) & (df['bb_on_promo']==False) & (df['sd_on_promo']==False)]

sd_on_promo_only_df=df.loc[(df['bm_on_promo']==False) & (df['bb_on_promo']==False) & (df['sd_on_promo']==True)]
bm_on_promo_only_df=df.loc[(df['bm_on_promo']==True) & (df['bb_on_promo']==False) & (df['sd_on_promo']==False)]
bb_on_promo_only_df=df.loc[(df['bm_on_promo']==False) & (df['bb_on_promo']==True) & (df['sd_on_promo']==False)]




#print("1no promo=\n",no_promo_df)
#print("sd_df_on_prpomo=\n",sd_on_promo_only_df)
#print("2sd on promo only=\n",sd_on_promo_only_df)
#print("sd_df_on_prpomo=\n",sd_on_promo_only_df)


#plot_df(no_promo_df,"WW jams ")
#no_promo_df=add_dates_back(no_promo_df,all_dates)
#print("2no promo=\n",no_promo_df)
#column_list=["bb_total_units","sd_total_units","bm_total_units"]

#df_list=["no_promo_df","sd_on_promo_only_df","bm_on_promo_only_df"]

#for new_df in df_list:

#plot_df(no_promo_df[column_list],"baseline weeks with no promos")

df=sd_on_promo_only_df

#print("df.columns",df.columns)
sns.lmplot(x=df.columns[1],y=df.columns[2],row=df.columns[3],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=False) 
sns.lmplot(x=df.columns[1],y=df.columns[0],row=df.columns[3],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 

sns.lmplot(x=df.columns[0],y=df.columns[2],row=df.columns[3],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 
sns.lmplot(x=df.columns[0],y=df.columns[1],row=df.columns[3],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 

sns.lmplot(x=df.columns[2],y=df.columns[0],row=df.columns[3],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 
sns.lmplot(x=df.columns[2],y=df.columns[1],row=df.columns[3],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 


df=sd_on_promo_only_df

#print("df.columns",df.columns)
sns.lmplot(x=df.columns[1],y=df.columns[2],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 
sns.lmplot(x=df.columns[1],y=df.columns[0],col=df.columns[4],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 

#sns.lmplot(x=df.columns[2],y=df.columns[1],col=df.columns[5],data=df,hue=df.columns[4],fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x=df.columns[2],y=df.columns[0],col=df.columns[5],data=df,hue=df.columns[4],fit_reg=True,robust=True,legend=True) 


#sns.lmplot(x=df.columns[0],y=df.columns[2],col=df.columns[3],data=df,hue=df.columns[5],fit_reg=True,robust=True,legend=True) 
#sns.lmplot(x=df.columns[0],y=df.columns[1],col=df.columns[3],data=df,hue=df.columns[4],fit_reg=True,robust=True,legend=True) 


#sns.lmplot(x=df.columns[2],y=df.columns[2],col=df.columns[4],data=df,hue=df.columns[4],fit_reg=True,robust=True,legend=True) 



# plot_df(no_promo_df,"no promos",0)   #sd

# print("SD promos only")
# plot_df(sd_on_promo_only_df,"SD promos only",1)   #sd
# print("BM promos only")
# plot_df(bm_on_promo_only_df,"BM promos only",2)   # bm
# #plot_df(bb_on_promo_only_df[column_list],"BB promos only")


