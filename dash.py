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

import BB_data_dict as dd

import os
if dd.dash_verbose:
    pass
else:    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


# TensorFlow ≥2.0 is required
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
if dd.dash_verbose==False:
     tf.autograph.set_verbosity(0,alsologtostdout=False)   
#     tf.get_logger().setLevel('INFO')
tf.config.experimental_run_functions_eagerly(False)   #True)   # false

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

#tf.autograph.set_verbosity(3, True)




from tensorflow import keras
#from keras import backend as K

assert tf.__version__ >= "2.0"

 
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import timedelta
import calendar
import xlsxwriter

import xlrd

from pathlib import Path,WindowsPath
from random import randrange

import pickle
import multiprocessing

import warnings

from collections import namedtuple
from collections import defaultdict
from datetime import datetime
from pandas.plotting import scatter_matrix

from matplotlib import pyplot, dates
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.ticker as ticker


import time



#plt.ion() # enables interactive mode

#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)




def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./dashboard2_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)



output_dir = log_dir("dashboard2")
os.makedirs(output_dir, exist_ok=True)





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


def add_a_week(df):
    last_date=df.columns[-1]
 #   print("last date",last_date)
 #   new_date=pd.Timestamp("2015-01-01") + pd.offsets.Day(7))
    new_date=df.columns[-1] + pd.offsets.Day(7)
 #   print("new_date=",new_date)
    df[new_date]=np.nan
  #  print("df=\n",df,"\n",df.T)
    
    return df


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(output_dir, fig_id + "." + fig_extension)
  #  print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
    return



def take_out_zeros(df,cols):
    # cols of a list of column names
    df[cols]=df[cols].clip(lower=1000.0,axis=1)
    df[cols]=df[cols].replace(1000.0, np.nan)
    return df


def add_trues_and_falses(df,cols):
    df[cols]=df[cols].replace(1,True)
    df[cols]=df[cols].replace(0,False)
    return df


    
def plot_brand_index(tdf,y_col,col_and_hue,savename):    
 #   tdf=get_xs_name(df,("jams",3))
 #   print("tdf=\n",tdf.columns.to_list())
   # plot_query2(tdf,['coles_beerenberg_jams_total_scanned','coles_st_dalfour_jams_total_scanned','coles_bonne_maman_jams_total_scanned'],'beerenberg total scanned Coles jam units per week')
   # plot_query2(tdf,['woolworths_beerenberg_jams_total_scanned','woolworths_st_dalfour_jams_total_scanned','woolworths_bonne_maman_jams_total_scanned'],'BB total scanned ww jam units per week')
    tdf=tdf.astype(np.float64)
 
    

   # tdf.columns.type=np.str
 #   print(tdf.columns)
 #   print(tdf.columns.nlevels)
 #   print(tdf.columns.names)
    #tdf['coles_scan_week']=tdf.index
   #  tdf.index=tdf.index.astype(str)
    #print("tdf=\n",tdf,"\ny_col=",y_col)
  #  print(tdf.loc[:,'6'])
   # tdf=add_trues_and_falses(tdf,['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo'])
    tdf=add_trues_and_falses(tdf,col_and_hue[0])
    tdf=add_trues_and_falses(tdf,col_and_hue[1])
  #  print("tdf=\n",tdf,"\n",tdf.T)
  #  tdf=add_trues_and_falses(tdf,'7')

 #   tdf[['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo']]=tdf[['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo']].replace(1.0,True)
 #   tdf[['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo']]=tdf[['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo']].replace(0.0,False)
  #  print("tdf1=\n",tdf)
  #  tdf[['coles_bonne_maman_jams_off_promo_scanned']]=tdf[['coles_bonne_maman_jams_off_promo_scanned']].clip(lower=1000.0,axis=1)
  #  tdf[['coles_bonne_maman_jams_off_promo_scanned']]=tdf[['coles_bonne_maman_jams_off_promo_scanned']].replace(1000.0, np.nan)
   # tdf=take_out_zeros(tdf,['coles_beerenberg_jams_off_promo_scanned'])
    #tdf=take_out_zeros(tdf,[y_col])
 
   # print("tdf2=\n",tdf,type(y_col),type(col_and_hue[0]),type(col_and_hue[1]))
    
    date=pd.to_datetime(tdf.index).strftime("%Y-%m-%d").to_list()
 #   print("date=",date)
    tdf['date']=date
    tdf['dates'] = pd.to_datetime(tdf['date']).apply(lambda date: date.toordinal())
    #tdf['date_ordinal'] = date.apply(lambda date: date.toordinal())


    
  #  tdf['datenum']=dates.datestr2num(date)
   # tdf.reset_index('scan_week',drop=True,inplace=True)
 #   print("tdf=",tdf.T)
    # date = tdf['scan_week'].to_list()   #['1975-12-03','2008-08-20', '2011-03-16']
    # value = [1,4,5]
    # df = pandas.DataFrame({
    #     'date': pd.to_datetime(date),   # pandas dates
    #     'datenum': dates.datestr2num(date), # maptlotlib dates
    #     'value': value
    # })
    
    # @pyplot.FuncFormatter
    # def fake_dates(x, pos):
    #     """ Custom formater to turn floats into e.g., 2016-05-08"""
    #     return dates.num2date(x).strftime('%Y-%m-%d')
    # Tighten up the axes for prettiness
    #ax.set_xlim(df['date_ordinal'].min() - 1, df['date_ordinal'].max() + 1)
    #ax.set_ylim(0, df['amount'].max() + 1)
    fig, ax = pyplot.subplots()
    ax.set_xlabel("",fontsize=8)
   # ax.tick_params(labelrotation=45)

   # ax.legend(loc='upper left')
  
   # ax=plt.gca()
    # just use regplot if you don't need a FacetGrid\
    sns.set(font_scale=0.6)
 #   sns.lmplot(x='date_ordinal', y='coles_beerenberg_jams_off_promo_scanned', col='coles_bonne_maman_jams_on_promo',hue='coles_st_dalfour_jams_on_promo',data=tdf)   #,color="green",label="")
    sns.lmplot(x='dates', y=y_col, col=col_and_hue[0],hue=col_and_hue[1],data=tdf,legend=False)   #,color="green",label="")
    ax=plt.gca()
    #sns.regplot(x='date_ordinal', y='coles_beerenberg_jams_off_promo_scanned',data=tdf,color="green",marker=".",label="")
    ax.set_xlabel("",fontsize=8)
   # ax.tick_params(labelrotation=45)

   # plt.legend(loc='upper left',title='coles st dalfour jams on promo',fontsize=10)
    plt.legend(loc='upper left',title=col_and_hue[1],fontsize=8,title_fontsize=8)

   # save_fig("coles00") 

    #new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]
    new_labels = [dt.date.fromordinal(int(item)) for item in ax.get_xticks()]
  #  print("new_labels=",new_labels)
    improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
  #  print("improved labels",improved_labels)
    ax.set_xticklabels(improved_labels,fontsize=8)
 #   ax2=plt.gca()
  #  tdf['improved labels']=improved_labels
 #   print("tdf2=",tdf)
    # here's the magic:
   # ax.xaxis.set_major_formatter(fake_dates)
    save_fig(savename) 
    return
    



def clean_up_name(name):
    name = name.replace(',', '_')
    name = name.replace(' ', '_')
    return name.replace("'", "")




def load_sales(filenames):  # filenames is a list of xlsx files to load and sort by date
    print("load:",filenames[0])
    df=pd.read_excel(filenames[0],sheet_name="AttacheBI_sales_trans",usecols=range(0,17),verbose=False)  # -1 means all rows   
 #   print("df size=",df.shape,df.columns)
    for filename in filenames[1:]:
        print("load:",filename)
        new_df=pd.read_excel(filename,sheet_name="AttacheBI_sales_trans",usecols=range(0,17),verbose=False)  # -1 means all rows  
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
      
    #print(df.head(3))
    #print(df.tail(3))
   
 
    df["period"]=df.date.dt.to_period('D')
 #   df["period"]=df.date.dt.to_period('W-THU')

    df['period'] = df['period'].astype('category')
  #  print("load sales df=\n",df)
    
 
    return df           
 


def load_IRI(scan_data_files): 
    np.random.seed(42)
    tf.random.set_seed(42)
    
    if dd.dash_verbose:
        print("\n\nLoad scan data spreadsheets...\n")
    # scan_data_files=["jam_scan_data_2020.xlsx","cond_scan_data_2020.xlsx","sauce_scan_data_2020.xlsx"]
    # #total_columns_count=1619+797
    # scan_dict_savename="scan_dict.pkl"
    
    #output_dir = log_dir("scandata")
    #os.makedirs(output_dir, exist_ok=True)
    
        
      
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors
    
    
    #
      
        
    np.random.seed(42)
    tf.random.set_seed(42)
            
    #column_list=list(["coles_scan_week","bb_total_units","bb_promo_disc","sd_total_units","sd_promo_disc","bm_total_units","bm_promo_disc"])      
    #rename_dict=dict({"qty":"BB_total_invoiced_sales"})
    #df=df.astype(convert_dict)    
        
    
    count=1
    for scan_file in scan_data_files:
        column_count=pd.read_excel(scan_file,-1).shape[1]   #count(axis='columns')
        if dd.dash_verbose:
            print("Loading...",scan_file,"->",column_count,"columns")
        convert_dict={col: np.float64 for col in range(1,column_count-1)}   #1619
        convert_dict['index']=np.datetime64
    
        if count==1:
            df=pd.read_excel(scan_file,-1,dtype=convert_dict,index_col=0,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
            df=df.T
            df['category']=[dd.category_dict[count]]*(column_count-1)
            df = df.set_index('category', append=True)
            df=df.T
    
        else:
       #     print(convert_dict)
         #   del df2
            df2=pd.read_excel(scan_file,-1,index_col=0,header=[0,1,2])
         #   print(df2)
            df2=df2.T
            df2['category']=[dd.category_dict[count]]*(column_count-1)
            df2 = df2.set_index('category', append=True)
            df2=df2.T
       #     print(df2)
            df=pd.concat([df,df2],axis=1)   #,keys=[df['index']])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
          #  del df2
       # print(df)
        count+=1 
        
        
    
    if dd.dash_verbose:
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
    df['second_y']=plot_type
    df['reverse']=plot_type
    df['column_no']= np.arange(1, len(df.index)+1)
    df['plot_with1']=0
    df['plot_with2']=0
    df['plot_with3']=0
    df['style']="b-"
    df['linewidth']=1
    df['column_name']=market_name+" "+df.index.get_level_values('product')+" "+df.index.get_level_values('category')+" "+df.index.get_level_values('measure')

 
    
    #print(df)
    #df=df.T
    df=df.set_index('market_name', append=True)
    
    df=df.set_index('plot_type', append=True)
    df=df.set_index('stacked', append=True)
    df=df.set_index('second_y', append=True)
    df=df.set_index('reverse', append=True)
    df=df.set_index('column_no', append=True)
    df=df.set_index('plot_with1', append=True)
    df=df.set_index('plot_with2', append=True)
    df=df.set_index('plot_with3', append=True)
    df=df.set_index('style', append=True)
    df=df.set_index('linewidth', append=True)
    df=df.set_index('column_name', append=True)
 
 
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
    if dd.dash_verbose:
        print(market_list)
    #market_dict={k:market_list[k] for k in range(len(market_list))}
    #market_rename_dict={market_list[k]:k for k in range(len(market_list))}
    
    #print("\nmd=",market_dict)
    #market_dict{0:}
    #product_list=list(set(list(df.columns.levels[1])))
    #print(product_list)
    
    #product_dict={k:product_list[k] for k in range(len(product_list))}
    #product_rename_dict={product_list[k]:k for k in range(len(product_list))}
    
    #print("\npd=",product_dict)
    
    
    measure_list=list(df.columns.levels[3])
    #stacked_list=list(df.columns.levels[3])
    #print("measure list=\n",measure_list)
    
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
    df.index.set_names('second_y', level=7,inplace=True)
    df.index.set_names('reverse', level=8,inplace=True)
    df.index.set_names('column_no', level=9,inplace=True)
    df.index.set_names('plot_with1', level=10,inplace=True)  
    df.index.set_names('plot_with2', level=11,inplace=True)  
    df.index.set_names('plot_with3', level=12,inplace=True)  
    df.index.set_names('style', level=13,inplace=True)  
    df.index.set_names('linewidth', level=14,inplace=True)  
    df.index.set_names('column_name', level=15,inplace=True)  

    
 
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
    
    
    
    
    df.rename(columns=dd.market_rename_dict,level='market',inplace=True)
    
    df.rename(columns=dd.market_rename_dict2,level='market_name',inplace=True)
    
    #print("1",df.T)
    #df.rename(columns=product_rename_dict,level='product',inplace=True)
    df.rename(columns=measure_rename_dict,level='plot_type',inplace=True)
    #print("2",df.T)
    df.rename(columns=dd.measure_conversion_dict,level='plot_type',inplace=True)
    
    df.rename(columns=measure_rename_dict,level='stacked',inplace=True)
    
    df.rename(columns=dd.stacked_conversion_dict,level='stacked',inplace=True)
    
    df.rename(columns=measure_rename_dict,level='second_y',inplace=True)
    
    df.rename(columns=dd.second_y_axis_conversion_dict,level='second_y',inplace=True)
    
    df.rename(columns=measure_rename_dict,level='reverse',inplace=True)
    
    df.rename(columns=dd.reverse_conversion_dict,level='reverse',inplace=True)
    
    #df.rename(columns=df.columns,level='column_no',inplace=True)

    #print("3",df.T)
    
    
    #######################################################
    # add brand, variety and catoery to multiindex index
      #  print(c,"=",variety_type_dict[c])
    #    print(c,find_in_dict(brand_dict,c),find_in_dict(variety_type_dict,c))  
     #   print(df.loc[find_in_dict(brand_dict,c)])
    
    brand_values=[find_in_dict(dd.brand_dict,c) for c in original_df.columns.get_level_values('product')]
    #print("brands:",brand_values)
    product_values=[find_in_dict(dd.variety_type_dict,c) for c in original_df.columns.get_level_values('product')]
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
    df=df.reorder_levels(['column_no','column_name','category','market','brand','variety','plot_type','plot_with1','plot_with2','plot_with3','style','linewidth','stacked','second_y','reverse','market_name','product','measure'],axis=0).T
    return df,original_df




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
     latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
     
     fig=s[['days since last order','units']].iloc[:-1].plot(x='days since last order', linestyle='None', color="red", marker='o')

     s[['days since last order','bestfit']].plot(x='days since last order',kind="line",ax=fig)

     plt.title(title+" (slope="+str(round(slope,3))+") w/c:"+str(latest_date))  #str(new_plot_df.columns.get_level_values(0)))
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

    s['days since last order']=X
    y=s[['qty']].to_numpy()   
    y=y[::-1,0]
    s['units']=y
   
    p = np.polyfit(X[:-1], y[:-1], 1)  # linear regression 1 degree
    
    s['bestfit']=np.polyval(p, X)
    figname=""
    title=""
    slope=round(p[0],6)
    if ((slope>0.12) | (slope<-0.1)):
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
  #  dds.drop(['period'],axis=1,inplace=True)
    #print(dds)
    #dds=dds.tail(365)
    
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
    ax.ticklabel_format(style='plain')
 
    
      # plt.ticklabel_format(style='plain')

    dds.tail(365)[['dates','mat']].plot(x='dates',y='mat',xlabel="",grid=True,title=title+" w/c:("+str(latest_date)+")",ax=ax)   #),'BB total scanned vs purchased Coles jam units per week')
    print(dds[['dates','mat7','diff7','30_day%','90_day%','365_day%','mat']].tail(8)) 
#    fig=dds.tail(dds.shape[0]-731)[['dates','30_day%','90_day%','365_day%']].plot(x='dates',y=['30_day%','90_day%','365_day%'],xlabel="",grid=True,title=title+" w/c:("+str(latest_date)+")",ax=ax)   #),'BB total scanned vs purchased Coles jam units per week')
#    fig=dds.tail(dds.shape[0]-731)[['dates']].plot(x='dates',y=['30_day%','90_day%','365_day%'],xlabel="",grid=True,title=title+" w/c:("+str(latest_date)+")",ax=ax)   #),'BB total scanned vs purchased Coles jam units per week')
 
    figname="B1fig_"+title
    save_fig(figname)
    fig=dds.tail(dds.shape[0]-731)[['dates','mat']].plot(x='dates',y=['mat'],grid=True,xlabel="",title=title+" w/c:("+str(latest_date)+")",ax=ax)   #),'BB total scanned vs purchased Coles jam units per week')
    figname="B2fig_"+title
    save_fig(figname)
 
 #   pickle.dump(fig,open(output_dir+figname+".pkl", 'wb'))
    return dds[['dates','mat7','diff7','30_day%','90_day%','365_day%','mat']].tail(18),figname
    



def find_active_products(sales_df,age):  # 90 days?  retuen product codes of products sold in past {age} days
    print("sales df1=\n",sales_df)
 #   sales_df=sales_df[sales_df['date']>]
 #   sales_df['recents']=pd.Timedelta(pd.to_datetime('today') -lastone['date']).days
    sales_df['diff1']=sales_df.date.diff(periods=1)
    #print("td=",sales_df['date']-pd.to_datetime('today'))   #,unit='days'))
  #  print("saels df2=\n",sales_df)
    return []



def find_in_dict(dictname,name):
    m= [k for k, v in dictname.items() if v in name.lower()]
    if len(m)>1:
        m=m[-1]
    elif len(m)==1:
        m=m[0]
    else:
        m=0
    return m    

    



def write_excel(df,filename):
        sheet_name = 'Sheet1'
        writer = pd.ExcelWriter(filename,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')     
        df.to_excel(writer,sheet_name=sheet_name,header=False,index=False)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')     
        writer.save()
        return


def write_excel2(df,filename):
        sheet_name = 'Sheet1'
        writer = pd.ExcelWriter(filename,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')     
        df.to_excel(writer,sheet_name=sheet_name,header=True,index=True)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')     
        writer.save()
        return







def load_data(scan_data_files,scan_data_filesT): 
    np.random.seed(42)
  #  tf.random.set_seed(42)
    
    print("\n\nLoad scan data spreadsheets...\n")
         
    
    count=1
    for scan_file,scan_fileT in zip(scan_data_files,scan_data_filesT):
      #  column_count=pd.read_excel(scan_file,-1).shape[1]   #count(axis='columns')
        #if dd.dash_verbose:
        print("Loading...",scan_file)   #,scan_fileT)   #,"->",column_count,"columns")
      
       # convert_dict={col: np.float64 for col in range(1,column_count-1)}   #1619
       # convert_dict['index']=np.datetime64
    
        if count==1:
 #           df=pd.read_excel(scan_file,-1,dtype=convert_dict,index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
            dfT=pd.read_excel(scan_file,-1,header=None,engine='xlrd',dtype=object)    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)

            write_excel(dfT.T,scan_fileT)

            df=pd.read_excel(scan_fileT,-1,header=None,index_col=[1,2,3,4,5,6,7,8,9,10,11,12,13],engine='xlrd',dtype=object)  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
        else:
       #     print(convert_dict)
         #   del df2
            dfT=pd.read_excel(scan_file,-1,header=None,engine='xlrd',dtype=object)    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
            
            write_excel(dfT.T,scan_fileT)

     
            df2=pd.read_excel(scan_fileT,-1,header=None,index_col=[1,2,3,4,5,6,7,8,9,10,11,12,13],engine='xlrd',dtype=object) #,na_values={"nan":0}) 
        
            df=pd.concat([df,df2],axis=0)   #,ignore_index=True)   #levels=['plotnumber','retailer','brand','productgroup','product','variety','plottype','yaxis','stacked'])   #,keys=[df['index']])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
          #  del df2
       # print(df)
        count+=1 
    df.index.set_names('plotnumber', level=0,inplace=True)
    df.index.set_names('retailer', level=1,inplace=True)
    df.index.set_names('brand', level=2,inplace=True)
    df.index.set_names('productgroup', level=3,inplace=True)
    df.index.set_names('product', level=4,inplace=True)
    df.index.set_names('variety', level=5,inplace=True)
    df.index.set_names('plottype', level=6,inplace=True)
    df.index.set_names('plottype1', level=7,inplace=True)
    df.index.set_names('plottype2', level=8,inplace=True)
    df.index.set_names('plottype3', level=9,inplace=True)
    df.index.set_names('sortorder', level=10,inplace=True)
    df.index.set_names('colname', level=11,inplace=True)
    df.index.set_names('measure', level=12,inplace=True)
   
    
   # a = df.index.get_level_values(0).astype(str)
   # b = df.index.get_level_values(6).astype(str)

   # df.index = [a,b]
    
   
     
    df=df.T
 #   print("df0=\n",df)
    df['date']=df.iloc[:,1]
 #   print("df1=\n",df)
    colnames=df.columns.levels[0].tolist()
  #  print("colnames=",colnames)
    

    colnames = colnames[-1:] + colnames[:-3]
    df = df[colnames]
 #   print("df2=\n",df)

    df = df[df.index != 0]
  #  print("df3=\n",df)

    df.set_index('date',drop=False,append=True,inplace=True)
    df=df.reorder_levels([1,0])
   # print("df4=\n",df)
 
    df=df.droplevel(1)
    #print("df5=\n",df)
    df=df.drop('date',level='plotnumber',axis=1)
   # print("df6=\n",df)

    df.fillna(0.0,inplace=True)
     #   df=df.drop('date',level='plotnumber')

   # print("df6.cols=\n",df.columns)
    df=df.T
    write_excel2(df,"testdf.xlsx")
  #  print("df4=\n",df)
    return df






def multiple_slice_scandata(df,query):
    new_df=df.copy(deep=True)
    for q in query:
        
        criteria=q[1]
     #   print("key=",key)
     #   print("criteria=",criteria)
        ix = new_df.index.get_level_values(criteria).isin(q)
        new_df=new_df[ix]    #.loc[:,(slice(None),(criteria))]
    new_df=new_df.sort_index(level=['sortorder'],ascending=[True],sort_remaining=True)   #,axis=1)

  #  write_excel2(new_df,"testdf2.xlsx")
    return new_df





def plot_type1(df):
    # first column is unit sales off proro  (stacked)
    # second column is unit sales on promo  (stacked)
    # third is price (second y acis)
   
      
    week_freq=8
   # print("plot type1 df=\n",df)
    df=df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
    
    df=df.T
    df['date']=pd.to_datetime(df.index).strftime("%Y-%m").to_list()
    newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
    df=df.T
    df.iloc[0:2]*=1000
    #print("plot type1 df=\n",df)
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
    ax.ticklabel_format(style='plain')
   
    df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('Units/week',fontsize=9)

    line=df.iloc[2].T.plot(use_index=False,xlabel="",kind='line',rot=0,style=["g-","k-"],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    ax.right_ax.set_ylabel('$ price',fontsize=9)
    fig.legend(title="Units/week vs $ price",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.3, 1.1))
  #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
    new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
    improved_labels = ['{}\n{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
    
 #   print("improived labels=",improved_labels[0])
    improved_labels=improved_labels[:1]+improved_labels[::week_freq]
    
    
  
    ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
    ax.set_xticklabels(improved_labels,fontsize=6)

    return




def plot_type2(df,this_year_df,last_year_df):
    # first column is total units sales
    # second column is distribution 
    
    
   #3 print("plotdf.T=\n",df.T)
   # pv = pd.pivot_table(df.T, index=df.index, columns=df.index,
   #                 values='value', aggfunc='sum')
   # print("pv=\n",pv)
   # pv.plot()
   # plt.show()
      
    week_freq=8
   # print("plot type1 df=\n",df)
    this_year_df=this_year_df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
    last_year_df=last_year_df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
   
  #  df=df.T
  #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
  #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
  #  df=df.T
    this_year_df.iloc[:1]*=1000
    last_year_df.iloc[:1]*=1000

    #print("plot type1 df=\n",df)
    fig, ax = pyplot.subplots()
    ax.ticklabel_format(style='plain')
   
    
#    fig = plt.figure()
#ax1 = fig.add_subplot(111)
    ax2 = ax.twiny()



    fig.autofmt_xdate()
 
 #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('Units/week this year vs LY',fontsize=9)
    
    

    line=this_year_df.iloc[:1].T.plot(use_index=True,grid=True,xlabel="",kind='line',style=["r-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
   # current_handles0, current_labels0 = ax.get_legend_handles_labels()

    line=last_year_df.iloc[:1].T.plot(use_index=True,grid=False,xlabel="",kind='line',style=["r:"],secondary_y=False,fontsize=9,legend=False,ax=ax2)   #,ax=ax2)
   # current_handles, current_labels = plt.gca().get_legend_handles_labels()
  #  current_handles1, current_labels1 = ax2.get_legend_handles_labels()

    #if this_year_df.shape[0]>=2:
     #   line=last_year_df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b:'],secondary_y=False,fontsize=9,legend=False,ax=ax2)   #,ax=ax2)
    line=this_year_df.iloc[1:2].T.plot(use_index=True,grid=False,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    current_handles2, current_labels2 = ax.get_legend_handles_labels()
    print(current_labels2)
   # print(current_handles2[0].current_labels2[0])
    # if df.shape[0]>=3:
   #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
#  ax.set_ylabel('Units/week',fontsize=9)

    ax.right_ax.set_ylabel('Distribution this year',fontsize=9)
  #  current_handles, current_labels = plt.gca().get_legend_handles_labels()
   # print("cl=",current_labels,line)
   # current_labels=current_labels+" Last year"
# sort or reorder the labels and handles
#reversed_handles = list(reversed(current_handles))
#reversed_labels = list(reversed(current_labels))

# call plt.legend() with the new values
#plt.legend(reversed_handles,reversed_labels)
    
        
        
    fig.legend([current_labels2[0],current_labels2[0]+" last year","Distribution (sold)"],title="Units/week TY vs LY",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.2, 1.1))
  #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
  #  new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
  #  improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
  #  improved_labels=improved_labels[::week_freq]
  
  #  ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
  #  ax.set_xticklabels(improved_labels,fontsize=8)

    return





def plot_type3(df):
       # first column is total units sales
    # second column is distribution 
    
   
      
    week_freq=8
   # print("plot type1 df=\n",df)
    df=df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
    
  #  df=df.T
  #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
  #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
  #  df=df.T
 #   df.iloc[:1]*=1000
 #   print("plot type3 df=\n",df)
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
    ax.ticklabel_format(style='plain')
   
 #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('$ Price',fontsize=9)

    line=df.T.plot(use_index=True,xlabel="",kind='line',style=["g-","r-","b-","k-","c-","m-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

  #  if df.shape[0]>=2:
   # line=df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

   # if df.shape[0]>=3:
   #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
#  ax.set_ylabel('Units/week',fontsize=9)

 #   ax.right_ax.set_ylabel('Units/week',fontsize=9)
    fig.legend(title="$ Price",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.3, 1.1))
  #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
  #  new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
  #  improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
  #  improved_labels=improved_labels[::week_freq]
  
  #  ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
  #  ax.set_xticklabels(improved_labels,fontsize=8)

   # return


   # print("plot 3")
    return




def plot_type4(df):
          # first column is total units sales
    # second column is distribution 
    
   
      
    week_freq=8
   # print("plot type1 df=\n",df)
    df=df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
    
  #  df=df.T
  #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
  #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
  #  df=df.T
    df.iloc[:]*=1000
 #   print("plot type3 df=\n",df)
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
    ax.ticklabel_format(style='plain')
   
 #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('Units/week',fontsize=9)

    line=df.T.plot(use_index=True,xlabel="",kind='line',style=["b-","r-","g-","k-","c-","m-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

  #  if df.shape[0]>=2:
  #  line=df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

   # if df.shape[0]>=3:
   #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
#  ax.set_ylabel('Units/week',fontsize=9)

  #  ax.right_ax.set_ylabel('Units/week',fontsize=9)
    fig.legend(title="Units/week",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.3, 1.1))
  #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
  #  new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
  #  improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
  #  improved_labels=improved_labels[::week_freq]
  
  #  ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
  #  ax.set_xticklabels(improved_labels,fontsize=8)

   # return



    return








def plot_slices(df):
 #   df.replace(0.0,np.nan,inplace=True)
        
      #   print(new_df)
    plottypes=list(set(list(set(df.index.get_level_values('plottype').astype(str).tolist()))+list(set(df.index.get_level_values('plottype1').astype(str).tolist()))+list(set(df.index.get_level_values('plottype2').astype(str).tolist()))+list(set(df.index.get_level_values('plottype3').astype(str).tolist()))))
   #     plottypes=list(set([p for p in plottypes if p!='0']))
   #     print("plotypes=",plottypes)
    for pt in plottypes:  
        plotnumbers=list(set(df.index.get_level_values('plotnumber').astype(str).tolist()))
        new_df=pd.concat((multiple_slice_scandata(df,[(pt,'plottype')]) ,multiple_slice_scandata(df,[(pt,'plottype1')]),multiple_slice_scandata(df,[(pt,'plottype2')]),multiple_slice_scandata(df,[(pt,'plottype3')])),axis=0)   #,(pt,'plottype1')])

 #   print("plotn",plotnumbers)
        for pn in plotnumbers:
            if (pt=='3') | (pt=='4') | (pt=='5') | (pt=='9'):
                plot_df=new_df
            else:
                plot_df=multiple_slice_scandata(new_df,[(pn,'plotnumber')])

         #   print("plot_df=\n",plot_df)
            plot_df.replace(0.0,np.nan,inplace=True)
            last_year_plot_df=plot_df.iloc[:,-(dd.e_scandata_number_of_weeks+52):-(dd.e_scandata_number_of_weeks-1)]
            this_year_plot_df=plot_df.iloc[:,-dd.e_scandata_number_of_weeks:]    
         #   print("this year plot df=",this_year_plot_df)
         #   print("last year plot df=",last_year_plot_df)
            if str(pt)=='1':   #standard plot type
                plot_type1(plot_df)
            elif str(pt)=='2':   #stacked bars plus right axis price
                plot_type2(df,this_year_plot_df,last_year_plot_df)
            elif str(pt)=='3':   # 
                plot_type3(plot_df)
            elif str(pt)=='4':   #unused 
                plot_type4(plot_df)
            elif str(pt)=='0':
                pass
            save_fig("slice_",pn+"_"+pt+"_"+str(randrange(999)))
      #      plt.show()
            
             
    plt.close('all')
    return





def graph_sales_year_on_year(sales_df,title,left_y_axis_title):
    prod_sales=sales_df[['salesval']].resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
    #print("prod sales1=\n",prod_sales)
    
    year_list = prod_sales.index.year.to_list()
    week_list = prod_sales.index.week.to_list()
    month_list = prod_sales.index.month.to_list()
    
    prod_sales['year'] = year_list   #prod_sales.index.year
    prod_sales['week'] = week_list   #prod_sales.index.week
    prod_sales['monthno']=month_list
    prod_sales.reset_index(drop=True,inplace=True)
    prod_sales.set_index('week',inplace=True)
    
    week_freq=4.3
    #print("prod sales3=\n",prod_sales)
    weekno_list=[str(y)+"-W"+str(w) for y,w in zip(year_list,week_list)]
    #print("weekno list=",weekno_list,len(weekno_list))
    prod_sales['weekno']=weekno_list
    yest= [dt.datetime.strptime(str(w) + '-3', "%Y-W%W-%w") for w in weekno_list]    #wednesday
    
    #print("yest=",yest)
    prod_sales['yest']=yest
    improved_labels = ['{}'.format(calendar.month_abbr[int(m)]) for m in list(np.arange(0,13))]
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
    ax.ticklabel_format(style='plain')
    styles=["b-","r:","g:","m:","c:"]
    new_years=list(set(prod_sales['year'].to_list()))
    #print("years=",years,"weels=",new_years)
    for y,i in zip(new_years[::-1],np.arange(0,len(new_years))):
        test_df=prod_sales[prod_sales['year']==y]
      #  print(y,test_df)
        fig=test_df[['salesval']].plot(use_index=True,grid=True,style=styles[i],xlabel="",ylabel=left_y_axis_title,ax=ax,title=title,fontsize=8)
     
    ax.legend(new_years[::-1],fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))   #,byweekday=mdates.SU)
    ax.set_xticklabels([""]+improved_labels,fontsize=8)
    figname=title
    save_fig(figname)
 
    #  save_fig
    return
   
  



     
   
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






def predict_order(joined_df,X_set_full,y_set_full,predrec,model):    #inv_hdf,mat_hdf,rec,model):
    scanned_sales=X_set_full.reshape(-1,1)[np.newaxis,...]
    Y_pred=np.stack(model(scanned_sales[:,-2,:]).numpy(),axis=2) #for r in range(scanned_sales.shape[1])]
  #  print("Y_pred",Y_pred,Y_pred.shape)
    j=np.concatenate((y_set_full[:-1],Y_pred[0,:,0]),axis=0)
  #  print("j=",j,j.shape)
  #  print("joined_df=\n",joined_df,joined_df.shape)
    joined_df=joined_df.T
    joined_df[predrec]=j  #[0,0]  #:np.concatenate((y_set[1:],Y_pred[0,:,0]),axis=0)
    joined_df=joined_df.T
    joined_df=joined_df.sort_index()
 #   print("joined_df2=\n",joined_df,joined_df.shape)

    return joined_df
    
   
    
   
def plot_prediction(df,title,latest_date):    
 #   dates=hdf.index.tolist()[7:]
    #print("dates:",dates,len(dates))
  #  df=pd.DataFrame({title+'_total_scanned':X_pred,title+'_ordered_prediction':Y_pred,title+'_total_invoiced_shifted_3wks':y_invoiced},index=dates)
   # df=pd.DataFrame({title+'_total_scanned':X_pred,title+'_ordered_prediction':Y_pred},index=dates)
 
    #shifted_df=df.shift(1, freq='W')   #[:-3]   # 3 weeks
    latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
    
    #df=gdf[['coles_BB_jams_total_scanned','all_BB_coles_jams_predicted']].rolling(mat,axis=0).mean()
    df=df.droplevel(['type'])
    df=df.sort_index()
  #  print("plor pred=\n",df)
  #  df.replace(0.0,np.nan,inplace=True)    # don't plot zero values
    df=df.T

  #  styles1 = ['b-','r:']
    styles1 = ['g:','r:','b-']
           # styles1 = ['bs-','ro:','y^-']
    linewidths = 1  # [2, 1, 4]
   # print("df=\n",df,df.shape)
    ax=plt.gca()
    df.iloc[-26:].plot(grid=True,title=title[:42]+" w/commencing:"+str(latest_date),style=styles1, lw=linewidths,ax=ax,fontsize=10)
    #plt.pause(0.001)
    
    #df.iloc[-6:].plot(grid=True,title=title,style=styles1, lw=linewidths)
    #plt.pause(0.001)
  #  ax.title(fontsize=10)
    ax.legend(title="")
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
    return 




def train_model(name,X_set,y_set,batch_length,no_of_batches,epochs,count):
   
    X,y=create_X_and_y_batches(X_set,y_set,batch_length,no_of_batches)
    
    
    
    ##########################3
    
    dataset=tf.data.Dataset.from_tensor_slices((X,y)).cache().repeat(dd.no_of_repeats)
    dataset=dataset.shuffle(buffer_size=1000,seed=42)
    
    train_set=dataset.batch(1).prefetch(1)
    valid_set=dataset.batch(1).prefetch(1)
       
     
    
    ##########################
    print(count,"Training with GRU :",name)
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
     
    if dd.dash_verbose:
        model.summary() 
     
    #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),self.MyCustomCallback()]
     
    history = model.fit(train_set ,verbose=0, epochs=epochs,
                         validation_data=(valid_set))  #, callbacks=callbacks)
    if dd.dash_verbose:     
        print("\nsave model :"+name+"_predict_model.h5\n")
    model.save(output_dir+name+"_sales_predict_model.h5", include_optimizer=True)
           
    plot_learning_curves(history.history["loss"], history.history["val_loss"],dd.epochs,"GRU :"+name)
    save_fig(name+"GRU learning curve")  #,images_path)
      
  #  plt.show()
    plt.close("all")
    return model




 

    
    
def main():            
    warnings.filterwarnings('ignore')
    pd.options.display.float_format = '{:.4f}'.format
      
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors
    visible_devices = tf.config.get_visible_devices('GPU') 

    print("\nDash : Beerenberg TF2 Salestrans analyse/predict dashboard- By Anthony Paech 25/5/20")
    print("=================================================================================================\n")       

    if dd.dash_verbose:
        print("Python version:",sys.version)
        print("\ntensorflow:",tf.__version__)
        #    print("eager exec:",tf.executing_eagerly())      
        print("keras:",keras.__version__)
        print("numpy:",np.__version__)
        print("pandas:",pd.__version__)
        print("matplotlib:",mpl.__version__)      
        print("sklearn:",sklearn.__version__)         
        print("\nnumber of cpus : ", multiprocessing.cpu_count())            
        print("tf.config.get_visible_devices('GPU'):",visible_devices)
        print("\n=================================================================================================\n")       
       
    
   
       
    np.random.seed(42)
    tf.random.set_seed(42)
      
    ##############################################################################
    
    warnings.filterwarnings('ignore')
    pd.options.display.float_format = '{:.2f}'.format

    
    
    ###################################################################################
    
    
    
    
    # try:
    #     with open("stock_level_query.pkl","rb") as f:
    #        stock_df=pickle.load(f)
    # except:
    if dd.dash_verbose:
        print("load:",dd.stock_level_query)
    stock_df=pd.read_excel(dd.stock_level_query)    # -1 means all rows   
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
    #stock_df=stock_df[(stock_df['qtyinstock']<=2000) & (stock_df['recent']==True) & ((stock_df['productgroup']>=10) & (stock_df['productgroup']<=17))]  # | (stock_df['productgroup']==12) | (stock_df['productgroup']==13) | (stock_df['productgroup']==14) | (stock_df['productgroup']<=17))]
    stock_df=stock_df[(stock_df['recent']==True) & (stock_df['qtyinstock']<=dd.low_stock_limit) & ((stock_df['productgroup']>=10) & (stock_df['productgroup']<=17))]  # | (stock_df['productgroup']==12) | (stock_df['productgroup']==13) | (stock_df['productgroup']==14) | (stock_df['productgroup']<=17))]
                    
    stock_report_df=stock_df[['productgroup','code','lastsalesdate','qtyinstock']].sort_values(['productgroup','qtyinstock'],ascending=[True,True])
    
    stock_report_df.replace({'productgroup':dd.productgroups_dict},inplace=True)
    
    print("Low stock report (below",dd.low_stock_limit,"units)\n",stock_report_df.to_string())
    
    #####################################
    
    # try:
    #     with open("production_made.pkl","rb") as f:
    #        production_made_df=pickle.load(f)
    # except:  
    if dd.dash_verbose:    
        print("load:",dd.production_made_query)
    production_made_df=pd.read_excel(dd.production_made_query,sheet_name=dd.production_made_sheet)    # -1 means all rows   
    with open("production_made.pkl","wb") as f:
        pickle.dump(production_made_df, f,protocol=-1)
    #print("stock df size=",stock_df.shape,stock_df.columns)
    #
        
    #stock_report_df=stock_df[['code','lastsalesdate','qtyinstock']].sort_values('lastsalesdate',ascending=True)
    #print(production_schedule_df.columns)
    production_made_df=production_made_df[['to_date','jobid','code','qtybatches','qtyunits']].sort_values('to_date',ascending=True)
    print("\nProduction recently made:\n",production_made_df.tail(50))
    
    
    
    
    
    ###############################
    
    # try:
    #     with open("production_planned.pkl","rb") as f:
    #        production_planned_df=pickle.load(f)
    # except:  
    if dd.dash_verbose:    
        print("load:",dd.production_planned_query)
    production_planned_df=pd.read_excel(dd.production_planned_query,sheet_name=dd.production_planned_sheet)    # -1 means all rows   
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
    print("\nProduction planned:\n",production_planned_df.head(50))
    
    ######################################################################
    if dd.dash_verbose:
        print("\n============================================================================\n")  
      
     
    ###################################################    
     
    
    #with open(dd.sales_df_savename,"rb") as f:
    sales_df=pd.read_pickle(dd.sales_df_savename)
    #    # sales_df=pickle.load(f)
    
    print("\n\nsales shape df=\n",sales_df.shape)
    
    first_date=sales_df['date'].iloc[-1]
    last_date=sales_df['date'].iloc[0]
    
    print("\nAttache sales trans analysis.  Current save is:")
    
    
    print("Data available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")
  
    
    answer3="n"
    answer3=input("\nCreate distribution report and sales trends? (y/n)")
    #answer3="y"
    
    
    answer2="n"
    answer2=input("Predict next weeks Coles and WW orders from scan data? (y/n)")
    
    answer="y"
    answer=input("Refresh salestrans?")
    
    start_timer = time.time()

############################################################################    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #  load enhanced
    
    
    if dd.dash_verbose:
        print("\n============================================================================")  
    # Big IRI scan data spreadsheets 
        print("\nLoad enhanced scan data:",dd.e_scandatalist,"......")
     
         
    df=load_data(dd.e_scandatalist,dd.transposed_datalist)
    
  #  print("df=\n",df)
####

 #   pdf=df.copy(deep=True)
    print("graphing scan data...")
  #  print("pdf=\n",pdf)
#    print(pdf.loc[multiple_slice_scandata(pdf,query=[('12','retailer'),('9','plottype3'),('11','plottype2'),('Wks on Promotion >= 5 % 6 wks','measure')])==1])
   # df.loc[df['a'] == 1,'b']
    #  we need to nan out vlaues where Beerenberg is on promotion
#
 #   print("coles new_pdf1=\n",new_pdf)
  
    
    pdf=df.copy(deep=True)
  #  print("graphing scan data...")
  #  print("pdf=\n",)
    pdf=pdf.iloc[:,-dd.brand_index_weeks_going_back:]      # remove first 20 weeks
    
 
    
 #  print("pdf=\n",pdf)
    new_pdf=multiple_slice_scandata(pdf,query=[('9','plottype3')])
    #  we need to nan out vlaues where Beerenberg is on promotion

  #  print("new_pdf2=\n",new_pdf)
 
 #   new_pdf=multiple_slice_scandata(pdf,query=[('11','plottype2')])
    #print(new_pdf.xs('11',axis=0,level='plottype2',drop_level=False))
   # print("new_pdf2=\n",new_pdf)
    
    new_pdf=new_pdf.droplevel([0,1,2,3,4,5,6,7,8,9,10])
    column_names=['-'.join(tup) for tup in new_pdf.index]
 #   print("colnames=",column_names)
 #   print("new_pdf2=\n",new_pdf)
   # new_pdf=new_pdf.T
   # new_pdf['name']=str(new_pdf.columns.get_level_values('colname')) + " "+str(new_pdf.columns.get_level_values('measure'))
    #new_pdf=new_pdf.T
    new_pdf = new_pdf.reset_index(level=[0,1],drop=True)  #'sortorder'])
   # new_pdf=new_pdf.set_index('sortorder')
     #new_pdf=new_pdf.droplevel([0])
    new_pdf=new_pdf.T
  #  print("newpdf2=\n",new_pdf.columns)
    newcols_dict={k:v for k,v in zip(new_pdf.columns,column_names)}
  #  print("newcols dict=\n",newcols_dict)
   # new_pdf.rename(columns={1001: '1001', 1010: '1010', 1012:'1012',1018:'1018'}, inplace=True)
    new_pdf.rename(columns=newcols_dict, inplace=True)

    



  #  print("new_pdf3.T=\n",new_pdf.T)
 #   plot_brand_index(new_pdf,y_col=('Coles Beerenberg all jams','Units (000) Sold off Promotion >= 5 % 6 wks'),col_and_hue=[('Coles Bonne Maman all jams','Wks on Promotion >= 5 % 6 wks'),('Coles St Dalfour all jams','Wks on Promotion >= 5 % 6 wks')],savename="coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
    plot_brand_index(new_pdf,y_col='Coles Beerenberg all jams-Units Sold off Promotion >= 5 % 6 wks',col_and_hue=['Coles Bonne Maman all jams-Wks on Promotion >= 5 % 6 wks','Coles St Dalfour all jams-Wks on Promotion >= 5 % 6 wks'],savename="brand index coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles

  #  plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo'],savename="coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
    plot_brand_index(new_pdf,y_col='Woolworths Beerenberg all jams-Units Sold off Promotion >= 5 % 6 wks',col_and_hue=['Woolworths Bonne Maman all jams-Wks on Promotion >= 5 % 6 wks','Woolworths St Dalfour all jams-Wks on Promotion >= 5 % 6 wks'],savename="brand index woolworths1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles

  #  plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_beerenberg_jams_off_promo_scanned',col_and_hue=['woolworths_bonne_maman_jams_on_promo','woolworths_st_dalfour_jams_on_promo'],savename="woolworths1")

 #('Woolworths Beerenberg all jams','Units (000) Sold off Promotion >= 5 % 6 wks')
 # ('Woolworths Beerenberg all jams',  'Wks on Promotion >= 5 % 6 wks')
 # ('Woolworths St Dalfour all jams',  'Units (000) Sold off Promotion >= 5 % 6 wks' )
 # ('Woolworths St Dalfour all jams',  'Wks on Promotion >= 5 % 6 wks')
 # ('Woolworths Bonne Maman all jams','Units (000) Sold off Promotion >= 5 % 6 wks' )
 # ('Woolworths Bonne Maman all jams', 'Wks on Promotion >= 5%  6 wks')
 
  #('Coles Beerenberg all jams','Units (000) Sold off Promotion >= 5 % 6 wks')
 # ('Coles Beerenberg all jams',  'Wks on Promotion >= 5% 6 wks')
 # ('Coles St Dalfour all jams',  'Units (000) Sold off Promotion >= 5 % 6 wks' )
 # ('Coles St Dalfour all jams',  'Wks on Promotion >= 5% 6 wks')
 # ('Coles Bonne Maman all jams','Units (000) Sold off Promotion >= 5 % 6 wks' )
 # ('Coles Bonne Maman all jams', 'Wks on Promotion >= 5% 6 wks')
    pdf=df.copy(deep=True)
  #  print("graphing scan data...")
  #  print("pdf=\n",pdf)
    new_pdf=multiple_slice_scandata(pdf,query=[('4','plottype1'),('1','brand')])
    
  #  print("new_pdf1=\n",new_pdf)
    new_pdf=new_pdf.droplevel([0,1,2,3,4,5,6,7,8,9,10])
    #column_names=['-'.join(tup) for tup in new_pdf.index]
    column_names=[tup[0]+" total scanned" for tup in new_pdf.index]

 #   print("colnames=",column_names)
 #   print("new_pdf2=\n",new_pdf)
   # new_pdf=new_pdf.T
   # new_pdf['name']=str(new_pdf.columns.get_level_values('colname')) + " "+str(new_pdf.columns.get_level_values('measure'))
    #new_pdf=new_pdf.T
    new_pdf = new_pdf.reset_index(level=[0,1],drop=True)  #'sortorder'])
   # new_pdf=new_pdf.set_index('sortorder')
     #new_pdf=new_pdf.droplevel([0])
    scan_sales_df=new_pdf.T
  #  print("newpdf2=\n",new_pdf.columns)
    newcols_dict={k:v for k,v in zip(scan_sales_df.columns,column_names)}
  #  print("newcols dict=\n",newcols_dict)
   # new_pdf.rename(columns={1001: '1001', 1010: '1010', 1012:'1012',1018:'1018'}, inplace=True)
    scan_sales_df.rename(columns=newcols_dict, inplace=True)
  #  print("scan sales df=\n",scan_sales_df)  #,"\n",new_pdf.T)

 
 
 
####

    
    if dd.e_scandata_number_of_weeks>0 & dd.e_scandata_number_of_weeks+53<df.shape[1]:
        df=df.iloc[:,-(dd.e_scandata_number_of_weeks+53):]
   #     print("df=\n",df)
    #new_df=slice_scandata(df,key='1',criteria='brand')
    #print("ss=",new_df)
    #new_df=multiple_slice_scandata(df,key=['1'],criteria='brand')
    #print("ms-",new_df)
    
        for q in dd.e_scandata_plotqueries:
            plot_slices(multiple_slice_scandata(df,query=q)) #   key=['1'],criteria='brand')
    #print("ms2",new_df)
    
    #print(new_df.columns,"\n",new_df.index)
        #       plot_slices(multiple_slice_scandata(df,query=[('1','brand'),('10','productgroup')])) #   key=['1'],criteria='brand')
   
  #   plot_slices(new_df)
       
    
  #   new_df=multiple_slice_scandata(df,query=[('10','retailer'),('0','variety')]) #   key=['1'],criteria='brand')
  # # print("ms2",new_df)
    
  #   #print(new_df.columns,"\n",new_df.index)
          
  #   plot_slices(new_df)
  #   new_df=multiple_slice_scandata(df,query=[('12','retailer'),('0','variety')]) #   key=['1'],criteria='brand')
  # # print("ms3",new_df)
    
  #   #print(new_df.columns,"\n",new_df.index)
          
  #   plot_slices(new_df)
 
    
  #   new_df=multiple_slice_scandata(df,query=[('10','retailer'),('1','variety')]) #   key=['1'],criteria='brand')
  # # print("ms2",new_df)
    
  #   #print(new_df.columns,"\n",new_df.index)
          
  #   plot_slices(new_df)
  #   new_df=multiple_slice_scandata(df,query=[('12','retailer'),('1','variety')]) #   key=['1'],criteria='brand')
  # # print("ms3",new_df)
    
  #   #print(new_df.columns,"\n",new_df.index)
          
  #   plot_slices(new_df)

    
 
    print("Enhanced scandata graphing finished...\n\n")
       

    
    
    
    
    #######################################
    if dd.dash_verbose:
        print("\n============================================================================")  
    # Big IRI scan data spreadsheets 
        print("\nLoad IRI all scan data2:",dd.scan_data_files,"......")
    
    
    df,original_df=load_IRI(dd.scan_data_files)
    if dd.dash_verbose:
        print("IRI shape=",df.shape,"\n")
    
    # new_level_name = "brand"
    # new_level_labels = ['p']
    #df1 = pd.DataFrame(data=1,index=df.index, columns=new_level_labels).stack()
    #df1.index.names = [new_level_name,'market','product','measure']
    #df=df.T.index.names=['brand','market','product','measure']
    
  #  print(df)
  #  print("\n",df.T)
  #  original_df=pd.DataFrame([])
    #full_index_df=recreate_full_index(df)
    #print(full_index_df)
    
    scan_dict={"original_df":original_df,
                "final_df":df,
                'scan_sales_df':scan_sales_df,
      #          "full_index_df":full_index_df,
           #     "market_rename_dict":dd.market_rename_dict,
            #   "product_dict":product_dict,
                "measure_conversion_dict":dd.measure_conversion_dict,
                "stacked_conversion_dict":dd.stacked_conversion_dict,
                'plot_type_dict':dd.plot_type_dict,
                'brand_dict':dd.brand_dict,
                'category_dict':dd.category_dict,
                'spc_dict':dd.spc_dict,
                'salesrep_dict':dd.salesrep_dict,
                'series_type_dict':dd.series_type_dict,
                'productgroups_dict':dd.productgroups_dict,
                'productgroup_dict':dd.productgroup_dict,
                'variety_type_dict':dd.variety_type_dict,
                'second_y_axis_conversion_dict':dd.second_y_axis_conversion_dict,
                'reverse_conversion_dict':dd.reverse_conversion_dict}
    
    
    with open(dd.scan_dict_savename,"wb") as f:
        pickle.dump(scan_dict,f,protocol=-1)
        
    ##############################################################    
    
    with open(dd.scan_dict_savename, 'rb') as g:
        scan_dict = pickle.load(g)
    
    
    
    # print("final_df shape:",scan_dict['final_df'].shape)
    # print("\n\n********************************************\n")
    # print("unknown brands=")
    # try:
    #     print(df.xs(0,level='brand',drop_level=False,axis=1))
    # except:
    #     print("no unknown brands\n")
    # #print("unknown variety")
    # #try:
    # #    print(df.xs(0,level='variety',drop_level=False,axis=1))
    # #except:
    # #    print("no unknown varieis\n")    
    # print("unknown measure type=")
    # try:
    #     print(df.xs(0,level='measure',drop_level=False,axis=1))
    # except:
    #     print("no unknown measures")
    
    
    # #
    # print("\n\n")
    # print("All scandata dataframe saved to",dd.scan_dict_savename,":\n",scan_dict['final_df'])
    
    
    # if dd.dash_verbose:
    #     print("\n============================================================================\n")  
      
     
    # ###################################################    
     
    
    # #with open(dd.sales_df_savename,"rb") as f:
    # sales_df=pd.read_pickle(dd.sales_df_savename)
    # #    # sales_df=pickle.load(f)
    
    # print("sales shape df=\n",sales_df.shape)
    
    # first_date=sales_df['date'].iloc[-1]
    # last_date=sales_df['date'].iloc[0]
    
    # print("\nAttache sales trans analysis.  Current save is:")
    
    
    # print("Data available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")
    # # #print("\n\n")   

    # answer3="n"
    # answer3=input("Create distribution report and sales trends? (y/n)")
    # #answer3="y"
    
    
    # answer2="n"
    # answer2=input("Predict next weeks Coles and WW orders from scan data? (y/n)")
    
    # answer="y"
    # answer=input("Refresh salestrans?")
    
    # start_timer = time.time()

    
    
    
    if answer=="y":
        sales_df=load_sales(dd.filenames)  # filenames is a list of xlsx files to load and sort by date
  #      with open(dd.sales_df_savename,"wb") as f:
  #            pickle.dump(sales_df, f,protocol=-1)
        sales_df.to_pickle(dd.sales_df_savename,protocol=-1)          
    
 #   print("\n")    
    sales_df.sort_values(by=['date'],ascending=True,inplace=True)
    last_date=sales_df['date'].iloc[-1]
    first_date=sales_df['date'].iloc[0]
    
    print("Attache sales trans analysis up to date.  New save is:",dd.sales_df_savename)
    print("Data available:",sales_df.shape[0],"records.\nfirst date:",first_date,"\nlast date:",last_date,"\n")
    #print(sales_df)
    
    
   #####################################################################     
    
    yearly_sales_df=sales_df.copy()
    
    yearly_sales_df['date']=pd.to_datetime(yearly_sales_df.date)
    #sales_df['week']=pd.to_datetime(sales_df.date,format="%m-%Y")
    
    #sales_df['year']=sales_df.date.year()
    yearly_sales_df.set_index('date',inplace=True)
  #  print("yearly sales df1=\n",yearly_sales_df)   #,sales_df.T) 
    graph_sales_year_on_year(yearly_sales_df,"A YonY Total $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['glset']=='EXS'],"A YonY Export $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['glset']=='ONL'],"A YonY Online $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['glset']=='NAT'],"A YonY National $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['glset']=='SHP'],"A YonY Shop $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['glset']=='DFS'],"A YonY DFS $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['specialpricecat']==10.00],"A YonY Woolworths (10) $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['specialpricecat']==12.00],"A YonY Coles (12) $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['specialpricecat']==88.00],"A YonY (088) $ sales per week","$/week")
    graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['specialpricecat']==122.00],"A YonY Harris farm (122) $ sales per week","$/week")
    

    
    
    
    ################################################
    
    dds=sales_df.groupby(['period'])['salesval'].sum().to_frame() 
    datelen=dds.shape[0]-365
    
    
    
    
    
    name="Beerenberg GSV MAT"
    print(name)
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()
    latest_date=dds['dates'].max()
    title=name+" w/c:("+str(latest_date)+")"
    
    dds.reset_index(inplace=True)
     #print(dds)
    #dds.drop(['period'],axis=1,inplace=True)
     
    fig=dds.tail(dds.shape[0]-731)[['dates','mat']].plot(x='dates',y=['mat'],grid=True,xlabel="",title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    figname="Bfig1_"+name
    save_fig(figname)
    dds[['dates','mat']].to_excel(output_dir+name+".xlsx") 
    
    #dds=sales_df.groupby(['period'])['salesval'].sum().to_frame() 
    
    name="Beerenberg GSV Annual growth rate"
    print("\n",name)
    title=name+" w/c:("+str(latest_date)+")"

    dds_mat=dds.groupby(['dates'])['salesval'].sum().to_frame() 
    result,figname=glset_GSV(dds_mat,name)
    dd.report_dict[dd.report(name,3,"_*","_*")]=result
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    result.to_excel(output_dir+name+".xlsx",engine='xlsxwriter') 
    
    
    #########################################
    name="shop GSV sales $"
    
    print("\n",name)
    shop_df=sales_df[(sales_df['glset']=="SHP")].copy()
    dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()

   # shop_df['mat']=shop_df['salesval'].rolling(365,axis=0).sum()
 #   print(shop_df)
   # shop_df['period']=pd.to_datetime(shop_df['date'],format="%d-%m-%Y")
 #   print("2=\n",shop_df)
   # shop_df.reset_index(inplace=True)
   # latest_date=dds['date'].max()
 #   title=name+" w/c:("+str(latest_date)+")"
    #shop_df.set_index('date',inplace=True)

#    print(shop_df.tail(shop_df.shape[0]-731)[['period','mat']])
  #  fig=shop_df.tail(shop_df.shape[0]-731)[['date','mat']].plot(x='date',y='mat',use_index=False,grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
  #  fig=shop_df.tail(shop_df.shape[0]-731)[['period','mat']].plot(x='period',y='mat',use_index=False,grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')

#    figname="Bfig_"+name
#    save_fig(figname)
    
   # dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    result,figname=glset_GSV(dds,name)
    dd.report_dict[dd.report(name,3,"CASHSHOP","_*")]=result
    dd.report_dict[dd.report(name,8,"CASHSHOP","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    result.to_excel(output_dir+name+".xlsx",engine='xlsxwriter') 
    
    ############################################
    
    name="ONL GSV sales $"
    print("\n",name)
    shop_df=sales_df[(sales_df['glset']=="ONL")].copy()
    dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()

    # fig=dds.tail(dds.shape[0]-731)[['date','mat']].plot(x='date',y=['mat'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    # figname="Afig_"+name
    # save_fig(figname)

    
    
    
    result,figname=glset_GSV(dds,name)
    dd.report_dict[dd.report(name,3,"_*","_*")]=result
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    result.to_excel(output_dir+name+".xlsx") 
    
    ############################################
    name="Export GSV sales $"
    print("\n",name)
    shop_df=sales_df[(sales_df['glset']=="EXS")]
    dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()

    
 
    # fig=dds.tail(dds.shape[0]-731)[['date','mat']].plot(x='date',y=['mat'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    # figname="Afig_"+name
    # save_fig(figname)

    
    
    result,figname=glset_GSV(dds,name)
    dd.report_dict[dd.report(name,3,"_*","_*")]=result
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    result.to_excel(output_dir+name+".xlsx") 
    
    ############################################
    name="NAT sales GSV$"
    
    print("\n",name)
    shop_df=sales_df[(sales_df['glset']=="NAT")]
    dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()

    # fig=dds.tail(dds.shape[0]-731)[['date','mat']].plot(x='date',y=['mat'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    # figname="Afig_"+name
    # save_fig(figname)

    
    
    result,figname=glset_GSV(dds,name)
    dd.report_dict[dd.report(name,3,"_*","_*")]=result
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    result.to_excel(output_dir+name+".xlsx") 
    
    ############################################
    name="WW (010) GSV sales $"
    #print(sales_df)
    print("\n",name)
    shop_df=sales_df[(sales_df['specialpricecat']==10)]
    dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()

    
    # fig=dds.tail(dds.shape[0]-731)[['date','mat']].plot(x='date',y=['mat'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    # figname="Afig_"+name
    # save_fig(figname)

    result,figname=glset_GSV(dds,name)
    dd.report_dict[dd.report(name,3,"_*","_*")]=result
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    result.to_excel(output_dir+name+".xlsx") 
    
    ############################################
    name="Coles (012) GSV sales $"
    
    print("\n",name)
    shop_df=sales_df[(sales_df['specialpricecat']==12)]
    dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()
  
    # fig=dds.tail(dds.shape[0]-731)[['date','mat']].plot(x='date',y=['mat'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    # figname="Afig_"+name
    # save_fig(figname)

    
    result,figname=glset_GSV(dds,name)
    dd.report_dict[dd.report(name,3,"_*","_*")]=result
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    result.to_excel(output_dir+name+".xlsx") 
    
    ############################################
    name="DFS GSV sales $"
    
    print("\n",name)
    shop_df=sales_df[(sales_df['glset']=="DFS")]
    dds=shop_df.groupby(['period'])['salesval'].sum().to_frame() 
    dds['mat']=dds['salesval'].rolling(365,axis=0).sum()
    dds['dates']=dds.index.tolist()
  
    # fig=dds.tail(dds.shape[0]-731)[['date','mat']].plot(x='date',y=['mat'],grid=True,title=title)   #),'BB total scanned vs purchased Coles jam units per week')
    # figname="Afig_"+name
    # save_fig(figname)

    
    
    result,figname=glset_GSV(dds,name)
    dd.report_dict[dd.report(name,3,"_*","_*")]=result
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
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
    
    saved_sales_df=sales_df.copy(deep=True)
    
    sales_df.replace({'productgroup':dd.productgroup_dict},inplace=True)
    sales_df.replace({'productgroup':dd.productgroups_dict},inplace=True)
    sales_df.replace({'specialpricecat':dd.spc_dict},inplace=True)


########################################################3


    pivot_df=pd.pivot_table(sales_df, values='qty', index=['productgroup','product'],columns=['year','month'], aggfunc=np.sum, margins=False,dropna=True,observed=True)

    name="pivot_table_units"
    pivot_df.to_excel(output_dir+name+".xlsx") 
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    
    pivot_df=pd.pivot_table(sales_df, values='salesval', index=['productgroup','product'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
 #   pivot_df.replace({'productgroup':dd.productgroup_dict},inplace=True)
 #   pivot_df.replace({'productgroup':dd.productgroups_dict},inplace=True)

    name="pivot_table_dollars"
    pivot_df.to_excel(output_dir+name+".xlsx") 
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    pivot_df=pd.pivot_table(sales_df[sales_df['productgroup']<"40"], values='qty', index=['productgroup'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
    #pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
    name="pivot_table_units_product_group"
    figname=plot_stacked_line_pivot(pivot_df,name,False)   #,number=6)
    
    
    pivot_df.to_excel(output_dir+name+".xlsx") 
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    
    
    pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset','specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)

    name="pivot_table_customers_x_glset_x_spc"
    pivot_df.to_excel(output_dir+name+".xlsx")
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    pivot_df=pd.pivot_table(sales_df, values='salesval', index=['glset'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
    #pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
    
    #print(pivot_df)  
    name="pivot_table_customers_x_glset"
    pivot_df.to_excel(output_dir+name+".xlsx")
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
    #pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
 #   pivot_df.replace({'specialpricecat':dd.spc_dict},inplace=True)

    name="Dollar sales per month by spc"
    figname=plot_stacked_line_pivot(pivot_df,name,False)   #,number=6)
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    pivot_df.to_excel(output_dir+name+".xlsx")
    
    
    #print(pivot_df) 
    pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
  #  pivot_df.replace({'specialpricecat':dd.spc_dict},inplace=True)
   
    name="pivot_table_customers_spc_nocodes"
    pivot_df.to_excel(output_dir+name+".xlsx")
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,8,"_*","_*")]=figname
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    pivot_df=pd.pivot_table(sales_df, values='salesval', index=['specialpricecat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
   # pivot_df.replace({'specialpricecat':dd.spc_dict},inplace=True)

    #pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
    #ax=plot_stacked_line_pivot(pivot_df,"Unit sales per month by productgroup",False)   #,number=6)
    
    #print(pivot_df) 
    name="pivot_table_customers_x_spc"
    pivot_df.to_excel(output_dir+name+".xlsx") 
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    pivot_df=pd.pivot_table(sales_df, values='salesval', index=['cat','code'],columns=['year','month'], aggfunc=np.sum, margins=True,dropna=True,observed=True)
    #pivot_df.sort_index(axis='columns', ascending=False, level=['month','year'])
    #print(pivot_df) 
    name="pivot_table_customers"
    pivot_df.to_excel(output_dir+name+".xlsx") 
    dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    #report_dict[report(name,5,"*","*")]=name+".xlsx"
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    
    ##################################################################3
    # update reports and save them as pickles for later
    
    if dd.dash_verbose:
        print("\nUpdate and save the qty reports from the coles_and_ww_pkl_dict\n")
    #print("dict keys=\n",dd.coles_and_ww_pkl_dict.keys())
    sales_df=saved_sales_df
    
    
    for key in dd.coles_and_ww_pkl_dict.keys():
        brand=dd.coles_and_ww_pkl_dict[key][0]
        spc=dd.coles_and_ww_pkl_dict[key][1]
        pg=str(dd.coles_and_ww_pkl_dict[key][2])
        pc=dd.coles_and_ww_pkl_dict[key][3]
        if (pc=="jams") | (pc=="_*") | (pc=="_t") | (pc=="_T"):
        #    print("pc=",pc,pg,spc)
            v=sales_df.query('specialpricecat==@spc & productgroup==@pg')[['date','qty']]
        else: 
            v=sales_df.query('specialpricecat==@spc & product==@pc')[['date','qty']]
     #   v.index = pd.to_datetime('date')  #, format='%Y')
  #      v.set_index('date')
      #  v=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        if dd.dash_verbose:
            print("saving",key)   #,"v=\v",v)  #,"v=\n",v)  #,"=\n",v)      
        #print(v)
        with open(key,"wb") as f:
              pickle.dump(v, f,protocol=-1)
    
    
    
    ##############################################################33
    # rank top customers and products
    #
    
    
    latest_date=sales_df['date'].max()
    
    end_date=sales_df['date'].iloc[-1]- pd.Timedelta(2, unit='d')
    #print(end_date)
    #print("ysdf=",sales_df)
    year_sales_df=sales_df[sales_df['date']>end_date]
    #print("ysdf=",year_sales_df)
    year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["code"]).transform(sum)
    pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
    #print("pv=",pivot_df)
    unique_code_pivot_df=pivot_df.drop_duplicates('code',keep='first')
    #unique_code_pivot_df=pd.unique(pivot_df['code'])
    name="Top 20 customers by $purchases in the last 2 days"
    print("\n",name)
    print(unique_code_pivot_df[['code','total_dollars']].head(20))
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    unique_code_pivot_df[['code','total_dollars']].head(20).to_excel(output_dir+name+".xlsx") 
    
   
    
    
    
    
    
    latest_date=sales_df['date'].max()
    
    end_date=sales_df['date'].iloc[-1]- pd.Timedelta(7, unit='d')
    #print(end_date)
    #print("ysdf=",sales_df)
    year_sales_df=sales_df[sales_df['date']>end_date]
    #print("ysdf=",year_sales_df)
    year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["code"]).transform(sum)
    pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
    #print("pv=",pivot_df)
    unique_code_pivot_df=pivot_df.drop_duplicates('code',keep='first')
    #unique_code_pivot_df=pd.unique(pivot_df['code'])
    name="Top 30 customers by $purchases in the last 7 days"
    print("\n",name)
    print(unique_code_pivot_df[['code','total_dollars']].head(30))
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    unique_code_pivot_df[['code','total_dollars']].head(30).to_excel(output_dir+name+".xlsx") 
    
    
    
    
    
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
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    unique_code_pivot_df[['code','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
    
    
    year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["specialpricecat"]).transform(sum)
    pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
    #print("pv=",pivot_df)
    name="Top 50 customers special price category by $purchases in the last 30 days"
    unique_code_pivot_df=pivot_df.drop_duplicates('specialpricecat',keep='first')
    unique_code_pivot_df.replace({'specialpricecat':dd.spc_dict},inplace=True)
    
    #unique_code_pivot_df=pd.unique(pivot_df['code'])
    print("\n",name)
    print(unique_code_pivot_df[['specialpricecat','total_dollars']].head(50))
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","-*")]=output_dir+name+".xlsx"
    
    unique_code_pivot_df[['specialpricecat','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
    
    
    year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["salesrep"]).transform(sum)
    pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
    #print("pv=",pivot_df)
    name="Top salesreps by $sales in the last 30 days"
    unique_code_pivot_df=pivot_df.drop_duplicates('salesrep',keep='first')
    unique_code_pivot_df.replace({'salesrep':dd.salesrep_dict},inplace=True)
    
    #unique_code_pivot_df=pd.unique(pivot_df['code'])
    print("\n",name)
    print(unique_code_pivot_df[['salesrep','total_dollars']].head(50))
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","-*")]=output_dir+name+".xlsx"
    
    unique_code_pivot_df[['salesrep','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
    
    latest_date=sales_df['date'].max()
    
    end_date=sales_df['date'].iloc[-1]- pd.Timedelta(365, unit='d')
    year_sales_df=sales_df[sales_df['date']>end_date]

    
    year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["salesrep"]).transform(sum)
    pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
    #print("pv=",pivot_df)
    name="Top salesreps by $sales in the last 365 days"
    unique_code_pivot_df=pivot_df.drop_duplicates('salesrep',keep='first')
    unique_code_pivot_df.replace({'salesrep':dd.salesrep_dict},inplace=True)
    
    #unique_code_pivot_df=pd.unique(pivot_df['code'])
    print("\n",name)
    print(unique_code_pivot_df[['salesrep','total_dollars']].head(50))
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","-*")]=output_dir+name+".xlsx"
    
    unique_code_pivot_df[['salesrep','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
    
    
    
    latest_date=sales_df['date'].max()
    
    end_date=sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
    year_sales_df=sales_df[sales_df['date']>end_date]

    
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
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    unique_code_pivot_df[['product','total_units','total_dollars']].head(50).to_excel(output_dir+name+".xlsx") 
    
    
    
    
    year_sales_df["total_dollars"]=year_sales_df['salesval'].groupby(year_sales_df["productgroup"]).transform(sum)
    year_sales_df["total_units"]=year_sales_df['qty'].groupby(year_sales_df["productgroup"]).transform(sum)
    pivot_df=year_sales_df.sort_values(by=["total_dollars"],ascending=[False])
    unique_pg_pivot_df=pivot_df.drop_duplicates('productgroup',keep='first')
    unique_pg_pivot_df.replace({'productgroup':dd.productgroup_dict},inplace=True)
    unique_pg_pivot_df.replace({'productgroup':dd.productgroups_dict},inplace=True)
    
    name="Top productgroups by $sales in the last 30 days"
    print("\n",name)
    print(unique_pg_pivot_df[['productgroup','total_units','total_dollars']].head(20))
    #pivot_df.to_excel("pivot_table_customers_ranking.xlsx") 
    dd.report_dict[dd.report(name,3,"_*","_*")]=unique_code_pivot_df
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
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
    dd.report_dict[dd.report(name,3,"_*","_*")]=credit_df
    dd.report_dict[dd.report(name,5,"_*","_*")]=output_dir+name+".xlsx"
    
    credit_df[['date','code','glset','qty','salesval']].tail(50).to_excel(output_dir+name+".xlsx") 
    
    
    
    sales_df=saved_sales_df
   
    
    
    #################################################################################################
    # Create distribution report and find all the good performing and poor performing outliers in retail sales
    if answer3=="y":

      #  print("\nChecking sales trends by customers and products of past year.....")
        
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
        startend_date=sales_df['date'].iloc[-1]- pd.Timedelta(721, unit='d')

        #print(end_date)
        year_sales_df=sales_df[sales_df['date']>end_date]
     #   last_year_sales_df=sales_df[sales_df['date']>startend_date & sales_df['date']<=end_date]
 
        #print("ysdf1=",year_sales_df)
        year_sales_df=year_sales_df[year_sales_df['productgroup'].isin(dd.product_groups_only) & year_sales_df['specialpricecat'].isin(dd.spc_only)]   
      #  last_year_sales_df=last_year_sales_df[last_year_sales_df['productgroup'].isin(dd.product_groups_only) & last_year_sales_df['specialpricecat'].isin(dd.spc_only)]   
 
        #print("ysdf2=",year_sales_df[['date','code','product']])
          
        #cust_list=year_sales_df.code.unique()
        #cust_list = cust_list[cust_list != 'OFFINV']
        #cust_licust_list.remove('OFFINV')
        #cust_list.sort()
        #prod_list=year_sales_df[['product','productgroup']].sort_values(by=['productgroup'])   #.unique()
        
        end_date=sales_df['date'].iloc[-1]- pd.Timedelta(90, unit='d')
        #print(end_date)
        ninetyday_sales_df=sales_df[sales_df['date']>end_date]
        #print("ysdf1=",year_sales_df)
        ninetyday_sales_df=ninetyday_sales_df[ninetyday_sales_df['productgroup'].isin(dd.product_groups_only) & ninetyday_sales_df['specialpricecat'].isin(dd.spc_only)]   
        
        #prod_list=list(set([tuple(r) for r in year_sales_df[['productgroup', 'product']].sort_values(by=['productgroup','product'],ascending=[True,True]).to_numpy()]))
        prod_list=list(set([tuple(r) for r in ninetyday_sales_df[['productgroup', 'product']].to_numpy()]))
        cust_list=list(set([tuple(r) for r in ninetyday_sales_df[['salesrep','specialpricecat', 'code']].to_numpy()]))
        #cust_list = cust_list[cust_list != (88.0,'OFFINV')]
        #print("cust_list=\n",len(cust_list))
        cust_list=[c for c in cust_list if c[2]!="OFFINV"]
            #     #r=[k for k, v in brand_dict.items() if v in product_list]  
        
        #print("\nnew cust_list=",cust_list,len(cust_list))
        
        
        #print("prod_list=\n",prod_list)
        #print("cust_list=\n",cust_list)
        #prod_list.sort()
        #print("prod_list=",prod_list)
        #print("c=",cust_list,len(cust_list))
        #print("p=",prod_list,len(prod_list))
        
        
        #spc_text=dd.spc_only.replace(dd.spc_dict,inplace=True)
       # spc_text=[]
        spc_text=[dd.spc_dict.get(int(e),'') for e in dd.spc_only]
        pg_text=[dd.productgroups_dict.get(int(e),'') for e in dd.product_groups_only]
        
        if dd.dash_verbose:
            print("\nCreating distribution report and sales trends graphs for special price categories:",spc_text,"\nin product groups:",pg_text,"....\n")
            print("unique customers=",len(cust_list))
            print("unique products=",len(prod_list))

        print("\n")
        
        cust_dict={k: v for v, k in enumerate(cust_list)}
        prod_dict={k: v for v, k in enumerate(prod_list)}
        #print("cist dict=\n",cust_dict)
       # print("prod dict=\n",prod_dict)
        dist_df=pd.DataFrame.from_dict(cust_dict,orient='index',dtype=object)  
        distdollars_df=pd.DataFrame.from_dict(cust_dict,orient='index',dtype=np.int32)  
  
        for p in prod_dict.keys():
        #    print pd.to_datetime(dict(year=df.Y, month=df.M, day=df.D))
            dist_df[p]= df.apply(lambda row : pd.to_datetime(dict(year=[2000],month=[1],day=[1])), axis=1)
            distdollars_df[p]=df.apply(lambda row : 0, axis=1)
        #    dist_df[p]=0 #pd.to_datetime({'year': 2000,'month':1,'day':1})   #0  #False #np.nan  #False#,columns=prod_list)
        
        dist_df.drop(0,inplace=True,axis=1)
        dist_df=dist_df.T
        dist_df.index=pd.MultiIndex.from_tuples(dist_df.index,sortorder=0,names=['productgroup','product'])
        dist_df.sort_index(level=0,ascending=True,inplace=True)
        dist_df=dist_df.T
        dist_df.index=pd.MultiIndex.from_tuples(dist_df.index,sortorder=0,names=['salesrep','specialpricecat','code'])
        
        dist_df.sort_index(level=0,ascending=True,inplace=True)
  
        distdollars_df.drop(0,inplace=True,axis=1)
        distdollars_df=distdollars_df.T
        distdollars_df.index=pd.MultiIndex.from_tuples(distdollars_df.index,sortorder=0,names=['productgroup','product'])
        distdollars_df.sort_index(level=0,ascending=True,inplace=True)
        distdollars_df=distdollars_df.T
        distdollars_df.index=pd.MultiIndex.from_tuples(distdollars_df.index,sortorder=0,names=['salesrep','specialpricecat','code'])
        
        distdollars_df.sort_index(level=0,ascending=True,inplace=True)
      
  
    
        #df.index = pd.MultiIndex.from_tuples(df.index,names=["brand","specialpricecat","productgroup","product","on_promo","names"])
        #print("df level (0)=\n",df.index.get_level_values(0))
        
        #print("dist_df before=\n",dist_df,"\n",dd.salesrep_dict)
        
        year_sales_df['counter']=0
        new_sales_df=year_sales_df.copy(deep=True)
        new_sales_df=new_sales_df.iloc[0:0]
 
    #    last_year_new_sales_df=last_year_year_sales_df.copy(deep=True)
    #    last_year_new_sales_df=last_year_new_sales_df.iloc[0:0]
    
 
    
        newninety_sales_df=ninetyday_sales_df.copy(deep=True)
        newninety_sales_df=newninety_sales_df.iloc[0:0]
        
        #print(new_sales_df)
        
        #figure_list=[]
        #dist_df=pd.DataFrame(cust_dict)
        #print("dist df ",dist_df)
        t=0
        total=len(cust_list)*len(prod_list)
        if dd.dash_verbose:
            print("total combinations=",total,"\n")
        
        #    product_list=find_active_products(new_sales_df,age=90)  # 90 days
        for cust in cust_list:
        
            for prod in prod_list:
                r=ninetyday_sales_df[(ninetyday_sales_df['code']==cust[2]) & (ninetyday_sales_df['product']==prod[1]) & (ninetyday_sales_df['salesval']>0.0) & (ninetyday_sales_df['qty']>0.0)].copy(deep=True)
              #  r=r.astype(np.datetime64)
                #   s['counter']=s.shape[0]
     
                s=year_sales_df[(year_sales_df['code']==cust[2]) & (year_sales_df['product']==prod[1]) & (year_sales_df['salesval']>0.0) & (year_sales_df['qty']>0.0)].copy(deep=True)
                dollars=s['salesval'].sum()
            #    print("4:",dollars)
                s['counter']=s.shape[0]
                if r.shape[0]>0:
              #      dist_df.loc[cust,prod]=r['date'].dt.strftime('%d/%m/%Y').max()      #pd.to_datetime({'year': 2020,'month': 1,'day': 1})  #  r.shape[0] #s.date.max()
                    dist_df.loc[cust,prod]=pd.to_datetime(r['date'].max(),utc=False).floor('d')  #,round(dollars,0)]
                    distdollars_df.loc[cust,prod]=np.round(dollars,0)
                  #  dist_df.loc[dollar_cust,prod]=round(dollars,0)   #pd.to_datetime({'year': 2020,'month': 1,'day': 1})  #  r.shape[0] #s.date.max()

                    #dist_df.loc[cust,prod]=r['date'].dt.strftime('%d/%m/%Y').max()      #pd.to_datetime({'year': 2020,'month': 1,'day': 1})  #  r.shape[0] #s.date.max()

                #     print("r['date']=",r['date'],"\n",r['date'].max())
                 #   print("no distribution=\n",cust,"->", prod)  #s[['code','product']])
                s=s.sort_values('date',ascending=False)
              #  s.index=s.date
                t+=1
                if t%10==0:
                    if dd.dash_verbose:                 
                        print("\r",cust,prod,"+",s.shape[0],"=",new_sales_df.shape[0],int(round(t/total*100,0)),"%               ",end='\r',flush=True)                    
                    else:    
                        print("\rDistribution report progress:",int(round(t/total*100,0)),"%               ",end='\r',flush=True)
        
                if s.shape[0]>7: 
                    s['slope'],figname,name=calculate_first_derivative(s,cust[2],prod[1],latest_date)  
                   # s['figure']=figure
                  #  figure_list.append(figure)
                    new_sales_df=new_sales_df.append(s)
                    if (figname!="") & (name!=""):
                        dd.report_dict[dd.report(name,8,cust[2],prod[1])]=figname
        if dd.dash_verbose: 
            print("\n\n")
        #print("distribution matrix =\n",dist_df)
        dist_df=dist_df.rename(dd.salesrep_dict,level='salesrep',axis='index')
        dist_df=dist_df.rename(dd.spc_dict,level='specialpricecat',axis='index')

        dist_df=dist_df.rename(dd.productgroup_dict,level='productgroup',axis='columns')
        dist_df=dist_df.rename(dd.productgroups_dict,level='productgroup',axis='columns')
    
   
    # add totals to distdollars_df
        xlen=distdollars_df.columns.nlevels+distdollars_df.shape[1]
        ylen=distdollars_df.index.nlevels+distdollars_df.shape[0]
     #   print("\n\ndistdollars size=",xlen,ylen)
        
     #   print("cust total=\n",distdollars_df.sum(axis=1))
     #   print("prod total=\n",distdollars_df.sum(axis=0))
        distdollars_df[("999",'total')]=distdollars_df.sum(axis=1)   # prod
     #   print("1\n",distdollars_df)
        distdollars_df=distdollars_df.T
        distdollars_df[("999",999,'total')]=distdollars_df.sum(axis=1)   # cust
        distdollars_df=distdollars_df.T
      #  print("2\n",distdollars_df)
 
        distdollars_df=distdollars_df.iloc[np.lexsort((distdollars_df.index, distdollars_df[("999",'total')]))]
        distdollars_df=distdollars_df.iloc[::-1].T
        distdollars_df=distdollars_df.iloc[np.lexsort((distdollars_df.index, distdollars_df[("999",999,'total')]))]
        distdollars_df=distdollars_df.iloc[::-1].T

     #   distdollars_df.sort_index(axis='index',kind = 'mergesort').sort_values(by=[("999",'total')],axis='index',ascending=False,inplace=True)
     #   print("4\n",distdollars_df)
       
        distdollars_df=distdollars_df.rename(dd.salesrep_dict,level='salesrep',axis='index')
        distdollars_df=distdollars_df.rename(dd.spc_dict,level='specialpricecat',axis='index')

        distdollars_df=distdollars_df.rename(dd.productgroup_dict,level='productgroup',axis='columns')
        distdollars_df=distdollars_df.rename(dd.productgroups_dict,level='productgroup',axis='columns')
 
    
    
      #  print(dist_df,"\n",dist_df.T)
#  list_data=pd.date_range(start='1/1/2018', end='1/08/2018').to_list()
# Create a Pandas dataframe from the data.
#df = pd.DataFrame(list_data)

# Create a Pandas Excel writer using XlsxWriter as the engine.
#excel_file = 'testfile.xlsx'
        sheet_name = 'Sheet1'

        writer = pd.ExcelWriter(output_dir+"distribution_report.xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
        writer2 = pd.ExcelWriter(output_dir+"distribution_report_with_dollars.xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')

#df.to_excel(writer, sheet_name=sheet_name)
        dist_df.to_excel(writer, sheet_name=sheet_name)
        distdollars_df.to_excel(writer2, sheet_name=sheet_name)

# Access the XlsxWriter workbook and worksheet objects from the dataframe.
# This is equivalent to the following using XlsxWriter on its own:
#
#    workbook = xlsxwriter.Workbook('filename.xlsx')
#    worksheet = workbook.add_worksheet()
#
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

            # Apply a conditional format to the cell range.
   #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
        worksheet.conditional_format('D4:ZZ1000', {'type': '3_color_scale'})

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()      
    
        workbook2 = writer2.book
        worksheet2 = writer2.sheets[sheet_name]
        money_fmt = workbook2.add_format({'num_format': '$#,##0', 'bold': False})
        total_fmt = workbook2.add_format({'num_format': '$#,##0', 'bold': True})

        worksheet2.set_column('E:ZZ', 12, money_fmt)
        worksheet2.set_column('D:D', 12, total_fmt)
        worksheet2.set_row(3, 12, total_fmt)

            # Apply a conditional format to the cell range.
   #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
        worksheet2.conditional_format('E5:ZZ1000', {'type': '3_color_scale'})

        # Close the Pandas Excel writer and output the Excel file.
        writer2.save()      
   
    

        #print("\nysdf3=",new_sales_df[['date','code','product','counter','slope']],new_sales_df.shape)
        new_sales_df.drop_duplicates(['code','product'],keep='first',inplace=True)
        #new_sales_df=new_sales_df[new_sales_df['slope']>0.02]
        new_sales_df.sort_values(['slope'],ascending=[False],inplace=True)
        name="growth rankings"
        if dd.dash_verbose:
            print("\nbest growth=",new_sales_df[['code','product','slope']].head(100).to_string())
            print("\nworst growth=",new_sales_df[['code','product','slope']].tail(50).to_string())
            print(new_sales_df.shape)
        dd.report_dict[dd.report(name,3,"_*","_*")]=new_sales_df
        new_sales_df[['code','product','slope']].to_excel(output_dir+name+".xlsx",merge_cells=False,freeze_panes=(2,2),engine='xlsxwriter') 
        
        
        #print("\n\nreport dict=\n",report_dict.keys())
        if dd.dash_verbose:
            print("reports being pickled and saved to",dd.report_savename)
        with open(dd.report_savename,"wb") as f:
            pickle.dump(dd.report_dict, f,protocol=-1)
          
        #plt.pause(0.001) 
        #plt.show()
        plt.close()
        
        print("\n")
        ptotrun=len(prod_list)
        ctotrun=len(cust_list)
        latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
        t_count=1
        for prod in prod_list:
            print("\rProduct unit sales graphs:",t_count,"/",ptotrun,end="\r",flush=True)
            prod_sales=sales_df[sales_df['product']==prod[1]].copy()
           # prod_sales['period2'] = pd.to_datetime('date')  #, format='%Y-%m-%d',exact=False)
         #   print("ps1=",prod_sales)
            prod_sales.set_index('date',inplace=True)
  
          #  print("ps2=",prod_sales)
 
           #print("ps3=",prod_sales)
 
            prod_sales=prod_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
        
            prod_sales['mat']=prod_sales['qty'].rolling(dd.mat,axis=0).mean()
 
          #  print("ps4=",prod_sales)
 
            # styles1 = ['b-','g:','r:']
            styles1 = ['b-']
           # styles1 = ['bs-','ro:','y^-']
            linewidths = 1  # [2, 1, 4]
            fig, ax = pyplot.subplots()
        
            #    fig = plt.figure()
                #ax1 = fig.add_subplot(111)
            ax2 = ax.twiny()


         #   print("prod sales=\m",prod_sales)
            last_years_prod_sales=prod_sales.iloc[:-52]
            prod_sales=prod_sales.iloc[-53:]
            
            
            
           # prod_sales['period']=prod_sales.index
            ax=prod_sales[['mat']].plot(grid=True,title=prod[1]+" units/week moving total "+str(dd.mat)+" weeks @w/c:"+str(latest_date),style=styles1, lw=linewidths)
            last_years_prod_sales[['mat']].plot(grid=False,style=styles1, lw=linewidths,ax=ax2)

            ax.legend(title="")
            ax.set_xlabel("",fontsize=8)

            save_fig(prod[1]+"_units_moving_total")
            
            graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['product']==prod[1]],str(prod[1])+" units per week","Units/week")
                

            
            
            t_count+=1
            
        print("\n")    
        t_count=1
        for cust in cust_list:
            print("\rCustomer dollar sales graphs:",t_count,"/",ctotrun,end="\r",flush=True)
            cust_sales=sales_df[sales_df['code']==cust[2]].copy()
            cust_sales.set_index('date',inplace=True)
            
            cust_sales=cust_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)

            cust_sales['mat']=cust_sales['salesval'].rolling(dd.mat2,axis=0).mean()
            #cust_sales.index = pd.to_datetime('period', format='%d-%m-%Y',exact=False)
 
            # styles1 = ['b-','g:','r:']
            styles1 = ['r-']
           # styles1 = ['bs-','ro:','y^-']
            linewidths = 1  # [2, 1, 4]
    
            latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
            ax=cust_sales[['mat']].plot(grid=True,title=cust[2]+" dollars/week moving total "+str(dd.mat2)+" weeks @w/c:"+str(latest_date),style=styles1, lw=linewidths)
            ax.legend(title="")
            ax.set_xlabel("",fontsize=8)


            save_fig(cust[2]+"_dollars_moving_total")
            
            
            graph_sales_year_on_year(yearly_sales_df[yearly_sales_df['code']==cust[2]],str(cust[2])+" $ sales per week","$/week")
                
            
            t_count+=1
    
        print("\n")
        plt.close("all")
  
    
    
    #############################
    # load scan data from excel reports into a df and add multiindexes for graphing options
    if dd.dash_verbose:
        print("\nLoad prediction and brand index scan data....",dd.scandatalist,"\n")
       
    np.random.seed(42)
    tf.random.set_seed(42)
            
    df=pd.read_excel(dd.scandatalist[0],-1,skiprows=[0,1,2],dtype=object).T.reset_index(drop=True)  #,index_col=None)   #n,index_col=None)  #.T.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
    #print("start df=\n",df)
    for n in range(1,len(dd.scandatalist)):
        df2=pd.read_excel(dd.scandatalist[n],-1,skiprows=[0,1,2],dtype=object,index_col=None).T.reset_index(drop=True)   #,index_col=None)  #.T   #.reset_index()  #,header=[0,1,2])  #,skip_rows=0)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
    
        df2.drop(df2.index[0],inplace=True)
        df=pd.concat((df,df2),axis=0)
    
    df.reset_index(drop=True,inplace=True)
    
 #   print("df1=\n",df,"\n",df.T)
    df = df.rename({0:"scan_week"})
    df=df.T
    #print("df2=\n",df,"\n",df.T)
    df = df.dropna(subset=['scan_week'])
    df.fillna(0.0,inplace=True)
  #  print("df3=\n",df,"\n",df.T)
    df = df.astype(dd.coles_and_ww_convert_dict) 
    df['scan_week']=pd.to_datetime(df['scan_week'],format="%d/%m/%Y",exact=False)   #,yearfirst=True)
    
    df = df.rename(dd.coles_and_ww_col_dict,axis='columns')
    #print("after rename2=\n",df.index)
    df.drop_duplicates(keep='first', inplace=True)
    # delete weeks with all zeros
    #print("df before delete=\n",df)  #.to_string())  #.tail(10),"\n",df.T,"\n",df.columns,df.T.columns)
    #df = df[(df.iloc[:,2:] != 0.0).any()]
    df=df[df.sum(axis=1)!=0]
    #print("empty1 df=\n",df.iloc[:,1:])
    #df2=df.iloc[:,1:]
    #print("df2=\n",df2)
    #print("df2.sum=\n",df2.sum(axis=1))
    
    #print("nonempty df=\n",df,"\n",df.T)  #df[df.sum(axis=1)==0])
    #print("empty3 df=\n",df[(df.iloc[:,2:]!=0).any()])
    
    #print("df after delete=\n",df)  #to_string())  #tail(10),"\n",df.T,"\n",df.columns,df.T.columns)
    
    
    
    df.reset_index(drop=True,inplace=True)
    
    df=df.T
    #print("df5=\n",df)
    df.index = pd.MultiIndex.from_tuples(df.index,names=["brand","specialpricecat","productgroup","product","type","names"])
    #print("df level (0)=\n",df.index.get_level_values(0))
    
    df=df.T
    
    #print("final loaded df=\n",df)
    df.drop_duplicates(keep='first', inplace=True)
    df=df.set_index(list(df.columns[[0]]))   #.dt.strftime('%d/%m/%Y')
    df.index.name = 'scan_week'
    df.index = pd.to_datetime(df.index, format = '%d/%m/%Y',infer_datetime_format=True)
    df=df.sort_index()
    df=df.astype(np.float32)  #,inplace=True)
    
    #print("after6=\n",df)
    #print(df.columns)
    df.replace(np.nan, 0.0,inplace=True)
    df=df*1000
    #test=get_xs_name(df,3,0)
  #  print("final loaded df=\n",df,"\n",df.T)
  #  print("ddcpkl=\n",dd.coles_and_ww_pkl_dict.keys())
#    print("cpkl=\n",coles_and_ww_pkl_dict.keys())
    
    ##################################################################################
     
    before_joined_list=df.columns.to_list()
    
    
           #joined_list=joined_df.index.to_list()
      #  print("joined list=\n",joined_list)
    before_joined_list_df=pd.DataFrame(before_joined_list,columns=df.columns.names)
    #    print("jldf=\n",joined_list_df)
    before_joined_list_df = before_joined_list_df[(before_joined_list_df['type'] == 0) | (before_joined_list_df['type'] == 1)]
    before_joined_list_df = before_joined_list_df[before_joined_list_df['brand'] != 0].iloc[:,:]
        
    before_joined_list_df.drop_duplicates(keep='first',inplace=True)
    #records = list(before_joined_list_df.iloc[:,:4].to_records(index=False))
       # df.drop(df.columns[0], axis=1)
 #   print("\nbefore joined list df=\n",before_joined_list_df.to_string())
 #   print("records=",records,len(records))
    unique_records = list(before_joined_list_df.iloc[:,:4].drop_duplicates(keep='first').to_records(index=False))

   # records2=list(set(records))
   # print("unique records=",unique_records,len(unique_records))
   
    ############################################
    # total all other on_promo and off_promo on matching productcodes (level 3)
    # and level 4 is either 0 or 1
    # get a list of products
    #products=list(set(list(df.columns.get_level_values(3))))
    #print("products=",products)
 #   print("df.columns=\n",df.columns.to_list()) 
    df=df.T
 #   print("before df=\n",df)   

#    pred_count=0 
    for rec in unique_records:
   #     print("rec=",rec)
       # cc=list(df.loc[rec].name)
        cc=list(rec)
        cc2=list(rec)
        cc3=list(rec)
        cc4=list(rec)
      #  print("starting cc=",cc)

        
        brand=dd.brand_dict[cc[0]]
        cust=dd.spc_dict[cc[1]]
          #customer=cc[1]
         #p=cc[3]    # product name
        cc.append(2)  # cc[4]   # type is total                 
        cc.append(str(cust)+"_"+str(brand)+"_"+str(cc[3])+"_total_scanned")   # cc[5]= new name
        cc=tuple(cc)
        
        
        

     #   cc2=list(rec)
      #  print("starting cc=",cc)

        
       # brand=dd.brand_dict[cc2[0]]
       # cust=dd.spc_dict[cc2[1]]
          #customer=cc[1]
         #p=cc[3]    # product name
        cc2.append(6)  # cc[4]   # type is moving total                 
        cc2.append(str(cust)+"_"+str(brand)+"_"+str(cc2[3])+"_4wk_moving_total_scanned")   # cc[5]= new name
        cc2=tuple(cc2)

        # cc3.append(4)  # cc[4]   # type is total                 
        # cc3.append(str(cust)+"_"+str(brand)+"_"+str(cc3[3])+"_invoiced_shifted_3wks")   # cc[5]= new name
        # cc3=tuple(cc3)

        # cc4.append(7)  # cc[4]   # type is moving total                 
        # cc4.append(str(cust)+"_"+str(brand)+"_"+str(cc4[3])+"_4wk_moving_total_invoiced_shifted_3wks")   # cc[5]= new name
        # cc4=tuple(cc4)


        
        
      #  print("finishing cc=",cc)
    #     df[cc]=df.T.xs(cc[5],level=5).sum()
        testslice=df.xs(rec,level=[0,1,2,3],axis=0,drop_level=False)
        
    
      #  print("rec=",rec,"row count=",row_count,"test=\n",testslice)
        df=df.T
     #   print("cc=",cc,"cc2=",cc2)  #"before df=\n",df)
        df[cc]=testslice.sum(axis=0)
     #   print("cc2 mat=",cc2,df[cc].rolling(4,axis=0).mean())   # 4weeks

        df[cc2]=df[cc].rolling(dd.mat,axis=0).mean()   # 4weeks
        
     #   df[cc4]=df[cc3].rolling(dd.mat,axis=0).mean()   # 4weeks
          
    
      #  print("df[cc]=",cc,"\n",df[cc])
        df=df.T

    df=df.sort_index()
   # print("after df=\n",df,"\n",df.T)
 
    df=df.T
 
#     df[1,12,10,"jams",2,'coles_beerenberg_jams_total_scanned']=df[1,12,10,"jams",0,'coles_beerenberg_jams_off_promo_scanned']+df[1,12,10,"jams",1,'coles_beerenberg_jams_on_promo_scanned']
#     df[2,12,10,"jams",2,'coles_st_dalfour_jams_total_scanned']=df[2,12,10,"jams",0,'coles_st_dalfour_jams_off_promo_scanned']+df[2,12,10,"jams",1,'coles_st_dalfour_jams_on_promo_scanned']
#     df[3,12,10,"jams",2,'coles_bonne_maman_jams_total_scanned']=df[3,12,10,"jams",0,'coles_bonne_maman_jams_off_promo_scanned']+df[3,12,10,"jams",1,'coles_bonne_maman_jams_on_promo_scanned']
    
 
#     df[1,12,10,"jams",9,'coles_beerenberg_jams_on_promo']=(df[1,12,10,"jams",1,'coles_beerenberg_jams_on_promo_scanned']>0)
#     df[2,12,10,"jams",9,'coles_st_dalfour_jams_on_promo']=(df[2,12,10,"jams",1,'coles_st_dalfour_jams_on_promo_scanned']>0)
#     df[3,12,10,"jams",9,'coles_bonne_maman_jams_on_promo']=(df[3,12,10,"jams",1,'coles_bonne_maman_jams_on_promo_scanned']>0)
    
#     # df[0,10,10,"jams",0,'woolworths_jams_total_scanned']=df[0,10,10,"jams",0,'woolworths_total_jam_curd_marm_off_promo_scanned']+df[0,10,10,"jams",1,'woolworths_total_jam_curd_marm_on_promo_scanned']
    
#     df[1,10,10,"jams",2,'woolworths_beerenberg_jams_total_scanned']=df[1,10,10,"jams",0,'woolworths_beerenberg_jams_off_promo_scanned']+df[1,10,10,"jams",1,'woolworths_beerenberg_jams_on_promo_scanned']
#     df[2,10,10,"jams",2,'woolworths_st_dalfour_jams_total_scanned']=df[2,10,10,"jams",0,'woolworths_st_dalfour_jams_off_promo_scanned']+df[2,10,10,"jams",1,'woolworths_st_dalfour_jams_on_promo_scanned']
#     df[3,10,10,"jams",2,'woolworths_bonne_maman_jams_total_scanned']=df[3,10,10,"jams",0,'woolworths_bonne_maman_jams_off_promo_scanned']+df[3,10,10,"jams",1,'woolworths_bonne_maman_jams_on_promo_scanned']
     
#     df[1,10,10,"jams",9,'woolworths_beerenberg_jams_on_promo']=(df[1,10,10,"jams",1,'woolworths_beerenberg_jams_on_promo_scanned']>0)
#     df[2,10,10,"jams",9,'woolworths_st_dalfour_jams_on_promo']=(df[2,10,10,"jams",1,'woolworths_st_dalfour_jams_on_promo_scanned']>0)
#     df[3,10,10,"jams",9,'woolworths_bonne_maman_jams_on_promo']=(df[3,10,10,"jams",1,'woolworths_bonne_maman_jams_on_promo_scanned']>0)
    
    
 




# ##########################################################################################33
#    # brand index graphs

    
#     # coles beerenberg jams 1
#     plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo'],savename="coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # coles beerenberg jams 2
#   #  plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_st_dalfour_jams_on_promo','coles_bonne_maman_jams_on_promo'],savename="coles2")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # coles beerenberg jams 3
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_beerenberg_jams_on_promo','coles_st_dalfour_jams_on_promo'],savename="coles3")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # # coles beerenberg jams 4
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_st_dalfour_jams_on_promo','coles_beerenberg_jams_on_promo'],savename="coles4")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     #  # coles beerenberg jams 5
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_beerenberg_jams_on_promo','coles_bonne_maman_jams_on_promo'],savename="coles5")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # # coles beerenberg jams 6
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_bonne_maman_jams_on_promo','coles_beerenberg_jams_on_promo'],savename="coles6")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # # coles beerenberg jams 7   
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_st_dalfour_jams_off_promo_scanned',col_and_hue=['coles_bonne_maman_jams_on_promo','coles_beerenberg_jams_on_promo'],savename="coles7")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # # coles beerenberg jams 8
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_st_dalfour_jams_off_promo_scanned',col_and_hue=['coles_beerenberg_jams_on_promo','coles_bonne_maman_jams_on_promo'],savename="coles8")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # # coles beerenberg jams 9
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_bonne_maman_jams_off_promo_scanned',col_and_hue=['coles_st_dalfour_jams_on_promo','coles_beerenberg_jams_on_promo'],savename="coles9")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
#     # # coles beerenberg jams 10
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_bonne_maman_jams_off_promo_scanned',col_and_hue=['coles_beerenberg_jams_on_promo','coles_st_dalfour_jams_on_promo'],savename="coles10")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
 
   
#     # woolworths beerenberg jams 1
#     plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_beerenberg_jams_off_promo_scanned',col_and_hue=['woolworths_bonne_maman_jams_on_promo','woolworths_st_dalfour_jams_on_promo'],savename="woolworths1")
#     # woolworths beerenberg jams 2
#   #  plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_beerenberg_jams_off_promo_scanned',col_and_hue=['woolworths_st_dalfour_jams_on_promo','woolworths_bonne_maman_jams_on_promo'],savename="woolworths2")   
#     # woolworths beerenberg jams 3
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_beerenberg_jams_off_promo_scanned',col_and_hue=['woolworths_beerenberg_jams_on_promo','woolworths_st_dalfour_jams_on_promo'],savename="woolworths3")
#     # # woolworths beerenberg jams 4
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_beerenberg_jams_off_promo_scanned',col_and_hue=['woolworths_st_dalfour_jams_on_promo','woolworths_beerenberg_jams_on_promo'],savename="woolworths4")   
#     # # woolworths beerenberg jams 5
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_beerenberg_jams_off_promo_scanned',col_and_hue=['woolworths_bonne_maman_jams_on_promo','woolworths_beerenberg_jams_on_promo'],savename="woolworths5")
#     # # woolworths beerenberg jams 6
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_beerenberg_jams_off_promo_scanned',col_and_hue=['woolworths_beerenberg_jams_on_promo','woolworths_bonne_maman_jams_on_promo'],savename="woolworths6")   
#     # # woolworths beerenberg jams 7
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_st_dalfour_jams_off_promo_scanned',col_and_hue=['woolworths_bonne_maman_jams_on_promo','woolworths_beerenberg_jams_on_promo'],savename="woolworths7")   
#     # # woolworths beerenberg jams 8
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_st_dalfour_jams_off_promo_scanned',col_and_hue=['woolworths_beerenberg_jams_on_promo','woolworths_bonne_maman_jams_on_promo'],savename="woolworths8")
#     # # woolworths beerenberg jams 9
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_bonne_maman_jams_off_promo_scanned',col_and_hue=['woolworths_st_dalfour_jams_on_promo','woolworths_beerenberg_jams_on_promo'],savename="woolworths9")   
#     # # woolworths beerenberg jams 10
#     # plot_brand_index(get_xs_name(df,("jams",3)).iloc[:],y_col='woolworths_bonne_maman_jams_off_promo_scanned',col_and_hue=['woolworths_beerenberg_jams_on_promo','woolworths_st_dalfour_jams_on_promo'],savename="woolworths10")
    
   
 
########################################################################################################

# predictions - join invoiced sales data to scan data
    
    
    if answer2=="y":
    
        
        ####################################
        # coles_pkl_dict which is save in a dictionary of report_dict as a pickle
        # coles_pkl_dict contains a list of files names as keys to run as the actual sales in the prediction vs actual df
        #
        
        with open(dd.report_savename,"rb") as f:
            report_dict=pickle.load(f)
        
        #print("report dict=",report_dict.keys())
       # coles_and_ww_pkl_dict=report_dict[dd.report('coles_and_ww_pkl_dict',0,"","")]
    #    print("dd.coles_and_ww_pkl dict=",dd.coles_and_ww_pkl_dict)
        
        ###########################################3
        
        joined_df=df.copy(deep=True)
        if dd.dash_verbose:
            print("resampling by week.")
       # joined_df=joined_df.resample('W-THU', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
        joined_df=joined_df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)

      #  print("joined_df=\n",joined_df)
        for key in dd.coles_and_ww_pkl_dict.keys():
           # savepkl="scanned_sales_plus_"+key
         #   print("Loading query dataframe:",key)
            with open(key,"rb") as f:
                actual_sales=pickle.load(f)
         #   print("key=",key,"coles_pkl_dict]key]=",pkl_dict[key],"\n",actual_sales)    
            actual_sales.reset_index(drop=True,inplace=True)  
            actual_sales.index=actual_sales.date
            actual_sales=actual_sales[['qty']]
          #  print(actual_sales)
       #     forecast_df = actual_sales.resample('W-SAT', label='left', loffset=pd.DateOffset(days=1)).sum().round(0)
            forecast_df = actual_sales.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)

        #    print(key,"fdf=\n",forecast_df)   #,"pdk=",dd.coles_and_ww_pkl_dict[key])
            joined_df=pd.concat([joined_df,forecast_df],axis=1)   #.sort_index(axis=1)
        #    joined_df=joined_df.rename(columns={"qty":key.rsplit(".", 1)[0]})
            joined_df=joined_df.rename(columns={"qty":dd.coles_and_ww_pkl_dict[key]})
         
            shifted_key=list(dd.coles_and_ww_pkl_dict[key])
          #  print("key=",key,shifted_key)
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
    #    print("joined_df.columns=\n",joined_df.columns.to_list())   #get_level_values(5).to_list())    
        #print("joined_df.index=\n",joined_df.index.to_list())   #get_level_values(5).to_list())    


        joined_df=joined_df.T
      #  print("\njoined_df before=\n",joined_df)

        
        #joined_df.index=joined_df.index.rename(columns={4:dd.series_type_dict})

        
       # print("joined_df.index=\n",list(joined_df.index))   #.get_level_values(5).to_list())    
 
  
        
        
        joined_df.index = pd.MultiIndex.from_tuples(joined_df.index,names=["brand","specialpricecat","productgroup","product","type","names"])
        joined_df=joined_df.rename(dd.series_type_dict,level=4,axis='index')

        joined_df=joined_df.sort_index()    #[('brand', 'specialpricecat','productgroup','product')], ascending=[True,True,True,True])
        
      #  print("\njoined_df after=\n",joined_df)
        
        
        joined_list=joined_df.index.to_list()
      #  print("joined list=\n",joined_list)
        joined_list_df=pd.DataFrame(joined_list,columns=joined_df.index.names)
    #    print("jldf=\n",joined_list_df)
        joined_list_df = joined_list_df[joined_list_df['brand'] != 0].iloc[:,:5]
        
        joined_list_df.drop_duplicates(keep='first',inplace=True)
        
       # df.drop(df.columns[0], axis=1)
       # print("\njoined list df=\n",joined_list_df.to_string())

        

        joined_df=joined_df.T
        #print("joined_df keys=\n",joined_df.keys())
        
        #print("joined df=\n",joined_df,"\n",joined_df.T)
        
        #products=list(set(list(joined_df.columns.get_level_values(3))))
        retailers=list(set(list(joined_df.columns.get_level_values(1))))
        
        #print("retailers=",retailers)
        
        
        #graph_list=[]
       
        #joined_df=joined_df.T
        
    #    joined_df['lastdate'] = pd.to_datetime(joined_df.index,format="%Y-%m-%d",exact=False)
        latest_date = pd.to_datetime(joined_df.index,format="%Y-%m-%d",exact=False).max()
      
     #   latest_date = joined_df['lastdate'].max()
        

        
      #  print("joined df=\n",joined_df)
      #  print("joined df.T=\n",joined_df.T)
        
        joined_df.to_excel("joinedsave.xlsx",engine='xlsxwriter')
        joined_df.T.to_excel("joinedsaveT.xlsx",engine='xlsxwriter')
        
        
        
        
      #  scan_sort=joined_df.T.droplevel(level=2,axis=0)
        #print("scan_sort=\n",scan_sort)
        
        #scan_sort=scan_sort.droplevel(level=1,axis=0)
        #scan_sort=scan_sort.droplevel(level=2,axis=0)
      #  scan_sort=scan_sort.droplevel(level=3,axis=0)
       # print("start scan sort=\n",scan_sort)  #,"\n",scan_sort.T)

#############################################################################################################33


        hdf=joined_df.copy(deep=True)
        if hdf.columns.nlevels>=2:
            for _ in range(hdf.columns.nlevels-1):
                hdf=hdf.droplevel(level=0,axis=1)
        
        ##############################################################################
       # print("hdf=",hdf)
       # print("joined_df columns=\n",list(joined_df.columns[5]))
        
        #hdf=get_xs_name2(joined_df,"",5)
        #print("hdf=\n",hdf.columns)
        #df=hdf[['coles_beerenberg_jams_total_scanned','coles_beerenberg_jams_invoiced','coles_beerenberg_jams_invoiced_shifted_3wks']].rolling(dd.mat,axis=0).mean()
        df=hdf[['coles_beerenberg_jams_total_scanned','coles_beerenberg_jams_invoiced_shifted_3wks']].rolling(dd.mat,axis=0).mean()
       # styles1 = ['b-','g:','r:']
        styles1 = ['b-','r:']
               # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
        
        latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
        ax=df.plot(grid=True,title="Coles BB units moving total "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
        ax.legend(title="")
        save_fig("Coles_BB_jams_units_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")
        
       

        df=hdf[['coles_beerenberg_jams_total_scanned','woolworths_beerenberg_jams_total_scanned']].rolling(dd.mat,axis=0).mean()
       # styles1 = ['b-','g:','r:']
        styles1 = ['r-','g-']
               # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
        
        latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
        ax=df.plot(grid=True,title="Coles vs WW BB units moving total "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
        ax.legend(title="")
        save_fig("Coles_vs_WW_BB_jams_units_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")

        ax=df.plot(grid=True,title="Coles vs WW BB units moving total stacked "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths,stacked=True)
        ax.legend(title="")
        save_fig("Coles_vs_WW_BB_jams_units_stacked_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")









        df=hdf[['coles_st_dalfour_jams_total_scanned','woolworths_st_dalfour_jams_total_scanned']].rolling(dd.mat,axis=0).mean()
       # styles1 = ['b-','g:','r:']
        styles1 = ['r-','g-']
               # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
        
        latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
        ax=df.plot(grid=True,title="Coles vs WW SD units moving total "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
        ax.legend(title="")
        save_fig("Coles_vs_WW_SD_jams_units_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")


        ax=df.plot(grid=True,title="Coles vs WW SD units moving total stacked "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths,stacked=True)
        ax.legend(title="")
        save_fig("Coles_vs_WW_SD_jams_units_stacked_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")



        df=hdf[['coles_bonne_maman_jams_total_scanned','woolworths_bonne_maman_jams_total_scanned']].rolling(dd.mat,axis=0).mean()
       # styles1 = ['b-','g:','r:']
        styles1 = ['r-','g-']
               # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
        
        latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
        ax=df.plot(grid=True,title="Coles vs WW BM units moving total "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
        ax.legend(title="")
        save_fig("Coles_vs_WW_BM_jams_units_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")

        ax=df.plot(grid=True,title="Coles vs WW BM units moving total stacked "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths,stacked=True)
        ax.legend(title="")
        save_fig("Coles_vs_WW_BM_jams_units_stacked_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")




        df=hdf[['woolworths_beerenberg_jams_total_scanned','woolworths_st_dalfour_jams_total_scanned','woolworths_bonne_maman_jams_total_scanned']].rolling(dd.mat,axis=0).mean()
       # styles1 = ['b-','g:','r:']
        styles1 = ['b-','r-','g-']
               # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
        
        latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
        ax=df.plot(grid=True,title="WW premium jams units moving total "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
        ax.legend(title="")
        save_fig("WW_premium_jams_units_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")

        ax=df.plot(grid=True,title="WW premium jams units moving total stacked "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths, stacked=True)
        ax.legend(title="")
        save_fig("WW_premium_jams_units_stacked_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")



        df=hdf[['coles_beerenberg_jams_total_scanned','coles_st_dalfour_jams_total_scanned','coles_bonne_maman_jams_total_scanned']].rolling(dd.mat,axis=0).mean()
       # styles1 = ['b-','g:','r:']
        styles1 = ['b-','r-','g-']
               # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
        
        latest_date=pd.to_datetime(latest_date).strftime("%d/%m/%Y")
        ax=df.plot(grid=True,title="Coles premium jams units moving total "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
        ax.legend(title="")
        save_fig("coles_premium_jams_units_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")


        ax=df.plot(grid=True,title="Coles premium jams units moving total stacked "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths,stacked=True)
        ax.legend(title="")
        save_fig("coles_premium_jams_units_stacked_moving_total")
        #plt.show()
        #print(df)
        plt.close()   #"all")




        
        #hdf=get_xs_name2(joined_df,"",5)
        #print("hdf=\n",hdf.columns)
#        df=hdf[['woolworths_beerenberg_jams_total_scanned','woolworths_beerenberg_jams_invoiced','woolworths_beerenberg_jams_invoiced_shifted_3wks']].rolling(dd.mat,axis=0).mean()
        df=hdf[['woolworths_beerenberg_jams_total_scanned','woolworths_beerenberg_jams_invoiced_shifted_3wks']].rolling(dd.mat,axis=0).mean()

        #styles1 = ['b-','g:','r:']
        styles1 = ['b-','r:']
 
               # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
        
          
        ax=df.plot(grid=True,title="ww BB units moving total "+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths)
        ax.legend(title="")
        save_fig("Ww_BB_jams_units_moving_total")
        #plt.show()
        #print(df)
        plt.close()
        
        ax=df.plot(grid=True,title="ww BB units moving total stacked"+str(dd.mat)+" weeks w/c:"+str(latest_date),style=styles1, lw=linewidths,stacked=True)
        ax.legend(title="")
        save_fig("Ww_BB_jams_units_stacked_moving_total")
        #plt.show()
        #print(df)
        plt.close("all")
   
        
        ############################################################33
        # load previous runs coles_predictions
        
        # previous_df=pd.read_pickle("order_predict_results.pkl")
        # pred_cols = [col for col in previous_df.columns if 'prediction' in col]
        # previous_df=previous_df[pred_cols]
        # previous_df.columns=previous_df.columns+"_old"
        
        #############################################################
             # 
            # 
            # no of weeks
        #target_offset=3
        
        
        
        
        
        
        
        
        ###########################################################################################################
        # Predict
        
                
        joined_df=joined_df.T
        joined_df=joined_df.sort_index()
        joined_df.fillna(0.0,inplace=True)
        joined_df=add_a_week(joined_df)
        # joined_list=joined_df.columns.to_list()
 
        joined_index_df=pd.DataFrame(joined_list,columns=joined_df.index.names)
        #    print("jldf=\n",joined_list_df)
       # joined_index_df = joined_index_df[(joined_index_df['on_promo'] == 'baseline') | (joined_index_df['on_promo'] == 1)]
        joined_index_df = joined_index_df[joined_index_df['brand'] != 0]     #.iloc[:,:]
            
        joined_index_df.drop_duplicates(keep='first',inplace=True)
        #records = list(before_joined_list_df.iloc[:,:4].to_records(index=False))
           # df.drop(df.columns[0], axis=1)
     #   print("\nbefore joined list df=\n",before_joined_list_df.to_string())
     #   print("records=",records,len(records))
        unique_records = list(joined_index_df.iloc[:,:4].drop_duplicates(keep='first').to_records(index=False))
    
       # records2=list(set(records))
        print("Products to predict=",len(unique_records))
        if dd.dash_verbose:
            print("\n",unique_records)
     #   print("\n",dd.coles_and_ww_pkl_dict)
      #  print("\n")
        
        c_count=1
        for rec in unique_records:
            cc2=list(rec)
            cc2.append(4)
            cc2=tuple(cc2)
            
            brand=dd.brand_dict[cc2[0]]
            cust=dd.spc_dict[cc2[1]]
   
            cc3=list(rec) 
           # cc4=list(rec)
            cc3.append(7)  # cc[4]   # type is total                 
            cc3.append(str(cust)+"_"+str(brand)+"_"+str(cc3[3])+"_4wk_moving_total_invoiced_shifted_3wks")   # cc[5]= new name
            cc3=tuple(cc3)
    
            testslice=joined_df.xs(cc2,level=[0,1,2,3,4],axis=0,drop_level=False)
            slicesum=testslice.sum(axis=0)
            joined_df=joined_df.T
            joined_df[cc3]=slicesum.rolling(dd.mat,axis=0).mean()[:-1]   # 4weeks
            joined_df=joined_df.T
    
            xrec=list(rec)
            xrec.append(2)
            yrec=list(rec)
            yrec.append(4)
            invrec=list(rec)
            invrec.append(3)
            predrec=list(rec)
          #  predrec.append(invrec[3])
            predrec.append(3)
          #  print("predrec=",predrec)
          
        #    print("predrec1=",predrec)
            try:
                pname=str(joined_df.xs(tuple(predrec),level=[0,1,2,3,4],axis=0,drop_level=False).index[0][5])    #.get_level_values('names')   #.astype(str, copy = False)   #to_string()   #.name    #.get_level_values(5)
                pname=pname+"_prediction"
            except:
                pname="No_data_"+str(predrec)
         #   print("pname",pname)
            predrec[4]=8
            predrec.append(pname)  
            
            predrec=tuple(predrec)
       #     print("new predrec=",predrec)
 
        #    matrec=list(rec)
        #    matrec.append(6)
 
 
            total_rec=tuple(xrec)
            inv_shift_rec=tuple(yrec)

            mdf=joined_df.xs(rec,level=[0,1,2,3],axis=0,drop_level=False)
            X_hdf=joined_df.xs(xrec,level=[0,1,2,3,4],axis=0,drop_level=False)    #.T.to_numpy().astype(np.int32)
            y_hdf=joined_df.xs(yrec,level=[0,1,2,3,4],axis=0,drop_level=False)     #.T.to_numpy().astype(np.int32)
        #    inv_hdf=joined_df.xs(invrec,level=[0,1,2,3,4],axis=0,drop_level=False).T.to_numpy().astype(np.int32)
        #    mat_hdf=joined_df.xs(matrec,level=[0,1,2,3,4],axis=0,drop_level=False).T.to_numpy().astype(np.int32)
            y_hdf.fillna(0.0,inplace=True)
            X_hdf.fillna(0.0,inplace=True)

            if dd.dash_verbose:
                print("\nproduct slice to predict and plot:'",pname,"'-> (",c_count,"/",len(unique_records),") =\n")
        #    if dd.dash_verbose:
                print(mdf)

            try:
                X_set_full=X_hdf.T.to_numpy().astype(np.int32)[:,0]
                X_set=X_hdf.T.to_numpy().astype(np.int32)[5:-3,0]    # iloc[:,2] type 2 is total
                y_set_full=y_hdf.T.to_numpy().astype(np.int32)[:,0]
                y_set=y_hdf.T.to_numpy().astype(np.int32)[6:-2,0]  #[7:-1]    # iloc[:,3] type 2 is total
            except:
                if dd.dash_verbose:
                    print("X_set or y_set empty")
            else:    
                if np.all((y_set==0)):
                    if dd.dash_verbose:
                        print("No invoiced sales",str(rec))
                else:    
               
                #    dates=pd.to_datetime(joined_df.columns[7:-1]).strftime("%d-%m-%Y").tolist()
                    # add a new week to joined_df
                #    joined_df=add_a_week(joined_df)
                    
                    
               #     print("dates=\n",dates,len(dates))
                   #    print("\n\n",ptx+btx+p,mdf.T,X_set.shape,y_set.shape)
                    model=train_model(clean_up_name(str(rec)),X_set,y_set,dd.batch_length,dd.no_of_batches,dd.epochs,c_count)
                 #   if c_count==0:
               #     pmdf=add_a_week(mdf) 
                    joined_df=predict_order(joined_df,X_set_full,y_set_full,predrec,model)
                  #  else:    
                   #     joined_df=pd.concat((joined_df,predict_order(joined_df,X_set,y_set_full,predrec,model)),axis=0)
                #    c_count+=1   
                    mdf=joined_df.xs(rec,level=[0,1,2,3],axis=0,drop_level=False)
                    mdf_mask = mdf.index.get_level_values('type').isin([2,4,8])
               #     print("mdf_mask=\n",mdf_mask)
     
                    plot_mdf=mdf[mdf_mask]
                    if dd.dash_verbose:
                        print("product slice predicted=\n",pname,"\n",plot_mdf,"\n")

              #      print("plotmdf=\n",plot_mdf)
                    plot_mdf=plot_mdf.droplevel(['brand','specialpricecat','productgroup','product'])
     #               plot_mdf=plot_mdf.droplevel([0,1,2,3])
     
                    title=str(plot_mdf.index[0][1])
                  # plot_mdf[m].reset_index(level=2, drop=True)
                   # plot_mdf=mdf.xs((2,4,8),level='type',axis=0,drop_level=False)   # scanned total, shifted invoiced and predicted type 2,4,8
                #    print("plot mdf,title=\n",plot_mdf,"\n",title)
    
                    plot_prediction(plot_mdf,title,latest_date)
                    #print("Y_pred_series=",Y_pred_series)
            c_count+=1    






###########################################################################



        joined_df.fillna(0.0,inplace=True)
        p_mask = joined_df.index.get_level_values('type').isin([8])
        p_df=joined_df[p_mask].copy()
        p_df.index=p_df.index.droplevel(['brand','specialpricecat','productgroup','product','type'])
        
        print("\nColes and WW order Predictions...")
        
        print(p_df.iloc[:,-2:],"\n")
        m_df=p_df.iloc[:,-2:].copy()
        
        p_df=p_df.T
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        p_df.to_pickle("dash_run_"+now+dd.predictions_only_savename)
        
        # keep only types 2 - total, 4 - invoiced and 8 - predicted
        j_mask = joined_df.index.get_level_values('type').isin([2,4,8])
               #     print("mdf_mask=\n",mdf_mask)
     
        joined_df=joined_df[j_mask]
        
        
        # rename the columns for the spreadsheet writer
        joined_df=joined_df.rename(dd.productgroups_dict,level='productgroup',axis='index')
        joined_df=joined_df.rename(dd.brand_dict,level='brand',axis='index')
        joined_df=joined_df.rename(dd.spc_dict,level='specialpricecat',axis='index')
        joined_df=joined_df.rename(dd.series_type_dict,level='type',axis='index')


        
        joined_df.sort_index(axis=1,inplace=True)
        joined_df=joined_df.T
        if dd.dash_verbose:
            print("final joined_df=\n",joined_df)

        #     print("results=\n",results.tail(5))
            
        
        
        
        
        joined_df.to_pickle(output_dir+"order_predict_results.pkl")
        #     results.to_pickle("order_predict_results.pkl")
        
        
        sheet_name = 'Sheet1'

        writer = pd.ExcelWriter(output_dir+"order_predict_results.xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
       
        
        joined_df.to_excel(writer,sheet_name=sheet_name)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')
        
        writer.save()    

        writer = pd.ExcelWriter("dash_run_"+now+"_predict_results.xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
       
        
        p_df.to_excel(writer,sheet_name=sheet_name)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')
        
        writer.save()    



        writer = pd.ExcelWriter(output_dir+"mini_order_predict_results.xlsx",engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
       
        m_df.to_excel(writer,sheet_name=sheet_name)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')
        
        writer.save()    


    
    plt.close("all")
    end_timer = time.time()
    print("\nDash total runtime:",round(end_timer - start_timer,2),"seconds.\n")

    return



if __name__ == '__main__':
    main()


