#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 10:18:50 2020

@author: tonedogga
"""

import BB_data_dict as dd
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot, dates
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns

from datetime import datetime
import datetime as dt

import calendar
import xlsxwriter
import xlrd

import os 

import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import timedelta
import calendar
import xlsxwriter

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


#plt.ion() # enables interactive mode

#print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# dd.scan_dict_savename
#with open(dd.scan_dict_savename, 'rb') as g:
#        scan_dict = pickle.load(g)
    
 
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(output_dir, fig_id + "." + fig_extension)
  #  print("Saving figure", fig_id)
    if tight_layout:
        
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
    return
   
 

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./dashboard2_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)



output_dir = log_dir("dashboard2")
os.makedirs(output_dir, exist_ok=True)



   # scan_dict={"original_df":original_df,
   #              "final_df":df,
   #    #          "full_index_df":full_index_df,
   #         #     "market_rename_dict":dd.market_rename_dict,
   #          #   "product_dict":product_dict,
   #              "measure_conversion_dict":dd.measure_conversion_dict,
   #              "stacked_conversion_dict":dd.stacked_conversion_dict,
   #              'plot_type_dict':dd.plot_type_dict,
   #              'brand_dict':dd.brand_dict,
   #              'category_dict':dd.category_dict,
   #              'spc_dict':dd.spc_dict,
   #              'salesrep_dict':dd.salesrep_dict,
   #              'series_type_dict':dd.series_type_dict,
   #              'productgroups_dict':dd.productgroups_dict,
   #              'productgroup_dict':dd.productgroup_dict,
   #              'variety_type_dict':dd.variety_type_dict,
   #              'second_y_axis_conversion_dict':dd.second_y_axis_conversion_dict,
   #              'reverse_conversion_dict':dd.reverse_conversion_dict}




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





# def read_excel(df,filename):
#         sheet_name = 'Sheet1'
#         writer = pd.ExcelWriter(filename,engine='xlrd',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')     
#         df=pd.read_excel(writer,sheet_name=sheet_name,header=False,index=False)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')     
#         writer.save()
#         return df






def load_data(scan_data_files,scan_data_filesT): 
    np.random.seed(42)
  #  tf.random.set_seed(42)
    
    print("\n\nLoad scan data spreadsheets...\n")
         
    
    count=1
    for scan_file,scan_fileT in zip(scan_data_files,scan_data_filesT):
      #  column_count=pd.read_excel(scan_file,-1).shape[1]   #count(axis='columns')
        #if dd.dash_verbose:
        print("Loading...",scan_file,scan_fileT)   #,"->",column_count,"columns")
      
       # convert_dict={col: np.float64 for col in range(1,column_count-1)}   #1619
       # convert_dict['index']=np.datetime64
    
        if count==1:
 #           df=pd.read_excel(scan_file,-1,dtype=convert_dict,index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
            dfT=pd.read_excel(scan_file,-1,header=None,engine='xlrd',dtype=object)    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)

            write_excel(dfT.T,scan_fileT)

            df=pd.read_excel(scan_fileT,-1,header=None,index_col=[1,2,3,4,5,6,7,8,9,10,11],engine='xlrd',dtype=object)  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
        else:
       #     print(convert_dict)
         #   del df2
            dfT=pd.read_excel(scan_file,-1,header=None,engine='xlrd',dtype=object)    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
            
            write_excel(dfT.T,scan_fileT)

     
            df2=pd.read_excel(scan_fileT,-1,header=None,index_col=[1,2,3,4,5,6,7,8,9,10,11],engine='xlrd',dtype=object) #,na_values={"nan":0}) 
        
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
    df.index.set_names('sortorder', level=8,inplace=True)
    df.index.set_names('colname', level=9,inplace=True)
    df.index.set_names('measure', level=10,inplace=True)
   
    
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



# create a query language for a multiindex scan data df
 # a plot is described by a query list
# a plot title name in query[0] 
# plot details are in a dictionary at query[1]
# Seconary y columns is in a list of column numbers in query[1]
# a query list in query[1:]
# (the column names in the query contain the line plotting information in their multiindex for the y values in each column)
#


    
# def plot_query(query,scan_data): 
#    # scan_data=scan_data.T
#   #  print("plot query",scan_data)
#   #  scan_data=set_plot_properties(scan_dict['final_df'])

#     slice_df=slice_filter(query,scan_data)
#     #print("q=\n",title,"\n", q)
#     keep_list=['column_name',"style",'linewidth','stacked','second_y','reverse']
#     plot_df,title,sy,style,left,right,dont_plot_list,reverse=decode_query(slice_df,query,keep_list)
    
#     print(plot_df.T,plot_df)
#     date=pd.to_datetime(plot_df.index).strftime("%Y-%m-%d").to_list()
#     plot_df['date']=date
#     plot_df.sort_values('date',ascending=True,inplace=True)
 
#     plot_df['dates'] = pd.to_datetime(plot_df['date']).apply(lambda date: date.toordinal())
#     #plot_df.sort_values('dates',ascending=True,inplace=True)
    
#     fig, ax = pyplot.subplots()
#     ax2 = ax.twinx()
     
#     for p in range(0,plot_df.shape[1]-2):
#         if p in dont_plot_list:
#             pass
#         else:
#             if p in sy:   
#                     plot_df.iloc[:,np.r_[p, -1]].plot(x='dates',grid=True,xlabel="",fontsize=8,title=title,subplots=True,secondary_y=sy,mark_right=True,style=style[p],ax=ax2)  #,legend=g.index[0][0])
#             else:
#                     plot_df.iloc[:,np.r_[p, -1]].plot(x='dates',grid=True,xlabel="",fontsize=8,title=title,subplots=True,mark_right=True,style=style[p],ax=ax)   #,legend=g.index[0][0])

#     # for p in range(0,plot_df.shape[1]-2):
#     #     if p in dont_plot_list:
#     #         pass
#     #     else:
#     #         if p in sy:   
#     #             if reverse[p]:
#     #                 ax2.invert_yaxis()
#     #         else:
#     #             if reverse[p]:
#     #                 ax.invert_yaxis()



#     if any(reverse):   # & len(sy)>0:
#         ax2.invert_yaxis()

#     ax2.legend(loc=(0.3,0),fontsize=6,title="Right y axis",title_fontsize=7)
#     ax2.set_ylabel(right,fontsize=8)      
#     ax.legend(loc=(0,0.91),title="Left y axis",fontsize=6,title_fontsize=7)
#     ax.set_ylabel(left,fontsize=8)

#     new_labels = [dt.date.fromordinal(int(item)) for item in ax.get_xticks()]
#   #  print("new_labels=",new_labels)
#     improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
#   #  print("improved labels",improved_labels)
#     ax.set_xticklabels(improved_labels,fontsize=8)
 
#     return    
 
    

   
# def slice_filter(query,df):
#  #   print("slice filter query=",query)
#     if isinstance(df.index, pd.MultiIndex):
#         transposed=False
#     else:
#       #  print("not multi index- transposing")
#      #   print("no levels=",df.index.nlevels)   
#         df=df.T
#         transposed=True
        
#     if isinstance(df.index, pd.MultiIndex):
#      #   print("yes multi index T'ed levels=",df.index.nlevels)    
#         index_names=df.index.names
#      #   print(type(df.index.names))   #.type)
#      #   df.index.get_level_values(0).dtype
#         # all(isinstance(n, int) for n in lst)
#          # receive a query is a list of tuples of a list of values to filter and a level name as a query
#         #secondary_y_list=query[1] 
#         for q in query[1:]: 
#             filter_list=q[0]
#             level_name=q[1]
#             if level_name in index_names:  
#               #  print("type=",df.index.get_level_values(level_name).dtype)
#                # print("filter list=",filter_list)
#                 filter_list_type=[type(f) for f in filter_list]
#         #        print("filter list type=",q,">",filter_list_type)
#                 for n in filter_list:
#                     if all(isinstance(n, f) for f in filter_list_type):
#                         pass
#                        # print(n,"types are correct")
#                    #     print("filter list dtype=",filter_list_type)
    
#                 #        print("level name=",level_name)
            
#                     else:
#                        print("types in query are incorrect")
#                        return pd.DataFrame() 
                   
     
#              #  test_df=df.xs(xrec,level=[0,1,2,3,4],axis=0,drop_level=False)    #.T.to_numpy().astype(np.int32)
#             #                     mdf=joined_df.xs(rec,level=[0,1,2,3],axis=0,drop_level=False)
#                     mdf_mask = df.index.get_level_values(level_name).isin(filter_list)
#                        #     print("mdf_mask=\n",mdf_mask)
#                     df=df[mdf_mask]
#             else:
#                 print("level name error",q)
#                 return pd.DataFrame()  
                
                   
                   
#         if transposed:           
#             df=df.T
        
#         return df

#     else:    
#         print("index is not multilevel")
#         return pd.DataFrame()   
    


# def tidy_up(df,keep_list):    
#     level_list=df.columns.names
# #    level_list.sort(inplace=True)
#  #   print(level_list)  
#  #   remove_level_list=[]  
#     remove_level_list=[l for l in level_list if l not in keep_list]
#   #  print(remove_level_list)
#    # for l in remove_level_list:
#     df.columns=df.columns.droplevel(remove_level_list)
#     if len(keep_list)>1:
#         df=df.reorder_levels(keep_list,axis=1)
#     return df,pd.DataFrame.from_records(df.columns,columns=df.columns.names) 




# def decode_query(slice_df,query,keep_list):  
#     query_dict=query[0]
#     title=query_dict['title']
#     gdf,r=tidy_up(slice_df,keep_list)
#     #print("g=\n",g,"\n",g.T)
    
#     #print("r=\n",r,"\n",r['column_name'])
#     #styles1 = r['style'].tolist()  # ['g:','r:','b-']
#                # styles1 = ['bs-','ro:','y^-']
#     #linewidths = max(r['linewidth'].tolist())   #1  # [2, 1, 4]
#     reverse=gdf.columns.get_level_values('reverse').tolist()
#     dont_plot_list=query_dict['remove']
    
#     gdf.columns=gdf.columns.droplevel(list(range(1,len(keep_list))))
# #    sy=[gdf.columns[n] for n in query_dict["second_y"]]
#     sy=[]
#     for n in range(0,gdf.shape[1]): 
#       if n in query_dict["second_y"]:
#           if n in dont_plot_list:
#               sy.append(False)
#           else:
#               sy.append(True)
#       else:
#           sy.append(False)

#     #sy=query_dict['second_y'] 
#     style=query_dict['style']
#     left=query_dict['left']
#     right=query_dict['right']

#    # reverse=query_dict['reverse']
#  #   print("sy=",sy)
#     return gdf,title,sy,style,left,right,dont_plot_list,reverse



# def set_plot_properties(df):
#     df=reverse_rankings(df)
#   #  print(df.T)
#     return df



# def reverse_rankings(df):
#     replace_list=[(m.lower().find("ranked")!=-1) for m in df.columns.get_level_values('measure')]
#   #  print("len(rlist)=",replace_list,len(replace_list))

#     df=df.T
#     df["reverse"]=replace_list
#     df.index=df.index.droplevel('reverse')
#     df=df.set_index('reverse', append=True)
#     df=df.T

#     return df

# def list_of_special_plots(df):
#     stacked=df.index.get_level_values('stacked').tolist().count('1')
#     print("stcked cnt=",stacked)
#     second_yaxis=df.index.get_level_values('yaxis').tolist().count('1')
#     print("secondy axis=",second_yaxis)
#     return []




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
    df=df.droplevel([0,1,2,3,4,5,6,7,8])
    
    df=df.T
    df['date']=pd.to_datetime(df.index).strftime("%Y-%m").to_list()
    newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
    df=df.T
    df.iloc[0:2]*=1000
    #print("plot type1 df=\n",df)
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
 
    df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('Units/week',fontsize=9)

    line=df.iloc[2].T.plot(use_index=False,xlabel="",kind='line',rot=0,style="g-",secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    ax.right_ax.set_ylabel('$ price',fontsize=9)
    fig.legend(title="Units/week vs $ price",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.4, 1.1))
  #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
    new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
    improved_labels = ['{}\n{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
    
 #   print("improived labels=",improved_labels[0])
    improved_labels=improved_labels[:1]+improved_labels[::week_freq]
    
    
  
    ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
    ax.set_xticklabels(improved_labels,fontsize=6)

    return




def plot_type2(df):
    # first column is total units sales
    # second column is distribution 
    
   
      
    week_freq=8
   # print("plot type1 df=\n",df)
    df=df.droplevel([0,1,2,3,4,5,6,7,8])
    
  #  df=df.T
  #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
  #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
  #  df=df.T
    df.iloc[:1]*=1000
    #print("plot type1 df=\n",df)
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
 
 #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('Units/week',fontsize=9)

    line=df.iloc[:1].T.plot(use_index=True,xlabel="",kind='line',style=["r-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

    if df.shape[0]>=2:
        line=df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

   # if df.shape[0]>=3:
   #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
#  ax.set_ylabel('Units/week',fontsize=9)

        ax.right_ax.set_ylabel('Distribution',fontsize=9)
    fig.legend(title="Units/week vs distribution",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.4, 1.1))
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
    df=df.droplevel([0,1,2,3,4,5,6,7,8])
    
  #  df=df.T
  #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
  #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
  #  df=df.T
 #   df.iloc[:1]*=1000
 #   print("plot type3 df=\n",df)
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
 
 #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('$ Price',fontsize=9)

    line=df.T.plot(use_index=True,xlabel="",kind='line',style=["g-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

  #  if df.shape[0]>=2:
   # line=df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

   # if df.shape[0]>=3:
   #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
#  ax.set_ylabel('Units/week',fontsize=9)

 #   ax.right_ax.set_ylabel('Units/week',fontsize=9)
    fig.legend(title="$ Price",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.4, 1.1))
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
    df=df.droplevel([0,1,2,3,4,5,6,7,8])
    
  #  df=df.T
  #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
  #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
  #  df=df.T
    df.iloc[:]*=1000
 #   print("plot type3 df=\n",df)
    fig, ax = pyplot.subplots()
    fig.autofmt_xdate()
 
 #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    ax.set_ylabel('Units/week',fontsize=9)

    line=df.T.plot(use_index=True,xlabel="",kind='line',style=["b-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

  #  if df.shape[0]>=2:
  #  line=df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)

   # if df.shape[0]>=3:
   #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
#  ax.set_ylabel('Units/week',fontsize=9)

  #  ax.right_ax.set_ylabel('Units/week',fontsize=9)
    fig.legend(title="Units/week",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.4, 1.1))
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
    plottypes=list(set(list(set(df.index.get_level_values('plottype').astype(str).tolist()))+list(set(df.index.get_level_values('plottype1').astype(str).tolist()))))
   #     plottypes=list(set([p for p in plottypes if p!='0']))
   #     print("plotypes=",plottypes)
    for pt in plottypes:  
        plotnumbers=list(set(df.index.get_level_values('plotnumber').astype(str).tolist()))
        new_df=pd.concat((multiple_slice_scandata(df,[(pt,'plottype')]) ,multiple_slice_scandata(df,[(pt,'plottype1')])),axis=0)   #,(pt,'plottype1')])

 #   print("plotn",plotnumbers)
        for pn in plotnumbers:
            if (pt=='3') | (pt=='4'):
                plot_df=new_df
            else:
                plot_df=multiple_slice_scandata(new_df,[(pn,'plotnumber')])

    #        print("plot_df=\n",plot_df)
            plot_df.replace(0.0,np.nan,inplace=True)
        
            if str(pt)=='1':   #standard plot type
                plot_type1(plot_df)
            elif str(pt)=='2':   #stacked bars plus right axis price
                plot_type2(plot_df)
            elif str(pt)=='3':   # 
                plot_type3(plot_df)
            elif str(pt)=='4':   #unused 
                plot_type4(plot_df)
            elif str(pt)=='0':
                pass
            save_fig(pn+"_"+pt+"_"+str(randrange(999)))
            plt.show()
            
             
    plt.close('all')
    return
  
    
  
    
 
scandatalist=["coles_scan_data_enhanced_sept2020.xlsx","ww_scan_data_enhanced_sept2020.xlsx"] 
transposed_datalist=["coles_scan_dataT.xlsx","ww_scan_dataT.xlsx"]  
 
df=load_data(scandatalist,transposed_datalist)
#print(df)
#new_df=slice_scandata(df,key='1',criteria='brand')
#print("ss=",new_df)
#new_df=multiple_slice_scandata(df,key=['1'],criteria='brand')
#print("ms-",new_df)
new_df=multiple_slice_scandata(df,query=[('1','brand'),('10','productgroup')]) #   key=['1'],criteria='brand')
#print("ms2",new_df)

#print(new_df.columns,"\n",new_df.index)
      
plot_slices(new_df)
   

new_df=multiple_slice_scandata(df,query=[('10','retailer'),('1','variety')]) #   key=['1'],criteria='brand')
print("ms2",new_df)

#print(new_df.columns,"\n",new_df.index)
      
plot_slices(new_df)
new_df=multiple_slice_scandata(df,query=[('12','retailer'),('1','variety')]) #   key=['1'],criteria='brand')
print("ms3",new_df)

#print(new_df.columns,"\n",new_df.index)
      
plot_slices(new_df)
   
