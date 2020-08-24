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
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from datetime import datetime
import datetime as dt

import calendar
import xlsxwriter

 
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
from datetime import timedelta
import calendar
import xlsxwriter

from pathlib import Path,WindowsPath


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
with open(dd.scan_dict_savename, 'rb') as g:
        scan_dict = pickle.load(g)
    
    
 

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












   

# create a query language for a multiindex scan data df
 # a plot is described by a query list
# a plot title name in query[0] 
# plot details are in a dictionary at query[1]
# Seconary y columns is in a list of column numbers in query[1]
# a query list in query[1:]
# (the column names in the query contain the line plotting information in their multiindex for the y values in each column)
#


    
def plot_query(query,scan_data): 
   # scan_data=scan_data.T
  #  print("plot query",scan_data)
  #  scan_data=set_plot_properties(scan_dict['final_df'])

    slice_df=slice_filter(query,scan_data)
    #print("q=\n",title,"\n", q)
    keep_list=['column_name',"style",'linewidth','stacked','second_y','reverse']
    plot_df,title,sy,style,left,right,dont_plot_list,reverse=decode_query(slice_df,query,keep_list)
    
    print(plot_df.T,plot_df)
    date=pd.to_datetime(plot_df.index).strftime("%Y-%m-%d").to_list()
    plot_df['date']=date
    plot_df.sort_values('date',ascending=True,inplace=True)
 
    plot_df['dates'] = pd.to_datetime(plot_df['date']).apply(lambda date: date.toordinal())
    #plot_df.sort_values('dates',ascending=True,inplace=True)
    
    fig, ax = pyplot.subplots()
    ax2 = ax.twinx()
     
    for p in range(0,plot_df.shape[1]-2):
        if p in dont_plot_list:
            pass
        else:
            if p in sy:   
                    plot_df.iloc[:,np.r_[p, -1]].plot(x='dates',grid=True,xlabel="",fontsize=8,title=title,subplots=True,secondary_y=sy,mark_right=True,style=style[p],ax=ax2)  #,legend=g.index[0][0])
            else:
                    plot_df.iloc[:,np.r_[p, -1]].plot(x='dates',grid=True,xlabel="",fontsize=8,title=title,subplots=True,mark_right=True,style=style[p],ax=ax)   #,legend=g.index[0][0])

    # for p in range(0,plot_df.shape[1]-2):
    #     if p in dont_plot_list:
    #         pass
    #     else:
    #         if p in sy:   
    #             if reverse[p]:
    #                 ax2.invert_yaxis()
    #         else:
    #             if reverse[p]:
    #                 ax.invert_yaxis()



    if any(reverse):   # & len(sy)>0:
        ax2.invert_yaxis()

    ax2.legend(loc=(0.3,0),fontsize=6,title="Right y axis",title_fontsize=7)
    ax2.set_ylabel(right,fontsize=8)      
    ax.legend(loc=(0,0.91),title="Left y axis",fontsize=6,title_fontsize=7)
    ax.set_ylabel(left,fontsize=8)

    new_labels = [dt.date.fromordinal(int(item)) for item in ax.get_xticks()]
  #  print("new_labels=",new_labels)
    improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
  #  print("improved labels",improved_labels)
    ax.set_xticklabels(improved_labels,fontsize=8)
 
    return    
 
    

   
def slice_filter(query,df):
 #   print("slice filter query=",query)
    if isinstance(df.index, pd.MultiIndex):
        transposed=False
    else:
      #  print("not multi index- transposing")
     #   print("no levels=",df.index.nlevels)   
        df=df.T
        transposed=True
        
    if isinstance(df.index, pd.MultiIndex):
     #   print("yes multi index T'ed levels=",df.index.nlevels)    
        index_names=df.index.names
     #   print(type(df.index.names))   #.type)
     #   df.index.get_level_values(0).dtype
        # all(isinstance(n, int) for n in lst)
         # receive a query is a list of tuples of a list of values to filter and a level name as a query
        #secondary_y_list=query[1] 
        for q in query[1:]: 
            filter_list=q[0]
            level_name=q[1]
            if level_name in index_names:  
              #  print("type=",df.index.get_level_values(level_name).dtype)
               # print("filter list=",filter_list)
                filter_list_type=[type(f) for f in filter_list]
        #        print("filter list type=",q,">",filter_list_type)
                for n in filter_list:
                    if all(isinstance(n, f) for f in filter_list_type):
                        pass
                       # print(n,"types are correct")
                   #     print("filter list dtype=",filter_list_type)
    
                #        print("level name=",level_name)
            
                    else:
                       print("types in query are incorrect")
                       return pd.DataFrame() 
                   
     
             #  test_df=df.xs(xrec,level=[0,1,2,3,4],axis=0,drop_level=False)    #.T.to_numpy().astype(np.int32)
            #                     mdf=joined_df.xs(rec,level=[0,1,2,3],axis=0,drop_level=False)
                    mdf_mask = df.index.get_level_values(level_name).isin(filter_list)
                       #     print("mdf_mask=\n",mdf_mask)
                    df=df[mdf_mask]
            else:
                print("level name error",q)
                return pd.DataFrame()  
                
                   
                   
        if transposed:           
            df=df.T
        
        return df

    else:    
        print("index is not multilevel")
        return pd.DataFrame()   
    


def tidy_up(df,keep_list):    
    level_list=df.columns.names
#    level_list.sort(inplace=True)
 #   print(level_list)  
 #   remove_level_list=[]  
    remove_level_list=[l for l in level_list if l not in keep_list]
  #  print(remove_level_list)
   # for l in remove_level_list:
    df.columns=df.columns.droplevel(remove_level_list)
    if len(keep_list)>1:
        df=df.reorder_levels(keep_list,axis=1)
    return df,pd.DataFrame.from_records(df.columns,columns=df.columns.names) 




def decode_query(slice_df,query,keep_list):  
    query_dict=query[0]
    title=query_dict['title']
    gdf,r=tidy_up(slice_df,keep_list)
    #print("g=\n",g,"\n",g.T)
    
    #print("r=\n",r,"\n",r['column_name'])
    #styles1 = r['style'].tolist()  # ['g:','r:','b-']
               # styles1 = ['bs-','ro:','y^-']
    #linewidths = max(r['linewidth'].tolist())   #1  # [2, 1, 4]
    reverse=gdf.columns.get_level_values('reverse').tolist()
    dont_plot_list=query_dict['remove']
    
    gdf.columns=gdf.columns.droplevel(list(range(1,len(keep_list))))
#    sy=[gdf.columns[n] for n in query_dict["second_y"]]
    sy=[]
    for n in range(0,gdf.shape[1]): 
      if n in query_dict["second_y"]:
          if n in dont_plot_list:
              sy.append(False)
          else:
              sy.append(True)
      else:
          sy.append(False)

    #sy=query_dict['second_y'] 
    style=query_dict['style']
    left=query_dict['left']
    right=query_dict['right']

   # reverse=query_dict['reverse']
 #   print("sy=",sy)
    return gdf,title,sy,style,left,right,dont_plot_list,reverse



def set_plot_properties(df):
    df=reverse_rankings(df)
  #  print(df.T)
    return df



def reverse_rankings(df):
    replace_list=[(m.lower().find("ranked")!=-1) for m in df.columns.get_level_values('measure')]
  #  print("len(rlist)=",replace_list,len(replace_list))

    df=df.T
    df["reverse"]=replace_list
    df.index=df.index.droplevel('reverse')
    df=df.set_index('reverse', append=True)
    df=df.T

    return df


   
 
    
 
    

#print("All scandata dataframe saved to",dd.scan_dict_savename)   #,":\n",scan_dict['final_df'])
scan_data=set_plot_properties(scan_dict['final_df'])
#print("scan_data=\n",scan_data.T,"\n",scan_data.shape)       

# a query is one dictionary with the details of the plot and then a list of tuples. each tuple contains a list of values to filter and a level name as a query

query=[{'title':"Coles BB jam comparison",'second_y':[1],'style':['g-','b-','r-','k-'],'left':"lefty","right":"righty","remove":[]},([2,4],'plot_type'),([1],'variety'),([1],'brand'),([12],"market")]
plot_query(query,scan_data)
 
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
    

