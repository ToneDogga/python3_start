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

# a plot is described by a query list
# a plot title name in query[0] 
# a query list in query[1:]
# (the column names in the query contain the line plotting information in their multiindex for the y values in each column)
#











   

# create a query language for a multiindex scan data df
    
def slice_filter(query,df):
    print("slice filter query=",query)
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
                       return "",pd.DataFrame(),pd.DataFrame()  
                   
     
             #  test_df=df.xs(xrec,level=[0,1,2,3,4],axis=0,drop_level=False)    #.T.to_numpy().astype(np.int32)
            #                     mdf=joined_df.xs(rec,level=[0,1,2,3],axis=0,drop_level=False)
                    mdf_mask = df.index.get_level_values(level_name).isin(filter_list)
                       #     print("mdf_mask=\n",mdf_mask)
                    df=df[mdf_mask]
            else:
                print("level name error",q)
                return "",pd.DataFrame(),pd.DataFrame()  
                
                   
                   
        if transposed:           
            df=df.T
        
        return query[0],df,pd.DataFrame.from_records(df.columns,columns=df.columns.names) 

    else:    
        print("index is not multilevel")
        return "",pd.DataFrame(),pd.DataFrame()     
    

def tidy_up(df,keep_level_list):
    
    level_list=df.columns.names
 #   print(level_list)    
    remove_level_list=[l for l in level_list if l not in keep_level_list]
  #  print(remove_level_list)
   # for l in remove_level_list:
    df.columns=df.columns.droplevel(remove_level_list)
    df=df.reorder_levels(keep_level_list,axis=1)
    return df


    
 
    

print("All scandata dataframe saved to",dd.scan_dict_savename)   #,":\n",scan_dict['final_df'])
scan_data=scan_dict['final_df']
print("scan_data=\n",scan_data,"\n",scan_data.shape)       

# receive a query is a list of tuples. each tuple contains a list of values to filter and a level name as a query
scan_data=scan_data
#print("scan_data=1\n",scan_data)
query=["tqitle",([2,4],'plot_type'),([1],'variety'),([1],'brand'),([12],"market")]
title,q,r=slice_filter(query,scan_data)
print("q=\n",title,"\n", q,"\n",r.T)

keep_list=['column_name','colour','marker','stacked','second_y','reverse']
g=tidy_up(q,keep_list)
print(g,"\n",g.T)

ax=g.plot(legend_off=True)
ax.legend(title="",fontsize=8)
ax.xticks(fontsize=4)
