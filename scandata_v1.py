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







def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(output_dir, fig_id + "." + fig_extension)
  #  print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    return





    
    
#def main():   
#if True:      


scan_data_files=["jam_scan_data_2020.xlsx","cond_scan_data_2020.xlsx","sauce_scan_data_2020.xlsx"]
total_columns_count=1619+797
scan_dict_savename="scan_dict.pkl"

output_dir = log_dir("scandata")
os.makedirs(output_dir, exist_ok=True)

    
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format
  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors


print("\n\n\nIRI scan data reader - By Anthony Paech 25/5/20")
print("=================================================\n")       

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

 
print("\n==================================================\n")       


   
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


plot_type_dict={0:"Price vs units report",
                   1:"Dollars on promo report",
                   2:"Units vs price report",
                   3:"Units on promo report",
                   4:"Distribution report"}


#Used to convert the meaure column names to a measure type to match the dict above
# the first digit is the plot report to group them in, the second is whether to stack the fields when plotting
measure_conversion_dict={0:4,
                         1:2,
                         2:1,
                         3:1,
                         4:0,
                         5:2,
                         6:0,
                         7:3,
                         8:3}


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
stacked_conversion_dict={0:False,
                         1:False,
                         2:True,
                         3:True,
                         4:False,
                         5:False,
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



