#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:24:55 2020

@author: tonedogga
"""
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
import matplotlib.ticker as ticker
import seaborn as sns

import matplotlib.dates as mdates

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




# def get_xs_name2(df,f,l):
#     #  returns a slice of the multiindex df with a tuple (column value,index_level) 
#     # col_value itselfcan be a tuple, col_level can be a list
#     # levels are (brand,specialpricecat, productgroup, product,name) 
#     #
#   #  print("get_xs_name df index",df.columns,df.columns.nlevels)
#     if df.columns.nlevels>=2:

#         df=df.xs(f,level=l,drop_level=False,axis=1)
#     #df=df.T
#    #     print("2get_xs_name df index",df.columns,df.columns.nlevels)
#         if df.columns.nlevels>=2:
#             for _ in range(df.columns.nlevels-1):
#                 df=df.droplevel(level=0,axis=1)
    
#     else:
#         print("not a multi index df columns=",df,df.columns)    
#     return df


def remove(value, deletechars):
    for c in deletechars:
        value = value.replace(c,'')
    return value;





def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(output_dir, fig_id + "." + fig_extension)
  #  print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    return



def plot_df_xs(df,plot_types):
    df=df.T
    #gcf().autofmt_xdate() 
#    plot_df=df.loc[:,(category,market,brand,variety,plot_type)]
 #   print("stacked:",list(df.index.get_level_values(0)))
    stacked=all(list(df.index.get_level_values(0)))  
 #   print(":sk=",stacked)
    df.index = df.index.droplevel(0)
    df=df.T
    styles1 = ['b-','g-','r-']
    df['date'] = pd.to_datetime(df.index,format="%d/%m/%y",exact=False)
    df = df.set_index('date', append=True)
  #  print("\n",df)
    
    df.index = df.index.droplevel(0)
    
  #  print(df)
    
  #  print(df.shape)   #,'_'.join(df.columns))  
   # ax.tick_params(pad=12)
    
 #   print("\n",df)
   # styles1 = ['bs-','ro:','y^-']
    linewidths = 1  # [2, 1, 4]
 #   print("\nplot type=",plot_type,"sk-",stacked)  #,"\ndf stacked",list(df.columns.levels[0]))
  #  if any(df['stacked']):
    if stacked:  


#        ax.xaxis.set_major_formatter(formatter)
   #     print("\n",df,df.T)
        
        #ax=plt.gca()
     #   plt.figure()
     #   df.plot(y=df.columns[0:2],secondary_y=df.columns[2],grid=True,title=plot_type+":stacked",kind="bar",rot=0,stacked=True,fontsize=7)   #,style=styles1, lw=linewidths,stacked=True,fontsize=7)

     #   if df.shape[1]==3:
     #       ax=df.plot(secondary_y=df.columns[2],kind="bar",rot=0,stacked=True,fontsize=7)   #,style=styles1, lw=linewidths,stacked=True,fontsize=7)

            #ax=df.plot(secondary_y=df.columns[2],kind="line",rot=0,fontsize=7)   #,style=styles1, lw=linewidths,stacked=True,fontsize=7)
      #              ax.set_ylabel('price',fontsize=7)
                
        # fig, ax1 = plt.subplots()
    #    fig, ax = plt.subplots()
        # ax2 = ax1.twinx()
        # ax1.plot(x, y1, 'g-')
        # ax2.plot(x, y2, 'b-')
        
        # ax1.set_xlabel('X data')
        # ax1.set_ylabel('Y1 data', color='g')
                # ax2.set_ylabel('Y2 data', color='b')
      # print(df)         
       if any(list(df.index.get_level_values(0))) & (df.shape[1]>2):   # second Y axis
        #    df.index = df.index.droplevel(0)
        #    print(df)            
                        
            fig = plt.figure() # Create matplotlib figure
            
             
            ax = fig.add_subplot(111) # Create matplotlib axes
            ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
            
            
#            df.plot(kind='bar', color=['red','blue'], ax=ax,rot=0,legend=False,stacked=True)  #, width=width, position=1)
            ax2=df.iloc[:,2:].plot(kind='line',linestyle=":",ax=ax2,use_index=False,color='blue',legend=False)   #, width=width, position=0)
            ax=df.iloc[:,:2].plot(kind='bar',color=['green','red'],rot=0,ax=ax,legend=False,stacked=True)  #, width=width, position=1)
      #      ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.


            ax.set_ylabel("Sales (000) / week",fontsize=8)
            ax2.set_ylabel('$ Price or Depth of Dist',fontsize=8)
            ax.set_xlabel("",fontsize=7)
            fig.legend(fontsize=7,loc='best')


     
            ticklabels = ['']*len(df.index)
         # Every 4th ticklable shows the month and day
            ticklabels[::8] = [item.strftime('%b') for item in df.index[::8]]
         # Every 12th ticklabel includes the year
            ticklabels[10::26] = [item.strftime('\n\n%Y') for item in df.index[10::26]]
   

 
     # #   plt.ylabel("sales (000)",fontsize=7)
     #  #  plt.xlabel("",fontsize=7)

            plt.gca().xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
     # #   plt.xtick_params(pad=100)
            plt.gcf().autofmt_xdate()



     #    plt.gca().legend(fontsize=6)      
     #    save_fig("scandata_"+plot_type+"_"+remove(str(df.columns[0]),'\/:*?"<>|'))
        
     #   # save_fig("scandata_"+plot_type+"_"+str(df.columns[0][:-10]))
          #  plt.show()   #ax=ax)
        #    plt.close()
       else:
            fig = plt.figure() # Create matplotlib figure
            
             
            ax = fig.add_subplot(111) # Create matplotlib axes
         #   ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
            
            
#            df.plot(kind='bar', color=['red','blue'], ax=ax,rot=0,legend=False,stacked=True)  #, width=width, position=1)
         #   ax2=df.iloc[:,2:].plot(kind='line',linestyle="-",ax=ax2,use_index=False,color='blue',legend=False)   #, width=width, position=0)
            ax=df.iloc[:,:2].plot(kind='bar',color=['green','red'],rot=0,ax=ax,legend=False,stacked=True)  #, width=width, position=1)
      #      ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.


            ax.set_ylabel("Sales (000) / week",fontsize=8)
          #  ax2.set_ylabel('Price',fontsize=8)
            ax.set_xlabel("",fontsize=7)
            fig.legend(fontsize=7,loc='best')


     
            ticklabels = ['']*len(df.index)
         # Every 4th ticklable shows the month and day
            ticklabels[::8] = [item.strftime('%b') for item in df.index[::8]]
         # Every 12th ticklabel includes the year
            ticklabels[10::26] = [item.strftime('\n\n%Y') for item in df.index[10::26]]
   

 
     # #   plt.ylabel("sales (000)",fontsize=7)
     #  #  plt.xlabel("",fontsize=7)

            plt.gca().xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
     # #   plt.xtick_params(pad=100)
            plt.gcf().autofmt_xdate()



     #    plt.gca().legend(fontsize=6)      
     #    save_fig("scandata_"+plot_type+"_"+remove(str(df.columns[0]),'\/:*?"<>|'))
        
     #   # save_fig("scandata_"+plot_type+"_"+str(df.columns[0][:-10]))
        #    plt.show()   #ax=ax)

           
 
    
 
    
 
    
 
    
 
    
      #  plt.show()
    else:
       ax=plt.gca()
   
       ax.xaxis.set_minor_locator(mpl.dates.MonthLocator())
       ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%b'))
#   ax.xtick_params(pad=20)
       ax.xaxis.set_major_locator(mpl.dates.YearLocator())
       ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
   
   
       df.plot(grid=True,title=str(plot_type),style=styles1,ax=ax,lw=linewidths,kind='line',stacked=False,fontsize=7,legend=False)
       ax.set_xlabel("",fontsize=7)
    #   ax.set_ylabel("Sales (000) / week",fontsize=8)
       ax.legend(fontsize=6,loc='best')
   #    plt.show()
     #  plt.figure()
#ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator([1, 7]))
#ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter('%b'))
#
#    ax = plt.gca()

 #   plt.gcf().autofmt_xdate()


#    ax.suptitle('test title', fontsize=20)
    #    plt.ylabel("sales (000)",fontsize=7)
    #    plt.xlabel("",fontsize=7)
    
       # plt.pause(0.001)
        
      #  figname="scan_fig_"+title
      #  save_fig(figname)
  #    print remove(filename, '\/:*?"<>|')

    save_fig("scandata_"+remove(str(plot_type)+"_"+str(df.columns[0]),'\/:*?"<>|'))
    #    plt.show()   #ax=ax)
    plt.close()
    return


#month_day_fmt = mdates.DateFormatter('%b %d') # "Locale's abbreviated month name. + day of the month"
#ax.xaxis.set_major_formatter(month_day_fmt)

    
    
#def main():   
#if True:      

scan_dict_savename="scan_dict.pkl"

output_dir = log_dir("scandata")
os.makedirs(output_dir, exist_ok=True)

    
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format
  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors


print("\n\n\nIRI scan data reporter - By Anthony Paech 21/7/20")
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






with open(scan_dict_savename, 'rb') as g:
    scan_data_dict = pickle.load(g)

#print(b)
#print("\n\n")
latest_date=scan_data_dict['final_df'].index[-1]
print("\nBeerenberg scan data at:",latest_date)
print("+++++++++++++++++++++==================\n")


df=scan_data_dict['final_df']
print(df)

market_dict=scan_data_dict['market_dict']
brand_dict=scan_data_dict['brand_dict']
category_dict=scan_data_dict['category_dict']
plot_type_dict=scan_data_dict['plot_type_dict']
variety_type_dict=scan_data_dict['variety_type_dict']
measure_conversion_dict=scan_data_dict['measure_conversion_dict']
stacked_conversion_dict=scan_data_dict['stacked_conversion_dict']
second_y_axis_conversion_dict=scan_data_dict['second_y_axis_conversion_dict']


# print("market",market_dict)  #=scan_data_dict['market_dict']
# print("brand",brand_dict)   #=scan_data_dict['brand_dict']
# print("cat",category_dict)   #=scan_data_dict['category_dict']
print("plot type",plot_type_dict)   #=scan_data_dict['measure_type_dict']
# print("variety",variety_type_dict)  #=scan_data_dict['variety_type_dict']
# print("measure name",measure_conversion_dict)    #=scan_data_dict['measure_conversion_dict']

#print(df.loc[[:,'measure']])

# report structure
# for each category, by market, by brand 2 graphs
# one units+price
# two dollars + depth of dist
#
print("Index levels=",df.T.index.names)
measures=df.T.index.get_level_values(-1).unique().tolist()
with open("measures_savename.pkl","wb") as f:
    pickle.dump(measures,f,protocol=-1)
print("\nmeasures=",measures,len(measures))
#measures2=list(measures)  
#print("\nmeasures2=",measures2)
  
print("\n")
report_count=1
for category in category_dict.values(): # level 0
 #   print(category)
    for market in market_dict.keys(): # level 1
  #      print(market)
        for brand in brand_dict.keys():  # level2
   #         print(brand)
            for variety in variety_type_dict.keys(): # level 3
    #            print(variety)
                for plot_type in plot_type_dict.keys(): #level 4
                  #  print("\n",plot_type)
                     #  plot_df_xs(df.loc[:,(category,market,brand,variety,plot_type)],plot_type_dict[plot_type],plot_type) 
                #    report_count+=1
                    cutshape=0
                    try:
                        cut=df.loc[:,(category,market,brand,variety,plot_type)]
                        cutshape=cut.shape[0]
                        
                   #     if cut:
                   #         plot_df_xs(cut,plot_type_dict[plot_type]) 

                #       report_count+=1
                    except:
                        pass
                    finally:
                         if cutshape>0:
                        #    print("cut=\n",cut) 
                            plot_df_xs(cut,plot_type_dict[plot_type]) 
                            print("\rReport ("+str(report_count)+") plotting:"+str(plot_type_dict[plot_type])+"_"+remove(str(cut.columns[0]),'\/:*?"<>|')+"                                            \r",end="\r",flush=True)

                        #      print("\r..success                                                              \r",end="\r",flush=True)
                            report_count+=1
                    #     else:
                    #        print("\r..fail. no data                                                        \r",end="\r",flush=True)

                   
                   #.xs(measure_type,level=1,drop_level=False,axis=1))
       #        try:
     #               xs_df=df.xs(category,level=0,drop_level=False,axis=1)
      #              print(category,xs_df)
                  #  except:
                  #      pass
         #print(df[[category,market]])

#print(df.loc[:,(1,1,2)])
plt.close("all")
print("\n")
#df=df.reorder_levels(['category','market','brand','variety','plot_type','stacked','market_name','product','measure'],axis=0).T
df=df.swaplevel(2,3,axis=1)
df=df.swaplevel(1,2,axis=1)
df=df.swaplevel(6,7,axis=1)

#df=df.reorder_levels(['category','variety','market','brand','plot_type','stacked','market_name','product','measure'],axis=0).T
print("latest 53 weeks only, Reordered index levels=",df.T.index.names)

df=df.iloc[-53:,:]

print("\n")

for category in category_dict.values(): # level 0
 #   print(category)
   for variety in variety_type_dict.keys(): # level 3
    #            print(variety)
        for market in market_dict.keys(): # level 1
  #      print(market)
            for brand in brand_dict.keys():  # level2
   #         print(brand)
#plot_type_dict[plot_type]

                for plot_type in plot_type_dict.keys(): #level 4
                  #  print("\n",plot_type)
        #            print("\rReport ("+str(report_count)+") plotting:"+str(plot_type_dict[plot_type])+"_"+remove(str(cut.columns[0]),'\/:*?"<>|')+"                                                                 \r",end="\r",flush=True)                                 #  plot_df_xs(df.loc[:,(category,market,brand,variety,plot_type)],plot_type_dict[plot_type],plot_type) 
                #    report_count+=1
                    cutshape=0
                    try:
                        cut=df.loc[:,(category,market,brand,variety,plot_type)]
                        cutshape=cut.shape[0]
                        
                   #     if cut:
                   #         plot_df_xs(cut,plot_type_dict[plot_type]) 

                #       report_count+=1
                    except:
                        pass
                    finally:
                         if cutshape>0:
                            plot_df_xs(cut,plot_type_dict[plot_type]) 
                            print("\rReport ("+str(report_count)+") plotting:"+str(plot_type_dict[plot_type])+"_"+remove(str(cut.columns[0]),'\/:*?"<>|')+"                                                            \r",end="\r",flush=True)
 
                        #      print("\r..success                                                              \r",end="\r",flush=True)
                            report_count+=1

plt.close("all")
print("\nFinished...\n")
