# IRI analysis by anthony Paech 28/12/19
#
# IRI spreadsheets are read in by IRI_reader_vx-xx.py
# the spreadsheets are turned into dataframes, queryies, split and shaped and joined and then saved in a dataframe dictionary
# which contains the key number which is the spreadhseet loaded (typically a particlar customer)
# the shape of the the dataframe
# and a data field dictionary of the column of each dataframe
#
#  this module loads the 
# promotional activity and brand interaction matrix
##Good
##Expanding market share
##Steal from competitors
##
##Ok
##Channel switch (Coles-> ww)
##Trade up or down
##
##Bad
##Brand position and price point erosion
##Pantry loading


#!/usr/bin/python3
from __future__ import print_function
from __future__ import division


import numpy as np
import pandas as pd
#from pandas.plotting import scatter_matrix

import csv
import sys
import datetime as dt
import joblib
import pickle
import json

from pandas.plotting import scatter_matrix
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import timeit


from collections import Counter,OrderedDict
    

#from sklearn.metrics import classification_report
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#import gc

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from numpy import cov



import IRI_cfg as cfg   # my constants setting file

#from functools import reduce
#from operator import concat

    

def get_load_filename_details(filename):
  #  print("6",filename[6])
  #  print("8:13",filename[8:13])
  #  print("14:19",filename[14:19])
  #  print("20:25",filename[20:25])

   # print("4:11",filename[4:11])
    print("load filename=",filename)
    return filename[6],filename[8:13],filename[14:19],filename[20:25]



def load_df(filename):
    return pd.read_pickle(filename)




def load_df_dict(pklsave):
    with open(pklsave,"rb") as f2:
        savenames=pickle.load(f2)

    if savenames:
        print("Savenames unpickled=",savenames,"\n\n")
    else:
        print("unpickling error on",cfg.pklsave,"\n\n")


    df_dict=dict({})
    for savename in savenames:
        df_flag,key,elem, query_no=get_load_filename_details(savename)
        key=int(key[1:])
        df_dict.setdefault(key, [])
        if df_flag:
            df_dict[key].append(load_df(savename))
        else:
            with open(savename, 'rb') as f2:
                #df_details.append(pickle.load(f2))
                df_dict[key].append(pickle.load(f2))
        #df_dict[key].append(df_details) 
    print("\n\ndf dict unpickled\n\n")
    return df_dict
   




##############################################################################################3333

def main():
  #  pd.set_option('display.expand_frame_repr', False)
  #  pd.set_option('display.max_rows', None)
    f=open(cfg.logfile,"w")

    startdate=pd.to_datetime(cfg.startdatestr, format="%Y/%m/%d %H:%M:%S")
    finishdate=pd.to_datetime(cfg.finishdatestr, format="%Y/%m/%d %H:%M:%S")

    # load df dictionary
    df_dict=load_df_dict(cfg.pklsave)
 

    # load colnames dictionary
    with open(cfg.colnamespklsave,"rb") as f3:
        df_colnames_dict=pickle.load(f3)



    print("\n\n\nnew finished df dict=\n",df_dict,"\n\n")
    

 #   print("\n\nnew colnames dict=\n",df_colnames_dict,"\n\n")
    print("\n\n New colnames dict=\n",json.dumps(df_colnames_dict,sort_keys=False,indent=4))

##############################################33
    # display quesry columns
    for key in df_colnames_dict.keys():
        print("\nsheet",key)   #,":",df_colnames_dict[key])
        for cquery in df_colnames_dict[key]:
            print("\nquery no",list(cquery)) #[1:])  #,"+",list(key2.values())[0])
            print(df_colnames_dict[key][1:4])
        #    print(cquery[0])
            #,df_colnames_dict.items())
      #  highest=max(df_colnames_dict.items())
      #  print("highest=",highest)




##########################################################################3

##    important_attributes=["bb_promo_disc","sd_promo_disc","bm_promo_disc","bb_total_units","sd_total_units","bm_total_units"]
##
##    scatter_matrix(X_df[important_attributes],alpha=0.2,figsize=(12,9))
##    plt.show()
##
##
## #   important_attributes2=["beerenberg_upspw_incremental","st_dalfour_upspw_incremental","bon_maman_upspw_incremental","cottees_upspw_incremental"]
##    important_attributes=["date","bb_total_units","sd_total_units","bm_total_units","c_total_units","bb_promo_disc","sd_promo_disc","bm_promo_disc","c_promo_disc"]
##    dbh=dbg[important_attributes]
##    corr_matrix=dbh.corr(method="pearson")    #important_attributes)   #.sort_values(ascending=False)   #[important_attributes])   #important_attributes2,method="pearson")
##
##    
##    corr_matrix=X_df.corr()   #.sort_values(ascending=False)   #[important_attributes])   #important_attributes2,method="pearson")
##    print("\n\nCorrelations:\n",corr_matrix,"\n\n")
##
##
##    sarraydf = pd.DataFrame (corr_matrix)
##
######## save to xlsx file
##    print("Correlation matrix array saved to",cfg.scalerdump1)
##    sarraydf.to_excel(cfg.scalerdump1, index=True)
##
##
##
##    dbe[["bb_promo_disc","sd_promo_disc","bm_promo_disc","c_promo_disc"]].hist()
##    plt.show()
##    dbe[["sd_promo_disc","bm_promo_disc","c_promo_disc"]].plot(kind='density', subplots=True, layout=(3,3), sharex=False)
##    plt.show()
##

    
############################################################################3

    f.close()
    return 



if __name__ == '__main__':
    main()

