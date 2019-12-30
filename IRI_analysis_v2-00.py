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


import matplotlib
matplotlib.use('Agg')
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
 #   print("load filename=",filename)
    return filename[6],filename[8:13],filename[14:19],filename[20:25]



def load_df(filename):
    return pd.read_pickle(filename)




def load_df_dict(pklsave):
    with open(pklsave,"rb") as f2:
        savenames=pickle.load(f2)

    if savenames:
        pass
        print("\n")
        #print("Savenames unpickled=",savenames,"\n\n")
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
 #   print("\n\ndf dict unpickled\n\n")
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
 
    #print(df_dict)
    
    # load colnames by queryno dictionary
    with open(cfg.colnamespklsave,"rb") as f3:
        df_colnames_dict=pickle.load(f3)


    # load ALL colnames dictionary
    with open(cfg.fullcolnamespklsave,"rb") as f3:
        colnames=pickle.load(f3)

  #  f.write(json.dumps(colnames)+"\n\n")
    f.write(json.dumps(df_colnames_dict,sort_keys=False,indent=4)+"\n\n")
    

 #   print("\n\n\nnew finished df dict=\n",df_dict,"\n\n")
    

 #   print("\n\nnew colnames dict=\n",df_colnames_dict,"\n\n")
  #  print("\n\n New colnames dict=\n",json.dumps(df_colnames_dict,sort_keys=False,indent=4))

  #  print("\n\n ALL colnames dict=\n",json.dumps(colnames,sort_keys=False,indent=4))
  #  print("\n\n ALL colnames dict=\n",colnames)





##############################################33
    # display querydict columns
    for queryno in cfg.querydict.keys():
        #print("\nquery no",queryno)   #,":",df_colnames_dict[key])

        spreadsheetno=cfg.querydict[queryno][0]
        columnlist=cfg.querydict[queryno][1]
        print("\nqueryno=",queryno,"spreadsheetno=",spreadsheetno,"columnlist=",columnlist)
        f.write("\nqueryno="+str(queryno)+" spreadsheetno= "+str(spreadsheetno)+" columnlist="+str(columnlist)+"\n")

        df=df_dict[spreadsheetno][0]
        print(df)
#        f.write(json.dumps(df,sort_keys=False,indent=4))
        f.write(df.head(5).to_string()+"\n")

        f.write("\n\n")        

        for colno in columnlist:
            print("-> colno no",colno,"column=",colnames[spreadsheetno][colno])
            f.write("-> colno no:"+str(colno)+" column="+str(colnames[spreadsheetno][colno])+"\n\n")

    f.write("\n\n")

    
#########################################################################
#
#  At this point we have loaded and unpickled
#  a list of dataframes for each queryno as defined in the IRI_cfg.py file. (df_dict)
# we also have a lookup table of ALL columns available (colnames)
# and a table by spreadsheet no and queryno of the actual used columns in the queries (df_colnames_dict)
#   these were all created by IRI_reader.py and IRI_cfg.py as a config file
#
# now we have to create new columns in the existing dataframes to calculate relationships
# to test hyphothesis
#
# we also have to smooth, join , shift and then test correlation (corr) or scatter (R2)
#
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
#
# Hypothesis
# 1) St Dalfours WW sales are gaining. incremental sales are huge during promotions.
# where is the growth coming from?
# possibilities to test:
# a) total category gain
# b) WW category gain
# c) coles category loss
# d) total premium category gain
# e) WW premium category gain
# f) coles premium category loss
# g) total premium category depth of distribution gain
# h) WW premium category depth of distribution gain
# i) coles premium category depth of distribution loss
# j) total varietal growth ie Strawberry jam gain
# k) WW varietal growth ie strawberry jam gain
# l) coles varietal loss
# m) total Premium competitors baseline gain
# n) WW premium competitors baseline gain
# o) coles premium competitors baseline loss
# p) total Premium competitiors incremental gain
# q) WW premium compeitors incremental gain
# r) coles premium competitors incremental loss
# s) total Premium competitiors total gain
# t) WW premium compeitors total gain
# u) coles premium competitors total loss
# v) total Mainstream competitors baseline gain
# w) total Mainstream competitors incremental gain
# x) total Mainstream competitors total gain
# y) WW Mainstream competitors baseline gain
# z) WW Mainstream competitors incremental gain
# aa) WW Mainstream competitors total gain
# ab) coles Mainstream competitors baseline loss
# ac) coles Mainstream competitors incremental loss
# ad) coles Mainstream competitors total loss
#
#

# 
#
#############################################33
    #  d), e) f)  
    c=0
    clean_df_list=[]
    for spreadsheetno in df_dict.keys():
        if cfg.infilenamedict[spreadsheetno][1]:
            df_dict[spreadsheetno][0].columns=[cfg.infilenamedict[spreadsheetno][1]]
        clean_df_list.append(df_dict[spreadsheetno][0])
        c+=1

    final_df=pd.concat(clean_df_list,axis=1)     # , left_index=True, right_index=True)




###############################################################33
#   d), e)  and f)

 #   final_df["WW","coles"].hist()
 #   plt.show()
    
#    scatter_matrix(final_df,alpha=0.2,figsize=(12,9))
#    plt.show()
#

  #  final_df["smooth_ww"]=final_df["ww"].rolling("42d",min_periods=3).mean()
  #  final_df["smooth_coles"]=final_df["coles"].rolling("42d",min_periods=3).mean()
  #  final_df["ww_ms"]=final_df["smooth_ww"]/(final_df["smooth_ww"]+final_df["smooth_coles"])*100
  #  final_df["coles_ms"]=final_df["smooth_coles"]/(final_df["smooth_ww"]+final_df["smooth_coles"])*100

    final_df=final_df.astype(float)

#####################################################33
    #  smoothing and scaling
    
   # final_df["ww_dod"]=10   #final_df["p_ww_bl"].astype(float) *10
   # final_df["coles_dod"]=10   #final_df["p_ww_bl"].astype(float) *10
    final_df["smooth_ww"]=final_df["ww_total"].rolling("42d",min_periods=3).mean()
    final_df["smooth_coles"]=final_df["coles_total"].rolling("42d",min_periods=3).mean()


##########################################################
    print("Final_df columns=\n",final_df.columns)
    f.write("Final_df columns=\n"+str(final_df.columns)+"\n\n")
    for plots in cfg.plotdict.keys():
        print("Plotting:",plots)
        column_list=cfg.plotdict[plots]

    #    print(final_df.columns)
    #    print(final_df[column_list])

        print(final_df[column_list].corr(),"\n\n")
        f.write("\nPlot:"+str(plots)+" Corr=\n"+final_df[column_list].corr().to_string()+"\n\n\n")
        
        final_df.plot(y=column_list)
      #  plt.show()
        plt.savefig("plot_dod"+str(plots)+".png")
        
        scatter_matrix(final_df[column_list],alpha=0.7,figsize=(12,9))
      #  plt.show()
        plt.savefig("scatter_dod"+str(plots)+".png")



#############################################33
    #  B)
##    c=0
##    clean_df_list=[]
##    for spreadsheetno in df_dict.keys():
##        if cfg.infilenamedict[spreadsheetno][1]:
##            df_dict[spreadsheetno][0].columns=[cfg.infilenamedict[spreadsheetno][1]]
##        clean_df_list.append(df_dict[spreadsheetno][0])
##        c+=1
##
##    final_df=pd.concat(clean_df_list,axis=1)     # , left_index=True, right_index=True)
##
##
## #   final_df["WW","coles"].hist()
## #   plt.show()
##    
###    scatter_matrix(final_df,alpha=0.2,figsize=(12,9))
###    plt.show()
###
##
##    final_df["smooth_ww"]=final_df["ww"].rolling("42d",min_periods=3).mean()
##    final_df["smooth_coles"]=final_df["coles"].rolling("42d",min_periods=3).mean()
##    final_df["ww_ms"]=final_df["smooth_ww"]/(final_df["smooth_ww"]+final_df["smooth_coles"])*100
##    final_df["sd"]=final_df["sd"].astype(float)
##  #  final_df["ww_m"]=final_df["WW"]/(final_df["WW"]+final_df["coles"])*100
##    
## #   final_df["bb_baseline_units_next_3_weeks"]=X_df["bb_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
##
##    print(final_df.columns)
##    print(final_df)
##    print(final_df[["ww_ms","sd"]].corr())
##
##
##    final_df.plot(y=['ww_ms',"sd"])
##  #  final_df.plot(y='57')
##
##    plt.show()
##
##    scatter_matrix(final_df[["ww_ms","sd"]],alpha=0.5,figsize=(12,9))
##    plt.show()
##





  

    



#
# 2) St Dalfours incremental sales are huge during promotions, particularly in WW.
# is the growth bad ie pantry loading or
# good stealing from Beerenberg or Bon Maman?
# 












































##########################################################################3


#    X_df["c_ave_baseline_units"]=X_df["c_baseline_units"].rolling("42d",min_periods=3).mean()
#    X_df["bm_ave_baseline_units"]=X_df["bm_baseline_units"].rolling("42d",min_periods=3).mean()

##    X_df["bb_baseline_units_next_3_weeks"]=X_df["bb_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
##    X_df["sd_baseline_units_next_3_weeks"]=X_df["sd_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()

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
##    dbe[["sd_promo_disc","bm_promo_disc","c_promo_disc"]].plot(kind='density', subplots=True, layout=(3,3), sharex=False)
##    plt.show()
### turns counter dictionary into a numpy array
 #   pc=pd.DataFrame(np.array(list(pcode_counts.items())),columns=["x","y"])
   # pc=np.array(list(pcode_counts.items()))[:,1]
 
##   # print("pc=",pc)
##   # #pdf=pd.DataFrame(pc).hist()
##    plt.bar(list(pcode_counts.keys()), list(pcode_counts.values()))
##  #  plt.hist(pc,bins=100)    #,histtype='stepfilled', density=False, bins=250) # density
##
##   # sns.distplot(pc["x"],kde=False, bins=200,rug=True)
##    plt.xlabel("Product Code")
##    plt.ylabel("Frequency")
##  #  plt.title("Product code frequency as a % of total transactions")
##    plt.show()
##
##    plt.bar(list(pg_counts.keys()), list(pg_counts.values()))
##    plt.xlabel("Product Groups")
##    plt.ylabel("Frequency")
##  #  plt.title("Product group frequency as a count of total transactions")
##    plt.show()
##
##    plt.bar(list(ccode_counts.keys()), list(ccode_counts.values()))
##
##    #sns.distplot(ccf["y"],kde=False,bins=200,rug=True)
##    plt.xlabel("Customer Code")

#  covariance
##    data1=dbe[["bb_total_upspw"]]
##    data2=dbe[["sd_promo_disc"]]
##
##    covariance = cov(data1, data2)
##    print("\nCovariance between bb_total_upspw and SD promo disc=",covariance)
##
##########  Spearman R
##    corr, _ = spearmanr(data1, data2)
##    print('\nSpearmans correlation between bb_total_upspw and SD promo disc=: %.3f' % corr)

    
############################################################################3

    f.close()
    return 



if __name__ == '__main__':
    main()

