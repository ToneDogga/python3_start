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
import itertools

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
   

def column_name(col_no,colnames,spreadsheetno):
    # returns a list of column names from the colnames dictionary 
    # the column name is a concatted and shrunk
    #
   # print("col_no1",col_no1)
   # print("col_no2",col_no2)

  #  print("colnames[0]=",colnames[0])   # 0 is the first spreadhsheet
    names=[]
    
    separator = '-'
    names.append(separator.join(colnames[spreadsheetno][col_no[0]]))
    names.append(separator.join(colnames[spreadsheetno][col_no[1]]))

   # print("names=",names)
    return names


##############################################################################################3333

def main():
  #  pd.set_option('display.expand_frame_repr', False)
  #  pd.set_option('display.max_rows', None)
    f=open(cfg.logfile,"w")




#################################################################
    # load raw spreadsheet for column combinations later
    with open(cfg.dfpklsave, 'rb') as handle:
        fullspreadsheetsave=pickle.load(handle)    #, protocol=pickle.HIGHEST_PROTOCOL)
        
    startdate=pd.to_datetime(cfg.startdatestr, format="%Y/%m/%d %H:%M:%S")
    finishdate=pd.to_datetime(cfg.finishdatestr, format="%Y/%m/%d %H:%M:%S")


    for key in fullspreadsheetsave.keys():
        fullspreadsheetsave[key][0]= fullspreadsheetsave[key][0][4:]  # remove headers

        
        fullspreadsheetsave[key][0] =  fullspreadsheetsave[key][0].set_index(pd.DatetimeIndex(pd.to_datetime(fullspreadsheetsave[key][0][cfg.column_zero_name], format="%Y/%m/%d %H:%M:%S",infer_datetime_format=True)))

        fullspreadsheetsave[key][0].drop(columns=[cfg.column_zero_name],inplace=True)

        fullspreadsheetsave[key][0].dropna(inplace=True)

        fullspreadsheetsave[key][0].sort_index(axis=0,ascending=[True],inplace=True)

        start =fullspreadsheetsave[key][0].index.searchsorted(startdate)    #dt.datetime(2013, 1, 2))
        finish=fullspreadsheetsave[key][0].index.searchsorted(finishdate)
        fullspreadsheetsave[key][0]=fullspreadsheetsave[key][0].iloc[start:finish]



#################################################################################333

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
    #  extra calculations, smoothing and scaling
    
    final_df["smooth_ww"]=final_df["ww_units_total"].rolling("42d",min_periods=3).mean()
    final_df["smooth_coles"]=final_df["coles_units_total"].rolling("42d",min_periods=3).mean()

    final_df["smooth_ww_mainstream"]=final_df["ww_units_mainstream"].rolling("42d",min_periods=3).mean()
    final_df["smooth_coles_mainstream"]=final_df["coles_units_mainstream"].rolling("42d",min_periods=3).mean()

    final_df["smooth_ww_premium"]=final_df["ww_units_premium"].rolling("42d",min_periods=3).mean()
    final_df["smooth_coles_premium"]=final_df["coles_units_premium"].rolling("42d",min_periods=3).mean()

    final_df["ww_total_mshare"]=final_df["smooth_ww"]/(final_df["smooth_ww"]+final_df["smooth_coles"])*100
    final_df["coles_total_mshare"]=final_df["smooth_coles"]/(final_df["smooth_ww"]+final_df["smooth_coles"])*100

    final_df["ww_premium_mshare"]=final_df["smooth_ww_premium"]/(final_df["smooth_ww_premium"]+final_df["smooth_coles_premium"])*100
    final_df["coles_premium_mshare"]=final_df["smooth_coles_premium"]/(final_df["smooth_ww_premium"]+final_df["smooth_coles_premium"])*100

    final_df["ww_mainstream_mshare"]=final_df["smooth_ww_mainstream"]/(final_df["smooth_ww_mainstream"]+final_df["smooth_coles_mainstream"])*100
    final_df["coles_mainstream_mshare"]=final_df["smooth_coles_mainstream"]/(final_df["smooth_ww_mainstream"]+final_df["smooth_coles_mainstream"])*100








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
        plt.savefig("plot"+str(plots)+".png")
        
        scatter_matrix(final_df[column_list],alpha=0.7,figsize=(12,9))
      #  plt.show()
        plt.savefig("scatter"+str(plots)+".png")
        plt.close("all")


#################################################################
        # plot every combination of 4 columns with one column in each of no_of_features columns


    no_of_features_list=[]
    no_of_measures_list=[]

  #  print(df_dict)
    i=0
    for ss in df_dict:
        no_of_features_list.append(df_dict[i][1][2])
        no_of_measures_list.append(df_dict[i][1][3])
        i+=1
        
    print(i,"no of features list=",no_of_features_list)
    print("no of measures list=",no_of_measures_list)
    f.write("no of features list="+str(no_of_features_list)+"\n")
    f.write("no of measures list="+str(no_of_measures_list)+"\n")
     
    



    usable_measures=[0,4,5,8,9,10,11,12,13]
    len_usable_measures=len(usable_measures)

    measures_array=np.array(usable_measures).T.reshape(-1,1)

    print("measures=",measures_array)
    lenma=len(measures_array)
    
    start_plot=plots+1


##    if False:
##        
##        print("\n\nCombination plot...\n")
##
###        start_plot=plots+1
##
##  #  no_of_features=10
  #  no_of_measures=14

##
##        print("Calculating combinations based on #features=",no_of_features_list,"#measures",no_of_measures_list,"\n")
##
##
##
##        h=range(0,max(no_of_features_list))
##
##        new_array = np.array(np.meshgrid(h,h,h,h)).T.reshape(-1,4)
##        unique_list=[]
##        for elem in new_array:
##           if len(elem)==len(set(elem)):
##               unique_list.append(list(elem))
##
##        #unique_list.sort()
##        #print("list",unique_list)
##
##        unique_list2 = sorted(list(set(map(tuple,unique_list))))
##
##        #print("u",unique_list2)
##        #unique_array=np.array(unique_list2)
##        del unique_list
##
##        unique_list3=[]
##        i=0
##        for row in unique_list2:
##           sr=tuple(sorted(row))
##         #  print("sr=",sr)
##
##           j=0
##           for row2 in unique_list2:
##              if i!=j:
##                 if sr==row2:
##                 #   print("match",i,":",sr,j,":",row2)
##                    break
##                 else:
##                #    print(" no match appending",i,":",sr,j,":",row2)
##                    unique_list3.append(sr)
##               #     print("ul=",unique_list3)   
##              j+=1
##           i+=1
##
##        del unique_list2
##
##        uniques=np.array(list(set(unique_list3)))                      
##        #print("uniques=",uniques)
##        no_of_uniques=len(uniques)
##        #print("no_of uniques=",no_of_uniques)
##
##
##        #print("len measures array",lenma)
##
##
##        i=0
##        big_list=[]
##        for row in uniques:
##           for j in range(0,lenma):
##              row3=list(np.array(row)*max(no_of_measures_list))+measures_array[j]+1
##              big_list.append(row3)
##              i+=1
##
##        uniques=np.array(big_list).reshape(-1,4)
##        len_uniques=uniques.shape[1]
##        print(uniques)
##        #print(l.shape)
##           
##
##    #####################################################################33333
##
##        
##        plot_work_size=len(uniques)
##        print("Plot work size=",plot_work_size,"\n")
##
##        
##        for spreadsheetno in range(0,2):   #    fullspreadsheetsave.keys():
##        
##            print("Final_df columns=\n",fullspreadsheetsave[spreadsheetno][0].columns)
##          #  f.write("Final_df columns=\n"+str(fullspreadsheetsave[spreadsheetno][0].columns)+"\n\n")
##
##                        
##            for p in range(plot_work_size):
##                column_list=list(uniques[p].astype(str))
##      #          print("\nPlotting spreadsheet:",spreadsheetno,"plot no:",p+start_plot,":",column_list,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
##                print("Plotting spreadsheet:",spreadsheetno,"plot no:",p+start_plot,"/",(2*plot_work_size)+start_plot+1,":",column_list)   #,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
##
##             #   f.write("\nPlotting spreadsheet: "+str(spreadsheetno)+" plot no: "+str(p+start_plot)+" : "+str(column_list)+"\n\n")   #,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
##
##                fullspreadsheetsave[spreadsheetno][0].plot(y=column_list)
##              #  plt.show()
##
##                plt.savefig("plot_"+str(spreadsheetno)+"_"+str(p)+".png")
##                plt.close("all")
##
##
##
##        start_plot=plots+1
##
##
##
##                        
##        for p in range(plot_work_size):
##            column_list=list(uniques[p].astype(str))
##    #          print("\nPlotting spreadsheet:",spreadsheetno,"plot no:",p+start_plot,":",column_list,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
##            print("Plotting spreadsheet:",spreadsheetno,"plot no:",p+start_plot,"/",(2*plot_work_size)+start_plot+1,":",column_list)   #,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
##
##         #   f.write("\nPlotting spreadsheet: "+str(spreadsheetno)+" plot no: "+str(p+start_plot)+" : "+str(column_list)+"\n\n")   #,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
##
##            fullspreadsheetsave[0][0].plot(y=full_column_list)
##            fullspreadsheetsave[1][0].plot(y=full_column_list)
##            plt.show()
##
##            plt.savefig("plot0_vs_1_"+str(p)+".png")
##            plt.close("all")
##
##
##
##
##            
##        del uniques
##        
#############################################33
# plot double combinations of coles and woolworths


    print("\n\nCombination plot...\n")


    h=range(0,max(no_of_features_list))

    new_array = np.array(np.meshgrid(h,h)).T.reshape(-1,2)
    unique_list=[]
    for elem in new_array:
       if len(elem)==len(set(elem)):
           unique_list.append(list(elem))

    #unique_list.sort()
    #print("list",unique_list)

    unique_list2 = sorted(list(set(map(tuple,unique_list))))

    #print("u",unique_list2)
    #unique_array=np.array(unique_list2)
    del unique_list

    unique_list3=[]
    i=0
    for row in unique_list2:
       sr=tuple(sorted(row))
     #  print("sr=",sr)

       j=0
       for row2 in unique_list2:
          if i!=j:
             if sr==row2:
             #   print("match",i,":",sr,j,":",row2)
                break
             else:
            #    print(" no match appending",i,":",sr,j,":",row2)
                unique_list3.append(sr)
           #     print("ul=",unique_list3)   
          j+=1
       i+=1

    del unique_list2

    uniques=np.array(list(set(unique_list3)))                      
    #print("uniques=",uniques)
    no_of_uniques=len(uniques)
    #print("no_of uniques=",no_of_uniques)


    plot_work_size=len(uniques)
    print("Plot work size=",plot_work_size,"\n")

    #print("measures=",measures_array)
    lenma=len(measures_array)
    #print("len measures array",lenma)


    i=0
    big_list=[]
    for row in uniques:
       for j in range(0,lenma):
          row3=list(np.array(row)*max(no_of_measures_list))+measures_array[j]+1
          big_list.append(row3)
          i+=1

    uniques=np.array(big_list).reshape(-1,2)
#    len_uniques=uniques.shape[1]
#    print(uniques)
     
                    
    for p in range(plot_work_size):
        column_list=list(uniques[p].astype(str))
      #  print("\ncolumn list=",column_list)
        
        clean_df_list=[]
        clean_df_list.append(fullspreadsheetsave[0][0][column_list])  # two from WW
        clean_df_list.append(fullspreadsheetsave[1][0][column_list])  # two from coles

     #   print("clean df list=",clean_df_list)
        
        joined_df=pd.concat(clean_df_list,axis=1)     # , left_index=True, right_index=True)

        
    #    print("joined=",joined_df.head(5))

        full_column_list=[]
        full_column_list.append(column_name(column_list,colnames,0))   # 0 is first spreadsheet
        full_column_list.append(column_name(column_list,colnames,1))   # 0 is first spreadsheet

        double_column_list=column_list*2

   #     print("double column list=",double_column_list)
   #     print("full column list=\n",full_column_list)
        merged_full_column_list = list(itertools.chain(*full_column_list))
        print("\nfull column list=\n",merged_full_column_list)
        f.write("\nfull column list=\n"+str(merged_full_column_list)+"\n\n")



        joined_df.columns=merged_full_column_list
       # print("joined_df=\n",joined_df)
        #plt.ylabel(merged_full_column_list)
        
#          print("\nPlotting spreadsheet:",spreadsheetno,"plot no:",p+start_plot,":",column_list,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
        print("plot no:",p+start_plot,"/",(plot_work_size)+start_plot-1,":",double_column_list)   #,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))
        f.write("plot no:"+str(p+start_plot)+"/"+str((plot_work_size)+start_plot-1)+":"+str(double_column_list)+"\n\n")   #,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))

     #   f.write("\nPlotting spreadsheet: "+str(spreadsheetno)+" plot no: "+str(p+start_plot)+" : "+str(column_list)+"\n\n")   #,fullspreadsheetsave[spreadsheetno][0][column_list].head(5))

     #   joined.plot(y=column_list)
        joined_df.plot(y=merged_full_column_list)
##        i=0
##        for ax in joined_df[double_column_list[i]]:
##            print("ax=",ax)
##            line, = ax.plot(y=double_column_list[i])
##            line.set_label(merged_full_column_list[i])
##            i+=1
            
        plt.legend(merged_full_column_list,fontsize="xx-small",loc="best")
        #plt.legend(merged_full_column_list,fontsize="xx-small",loc="best") 

        plt.title("WW vs Coles")
        plt.xlabel("Scan week")


    #    plt.show()
        plt.savefig("plot_vs_"+str(p+start_plot)+".png")
        plt.close("all")
##




 
#
# 2) St Dalfours incremental sales are huge during promotions, particularly in WW.
# is the growth bad ie pantry loading or
# good stealing from Beerenberg or Bon Maman?
# 



 
############################################################################3

    f.close()
    return 



if __name__ == '__main__':
    main()

