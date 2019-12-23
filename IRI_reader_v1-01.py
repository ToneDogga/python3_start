# IRI reader by anthony Paech 23/12/19
#
# IRI files are formatted
# Market (row 0)
# product (row 1)
# Measure (row 2)
#  With date as the only index in column 0
#
# read the raw untouched file from IRI temple 9 spreadsheet into pandas dataframe
# config variables saved as the filename in IRI_cfg.py
# create a dictionary of the column headings based on column numbers as keys to the column names.  eg column "3"
# so to get a column full name print(colnames["3"]
#
# Also create a dictionary of the column numbers as keys to the name of measure and a list of the column numbers of all the measures of that type
#


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


#!/usr/bin/python3
from __future__ import print_function
from __future__ import division


import numpy as np
import pandas as pd
#from pandas.plotting import scatter_matrix

import math
import csv
import sys
import datetime as dt
import joblib
import pickle
import json

import IRI_cfg as cfg   # my constants setting file

#from functools import reduce
#from operator import concat

def read_excel(filename,rows,no_of_cols):
    xls = pd.read_excel(filename,sheet_name="Sheet1",header=None,index_col=None,skip_blank_lines=False,keep_default_na=False,names=list(map(str, range(no_of_cols+1))))    #'salestransslice1.xlsx')
    if rows==-1:
        return xls   #.parse(xls.sheet_names[0])
    else:        
        return xls.head(rows)


def temp_columns(df,colnames,collist,choices):
# return the columns needed for a calc in a tempdf in the collist list

    print("choices=",choices)
    print("column=",collist[1][0])
    print("name=",collist[0])
    print("linked columns=",collist[1])
    colkey=collist[1][0]
    coldict={}
    i=0
    columns_list=[]
    for col in collist[1]:
        if i in choices:
            coldict.update({col : colnames[col]})
            columns_list.append(col)
        i+=1
        
 #  return df.loc[:, collist[1]],coldict
    print("cl=",columns_list) 
    return df.loc[:, columns_list],coldict


def temp_columns_by_col_no(df,colnames,collist):
# return the columns needed for a calc in a tempdf in the collist list

    print("columns selected=",collist)
    coldict={}

    for col in collist:
        coldict.update({col : colnames[col]})
            
        
        
 #  return df.loc[:, collist[1]],coldict
     
    return df.loc[:, collist],coldict



def build_working_df(df_list):
    final_X_df=df_list[0]
    for i in range(1,len(df_list)):
        final_X_df=pd.concat([final_X_df, df_list[i]],axis=1)     # , left_index=True, right_index=True)
    return final_X_df




def main():
  #  pd.set_option('display.expand_frame_repr', False)
  #  pd.set_option('display.max_rows', None)

    f=open(cfg.logfile,"w")
    print(cfg.infilename)
    f.write("IRI spreadsheet reader "+str(cfg.infilename)+"\n\n")
    
    df=read_excel(cfg.infilename,-1,cfg.no_of_cols)  # -1 means all rows
    if df.empty:
        print(cfg.infilename,"Not found. Check IRI_cfg.py file")
        f.write(cfg.infilename+" Not found. Check IRI_cfg.py file\n")
        sys.exit()

    
##################################################33

# create a dictionary "colnames" of column names indexed by column number integer as a string
    colnames=df[0:3].to_dict("list")  #other options : series records split index dict list
  #  print(colnames)

    coltwo=df[1:2].to_dict("list")

  #  print(coltwo)            


    for col in range(cfg.no_of_cols+1):
        colnames[str(col)][0]=cfg.colonevalue

 #   print(colnames["0"][2])
    colnames["0"][1]=cfg.column_zero_name
    colnames["0"][2]=cfg.column_zero_name

    coltwoval=coltwo["1"][0]
  #  print("coltwoval=",coltwoval)
    for col in range(2,cfg.no_of_cols+1):
        if colnames[str(col)][1]=="":
            colnames[str(col)][1]=coltwoval
        else:
            coltwoval=coltwo[str(col)][0]
        
   # print(colnames)
  #  print("\nColumn names=\n",json.dumps(colnames,sort_keys=False,indent=4))
    f.write("\nColumn names=\n"+str(json.dumps(colnames,sort_keys=False,indent=4))+"\n\n")

##################################################################
# create a dict grouping the common measures as values under integer index

    coldict=dict()
    for col in range(1,cfg.unique_measures_count+1):
        indexes=[]
        for u in range(cfg.unique_features_count):
            indexes.append(str(col+u*cfg.unique_measures_count))    #*unique_features_count) 
 #       print(indexes)
        coldict.update({str(col):[colnames[str(col)][2],indexes]})    #str(col+1*unique_measures_count),str(col+2*unique_measures_count)]})

  #  print("\nColumn links=\n",json.dumps(coldict,sort_keys=False,indent=4))
    f.write("\nColumn links=\n"+str(json.dumps(coldict,sort_keys=False,indent=4))+"\n\n")
    
################################################################33
# turn the raw excel import into a dataframe we can use
            
    X_df=df.iloc[3:,:cfg.no_of_cols+1]
      
    del df  # clear memory 

  #  print("Imported into pandas=\n",X_df.columns,"\n",X_df.shape)    #head(10))

##################################################33
##   # Remove ena(inplace=True)
##
##    X_df=date_deltas(X_df)
##
##    X_df=promotions(X_df)    # calculate the GMV (Gross margin value) as salesval-costval.  Used to highlight promotions
##    X_df.drop(columns=["salesval","costval"],inplace=True)
##    to_datetime
####    label_encoder=LabelEncoder()
####    X_df["prod_encode"] = label_encoder.fit_transform(X_df["product"].to_numpy())
####    joblib.dump(label_encoder,open(cfg.product_encode_save,"wb"))
#### #   X_df.drop(columns=["product"],inplace=True)
##
##    label_encoder=LabelEncoder()
##    X_df["code_encode"] = label_encoder.fit_transform(X_df["code"].to_numpy())
##    joblib.dump(label_encoder,open(cfg.code_encode_save,"wb"))
##    X_df.drop(columns=["code"],inplace=True)
##    print(X_df.columns)
##
##
#####################################################
##    
##    #Xr_df = Xr_df[["prod_encode","qty","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]]  
###    Xr_df = Xr_df[["prod_encode","qty","productgroup","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]]  
##    Xc_df = X_df[cfg.featureorder_c]    # for classifier
##    Xr_df = X_df[cfg.featureorder_r]    # for regression
## #   print(X_df.columns)
##
##    del X_df   # clear memory
##    Xr_df=order_delta(Xr_df)
##  #  Xc_df=order_delta(Xc_df)
##
##    label_encoder=LabelEncoder()
##    Xr_df["prod_encode"] = label_encoder.fit_transform(Xr_df["product"].to_numpy())
##    joblib.dump(label_encoder,open(cfg.product_encode_save,"wb"))
## #   Xr_df.drop(columns=["product"],inplace=True)
##
    


############################################3
#  Dates limiting .      print("Imported into pandas=\n",X_df.shape)    #head(10))


    startdate=pd.to_datetime(cfg.startdatestr, format="%Y/%m/%d %H:%M:%S")
    finishdate=pd.to_datetime(cfg.finishdatestr, format="%Y/%m/%d %H:%M:%S")

    X_df = X_df.set_index(pd.DatetimeIndex(pd.to_datetime(X_df[cfg.column_zero_name], format="%Y/%m/%d %H:%M:%S",infer_datetime_format=True)))

    X_df.drop(columns=[cfg.column_zero_name],inplace=True)

    X_df.dropna(inplace=True)

    X_df.sort_index(axis=0,ascending=[True],inplace=True)

    start = X_df.index.searchsorted(startdate)    #dt.datetime(2013, 1, 2))
    finish= X_df.index.searchsorted(finishdate)
    X_df=X_df.iloc[start:finish]


    print("Imported into pandas=\n",cfg.infilename,"shape:",X_df.shape,"\n\n")    #head(10))

    f.write("X_df=\n"+X_df.to_string()+"\n\n")

    f.write("Imported into pandas=\n"+str(cfg.infilename)+" shape:"+str(X_df.shape)+"\n\n")    #head(10))


    
    


   # print(X_df) 

################################################3
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


#test for expanding market share
#testing baseline and incremental unit sales stealing
#
#  test coles <-> ww switching
#  Testing trading up or down
#
#  test percent sold on promotion


######################################################################################3
# create a list of dataframes and then concatenate them horizonally
#  they are all column slices of the same X_df so are the same length
    df_list=[]
    df_colnames_dict=dict()

# Build dafaframes for calculations
############################################################
# 0) total units vs mainstream and premium

    tempcols,tempcolnames=temp_columns_by_col_no(X_df,colnames,["1","15","29"])
    df_list.append(tempcols)
    df_colnames_dict.update(tempcolnames)
    print(tempcols)
    print("\ndf colnames links=\n",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")


######################
 #   1)Baseline and incr vs total category   

    tempcols,tempcolnames=temp_columns_by_col_no(X_df,colnames,["48","62","44","58","54","55","68","69"])
    df_list.append(tempcols)
    df_colnames_dict.update(tempcolnames)
    print(tempcols)
    print("\ndf colnames links=\n",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")

######################
 #  2) Baseline and incr vs premium segment   

    tempcols,tempcolnames=temp_columns_by_col_no(X_df,colnames,["43","47","57","61"])
    df_list.append(tempcols)
    df_colnames_dict.update(tempcolnames)
    print(tempcols)
    print("\ndf colnames links=\n",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")



######################
 #  3) sd vs BB strawberry jam 

    tempcols,tempcolnames=temp_columns_by_col_no(X_df,colnames,["71","72","85","86","99","100"])
    df_list.append(tempcols)
    df_colnames_dict.update(tempcolnames)
    print(tempcols)
    print("\ndf colnames links=\n",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")











##################################################
#   Report on dataframe created
##    for i in range(0,len(df_list)):
##        print("\n\n",df_list[i])
##    print("\ndf colnames links=\n",json.dumps(df_colnames_dict,sort_keys=False,indent=4))
##    f.write("\ndf Colnames links=\n"+str(json.dumps(df_colnames_dict,sort_keys=False,indent=4))+"\n\n")
##
##
##
##    print("\n\n")

###################################################################################333
  #  print(len(df_list))
  # join all the working df's together
    
##    final_X_df=build_working_df(df_list)
##        
##
##    print("final=\n",final_X_df)
##    #print("df colnames list=\n",df_colnames_dict)
##    print("\ndf colnames links=\n",json.dumps(df_colnames_dict,sort_keys=False,indent=4))
##    f.write("\ndf Colnames links=\n"+str(json.dumps(df_colnames_dict,sort_keys=False,indent=4))+"\n\n")


###########################################################################3
# Calculations
# 0)   premium segment vs total
    df=df_list[0]
    df["smoothed_total_units"]=df["1"].rolling("50d",min_periods=3).mean()   # centre=True
    df["smoothed_premium_units"]=df["29"].rolling("50d",min_periods=3).mean()   # centre=True

  #  df["premium_percent"]=df["29"]/df["1"]
    df["smoothed_premium_percent"]=df["smoothed_premium_units"]/df["smoothed_total_units"]
    
    print(df)


  


########################################################################################    


    
##
####    X_df["bb_promo_disc"]=X_df["bb_promo_disc"].round(0)
####    X_df["sd_promo_disc"]=X_df["sd_promo_disc"].round(0)
####    X_df["c_promo_disc"]=X_df["c_promo_disc"].round(0)
####    X_df["bm_promo_disc"]=X_df["bm_promo_disc"].round(0)
####    # calculated totals and reorder remaining columns
####    X_df["bb_upw_baseline"]=X_df["bb_upspw_baseline"]*X_df["bb_dd"]
####    X_df["bb_upw_total"]=(X_df["bb_upspw_baseline"]+X_df["bb_upspw_incremental"])*X_df["bb_dd"]
####    X_df["sd_upw_baseline"]=X_df["sd_upspw_baseline"]*X_df["sd_dd"]
####    X_df["sd_upw_total"]=(X_df["sd_upspw_baseline"]+X_df["sd_upspw_incremental"])*X_df["sd_dd"]
####    X_df["c_upw_baseline"]=X_df["c_upspw_baseline"]*X_df["c_dd"]
####    X_df["c_upw_total"]=(X_df["c_upspw_baseline"]+X_df["c_upspw_incremental"])*X_df["c_dd"]
####    X_df["bm_upw_baseline"]=X_df["bm_upspw_baseline"]*X_df["bm_dd"]
####    X_df["bm_upw_total"]=(X_df["bm_upspw_baseline"]+X_df["bm_upspw_incremental"])*X_df["bm_dd"]
##
##    X_df["promo_sum"]=X_df["bb_on_promo"]+X_df["sd_on_promo"]+X_df["c_on_promo"]+X_df["bm_on_promo"]
##
#####################################
### rolling averages for baseline unit sales
##
### hang_over_weeks_ahead
##    hang_over_weeks_ahead=2   # 2 weeks 14 days
##
##    X_df.sort_index(axis='index',ascending=[True],inplace=True)
##
##
##    X_df["bb_ave_baseline_units"]=X_df["bb_baseline_units"].rolling("42d",min_periods=3).mean()   # centre=True
##    X_df["sd_ave_baseline_units"]=X_df["sd_baseline_units"].rolling("42d",min_periods=3).mean()
##    X_df["c_ave_baseline_units"]=X_df["c_baseline_units"].rolling("42d",min_periods=3).mean()
##    X_df["bm_ave_baseline_units"]=X_df["bm_baseline_units"].rolling("42d",min_periods=3).mean()
##
####    X_df["bb_baseline_units_next_3_weeks"]=X_df["bb_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
####    X_df["sd_baseline_units_next_3_weeks"]=X_df["sd_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
####    X_df["c_baseline_units_next_3_weeks"]=X_df["c_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
####    X_df["bm_baseline_units_next_3_weeks"]=X_df["bm_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
##
##    X_df["bb_baseline_units_next_week"]=X_df["bb_baseline_units"].shift(periods=-1).rolling("14d",min_periods=1).mean()#.shift(periods=-1)
##    X_df["sd_baseline_units_next_week"]=X_df["sd_baseline_units"].shift(periods=-1).rolling("14d",min_periods=1).mean()#.shift(periods=-1)
##    X_df["c_baseline_units_next_week"]=X_df["c_baseline_units"].shift(periods=-1).rolling("14d",min_periods=1).mean()#.shift(periods=-1)
##    X_df["bm_baseline_units_next_week"]=X_df["bm_baseline_units"].shift(periods=-1).rolling("14d",min_periods=1).mean()#.shift(periods=-1)
##
####    X_df["sd_baseline_units_next_3_weeks"]=X_df["sd_baseline_units"].shift(periods=-1).rolling(3,min_periods=1).mean()
####    X_df["c_baseline_units_next_3_weeks"]=X_df["c_baseline_units"].shift(periods=-1).rolling(3,min_periods=1).mean()
####    X_df["bm_baseline_units_next_3_weeks"]=X_df["bm_baseline_units"].shift(periods=-1).rolling(3,min_periods=1).mean()
##
##
##    X_df["bb_baseline_units_hangover"]=(X_df["bb_ave_baseline_units"]-X_df["bb_baseline_units_next_week"])*hang_over_weeks_ahead
##    X_df["sd_baseline_units_hangover"]=(X_df["sd_ave_baseline_units"]-X_df["sd_baseline_units_next_week"])*hang_over_weeks_ahead
##    X_df["c_baseline_units_hangover"]=(X_df["c_ave_baseline_units"]-X_df["c_baseline_units_next_week"])*hang_over_weeks_ahead
##    X_df["bm_baseline_units_hangover"]=(X_df["bm_ave_baseline_units"]-X_df["bm_baseline_units_next_week"])*hang_over_weeks_ahead
##
## #   X_df["bb_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["bb_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)
## #   X_df["sd_baseline_units_hangover"]=X_df["sd_baseline_units"]-X_df["sd_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)
####    X_df["c_baseline_units_hangover"]=X_df["c_baseline_units"]-X_df["c_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)
####    X_df["bm_baseline_units_hangover"]=X_df["bm_baseline_units"]-X_df["bm_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)
##
##
## #   X_df["bb_baseline_units_3wkave"]=X_df["bb_baseline_units_next_3_weeks"].rolling(3,min_periods=1).mean().shift(periods=-2)
##
##
####    X_df["bb_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["bb_baseline_units_next_3_weeks"]
####    X_df["sd_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["sd_baseline_units_next_3_weeks"]
####    X_df["c_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["c_baseline_units_next_3_weeks"]
####    X_df["bm_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["bm_baseline_units_next_3_weeks"]
####
##
##
##    X_df["bb_baseline_units_var"]=X_df["bb_baseline_units"]-X_df["bb_ave_baseline_units"]
##    X_df["sd_baseline_units_var"]=X_df["sd_baseline_units"]-X_df["sd_ave_baseline_units"]
##    X_df["c_baseline_units_var"]=X_df["c_baseline_units"]-X_df["c_ave_baseline_units"]
##    X_df["bm_baseline_units_var"]=X_df["bm_baseline_units"]-X_df["bm_ave_baseline_units"]
##
##
##    X_df["bb_gains_from_others"]=-(X_df["sd_baseline_units_var"]+X_df["bm_baseline_units_var"])
##    X_df["sd_gains_from_others"]=-(X_df["bb_baseline_units_var"]+X_df["bm_baseline_units_var"])
##    X_df["c_gains_from_others"]=-(X_df["bb_baseline_units_var"]+X_df["sd_baseline_units_var"]+X_df["bm_baseline_units_var"])
##    X_df["bm_gains_from_others"]=-(X_df["bb_baseline_units_var"]+X_df["sd_baseline_units_var"])
##
##
##   # print("\nBB only X_df=\n",X_df[["bb_baseline_units","bb_baseline_units_var","sd_baseline_units","sd_ave_baseline_units","sd_baseline_units_var","c_baseline_units","c_ave_baseline_units","c_baseline_units_var","bm_baseline_units","bm_ave_baseline_units","bm_baseline_units_var"]])
##
## #   print("\nBB only X_df=\n",X_df[["bb_baseline_units","bb_ave_baseline_units","bb_baseline_units_var","bb_gains_from_others","sd_baseline_units","sd_baseline_units_var","sd_gains_from_others","c_baseline_units","c_baseline_units_var","c_gains_from_others","bm_baseline_units","bm_baseline_units_var","bm_gains_from_others"]])
##
##    X_df["sd_gains_from_others"]*=X_df["sd_on_promo"]
##    X_df["sd_baseline_units_hangover"]*=X_df["sd_on_promo"]
##
##################################################333
##
##    X_df=X_df[["bb_on_promo","sd_on_promo","c_on_promo","bm_on_promo","promo_sum","bb_upspw_baseline","sd_upspw_baseline","c_upspw_baseline","bm_upspw_baseline","bb_upspw_incremental","sd_upspw_incremental","c_upspw_incremental","bm_upspw_incremental","bb_dd","sd_dd","c_dd","bm_dd","bb_promo_disc","sd_promo_disc","c_promo_disc","bm_promo_disc","bb_total_units","sd_total_units","c_total_units","bm_total_units","bb_baseline_units","sd_baseline_units","c_baseline_units","bm_baseline_units","bb_ave_baseline_units","sd_ave_baseline_units","c_ave_baseline_units","bm_ave_baseline_units","bb_baseline_units_var","sd_baseline_units_var","c_baseline_units_var","bm_baseline_units_var","bb_gains_from_others","sd_gains_from_others","c_gains_from_others","bm_gains_from_others","bb_incremental_units","sd_incremental_units","c_incremental_units","bm_incremental_units","bb_baseline_units_next_week","sd_baseline_units_next_week","c_baseline_units_next_week","bm_baseline_units_next_week","bb_baseline_units_hangover","sd_baseline_units_hangover","c_baseline_units_hangover","bm_baseline_units_hangover"]]
##
###   X_df.sort_values(by=["coles_scan_week","promo_sum","bb_on_promo","sd_on_promo","c_on_promo","bm_on_promo"],axis=0,ascending=[True,True,True,True,True,True],inplace=True)
##
##
##
##    start = X_df.index.searchsorted(startdate)    #dt.datetime(2013, 1, 2))
##    finish= X_df.index.searchsorted(finishdate)
##    X_df=X_df.iloc[start:finish]
##
##
### sd promo only with clear air
##    mask=((X_df["sd_on_promo"]==1) & (X_df["sd_incremental_units"]>=1.0) & (X_df["promo_sum"]==1))
##    #mask=((X_df["sd_incremental_units"]>=1.0) & (X_df["promo_sum"]==1))
##
##    X_bb=X_df   #[mask]
##
##
##    #X_df.sort_index(axis=0,ascending=[True],inplace=True)
##
##
##
## #   print(X_bb.columns)
## #   print("\nSD only X_df=\n",X_bb[["bb_baseline_units","bb_baseline_units_var","bb_gains_from_others","sd_baseline_units","sd_baseline_units_var","sd_gains_from_others","c_baseline_units","c_baseline_units_var","c_gains_from_others","bm_baseline_units","bm_baseline_units_var","bm_gains_from_others",]])
##    print("\nSD only X_df=\n",X_bb[["sd_on_promo","sd_promo_disc","bb_on_promo","bb_promo_disc","bb_baseline_units","bb_incremental_units","sd_baseline_units","sd_incremental_units","sd_baseline_units_hangover","sd_gains_from_others"]])
##
##### bb promo with any other
####    mask=((X_df["bb_on_promo"]==1) & (X_df["promo_sum"]==2))
####    X_bb=X_df[mask]
####    print(X_bb.columns)
####    print("\nBB with one other X_df=\n",X_bb)
####
####
##### sd promo only with clear air
####    mask=((X_df["sd_on_promo"]==1) & (X_df["promo_sum"]==1))
####    X_sd=X_df[mask]
####    print(X_sd.columns)
####    print("\nSd only X_df=\n",X_sd)
####
####
##### sd promo with one other
####    mask=((X_df["sd_on_promo"]==1) & (X_df["promo_sum"]==2))
####    X_sd=X_df[mask]
####    print(X_sd.columns)
####    print("\nSd with one other X_df=\n",X_sd)
####
####
##### c promo only with clear air
####    mask=((X_df["c_on_promo"]==1) & (X_df["promo_sum"]==1))
####    X_c=X_df[mask]
####    print(X_c.columns)
####    print("\nc only X_df=\n",X_c)
####
##### c promo with one other
####    mask=((X_df["c_on_promo"]==1) & (X_df["promo_sum"]==2))
####    X_c=X_df[mask]
####    print(X_c.columns)
####    print("\nc with 1 other X_df=\n",X_c)
####
####
##### bm promo only with clear air
####    mask=((X_df["bm_on_promo"]==1) & (X_df["promo_sum"]==1))
####    X_bm=X_df[mask]
####    print(X_bm.columns)
####    print("\nBM only X_df=\n",X_bm)
####
##### bm promo with one other
####    mask=((X_df["bm_on_promo"]==1) & (X_df["promo_sum"]==2))
####    X_bm=X_df[mask]
####    print(X_bm.columns)
####    print("\nBM with one other X_df=\n",X_bm)
####
####
##
##






    f.close()
    return 



if __name__ == '__main__':
    main()

