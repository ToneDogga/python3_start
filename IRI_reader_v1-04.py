# IRI reader by anthony Paech 23/12/19
#
# IRI files are formatted
# Market (row 0)
# product (row 1)
# Measure (row 2)
#  With date as the only index in column 0
#
# read in multiple raw xlsx untouched files from IRI temple 9 spreadsheet into pandas dataframe
# the file list comes from IRI_cfg.py as a dictionary
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

def read_excel(filename,rows):
    xls = pd.read_excel(filename,sheet_name="Sheet1",header=None,index_col=None,skip_blank_lines=False,keep_default_na=False,names=list(map(str, range(501))))    #'salestransslice1.xlsx')
  #  no_of_cols=xls.shape[1]
  #  print(filename,"no of cols=",xls.shape)
    if rows==-1:
        return xls    #,no_of_cols   #.parse(xls.sheet_names[0])
    else:        
        return xls.head(rows)   #,no_of_cols


def temp_columns(df,colnames,queryno,collist,choices):
# return the columns needed for a calc in a tempdf in the collist list

    print("choices=",choices)
    print("column=",collist[1][0])
    print("name=",collist[0])
    print("linked columns=",collist[1])
    colkey=collist[1][0]
    coldict=dict({})
    coldict.setdefault(queryno,[])

    i=0
    columns_list=[]
    for col in collist[1]:
        if i in choices:
            #coldict.update({key : colnames[col]})
            coldict[queryno].append(colnames[col])        
            columns_list.append(col)
        i+=1
        
 #  return df.loc[:, collist[1]],coldict
    print("cl=",columns_list) 
    return df.loc[:, columns_list],coldict


def temp_columns_by_col_no(df,colnames,queryno,collist):
# return the columns needed for a calc in a tempdf in the collist list

  # print("columns selected=",collist,"\ncolnames=\n",colnames)
    coldict=dict({})
    coldict.setdefault(queryno,[])

    for col in collist:
     #   coldict.update({key : colnames[col]})
       # coldict.setdefault(key,[])
        coldict[queryno].append(col)
        coldict[queryno].append(colnames[col])        
        
  #  coldict[queryno].insert(0,col)    
 #  return df.loc[:, collist[1]],coldict
     
    return df.loc[:, collist],coldict



def build_working_df(df_list):
    final_X_df=df_list[0]
    for i in range(1,len(df_list)):
        final_X_df=pd.concat([final_X_df, df_list[i]],axis=1)     # , left_index=True, right_index=True)
    return final_X_df


def load_IRI_xlsx(f):
    dfdict=dict({})
    i=0
    for infile in cfg.infilenamedict.keys():
        print(cfg.infilenamedict[infile])
        f.write("\n\nIRI spreadsheet reader "+str(cfg.infilenamedict[infile])+"\n\n")
    
        df=read_excel(cfg.infilenamedict[infile],-1)  # -1 means all rows
        df.dropna(axis="columns",inplace=True)
   
        if df.empty:
            print(cfg.infilenamedict[infile],"Not found. Check IRI_cfg.py file")
            f.write(cfg.infilenamdict[infile]+" Not found. Check IRI_cfg.py file\n")
        #   sys.exit()
        else:
            title1,row1_len,row2_len,row3_len=get_no_of_cols(df)
            dfdict.update({i:[df,title1,row1_len,row2_len,row3_len]})
            i+=1

    print("Spreadsheets loaded:\n\n",cfg.infilenamedict,"\n\n\n")
 #   f.write("Spreadsheets loaded:\n\n"+str(json.dumps(dfdict,sort_keys=False,indent=4))+"\n\n\n")

    return dfdict



def get_no_of_cols(df):
 #   print("df=\n",df)
    title1=np.unique(df[0:1].to_numpy())[1:]
    row1_len=len(title1)
    
    title2=np.unique(df[1:2].to_numpy())[1:]
    row2_len=len(title2)
    
    title3=np.unique(df[2:3].to_numpy())[1:]
    row3_len=len(title3)

 #   no_of_cols=row1_len*row2_len*row3_len

##    title1=list(map(set,df[0:1].items))
##    title2=list(map(set,df[1:2].values))
##    title3=list(map(set,df[2:3].values))

   # title1=title1[0][1]

#    print("title1=",title1)
#    print("title2=",title2)
#    print("title3=",title3)
#    print("r1=",row1_len,"r2=",row2_len,"r3=",row3_len,"no of cols=",no_of_cols)

    return title1[0],row1_len,row2_len,row3_len


def create_save_filename(market_no,elem_no,query_no,df_flag):
    if df_flag:
        s="IRI_df1_m"+str(market_no).zfill(4)+"_e"+str(elem_no).zfill(4)+"_q"+str(query_no).zfill(4)+".pkl"
    else:    
        s="IRI_df0_m"+str(market_no).zfill(4)+"_e"+str(elem_no).zfill(4)+"_q"+str(query_no).zfill(4)+".pkl"
    print("save filename=",s)
    return s

def get_load_filename_details(filename):
  #  print("6",filename[6])
  #  print("8:13",filename[8:13])
  #  print("14:19",filename[14:19])
  #  print("20:25",filename[20:25])

   # print("4:11",filename[4:11])
    print("load filename=",filename)
    return filename[6],filename[8:13],filename[14:19],filename[20:25]





def save_df(df,filename):
    df.to_pickle(filename)
    return


def load_df(filename):
    return pd.read_pickle(filename)


def save_df_dict(finished_df_dict):
    query_no=6
    savenames=[]

    for key in finished_df_dict.keys():
        savelist=finished_df_dict[key]
        i=0
        for elem in savelist:
            if isinstance(finished_df_dict[key][i], pd.DataFrame):
                savename=create_save_filename(key,i,query_no,True)
                save_df(finished_df_dict[key][i],savename)
            else:
                savename=create_save_filename(key,i,query_no,False)
                with open(savename, 'wb') as f2:
                    pickle.dump(finished_df_dict[key][i], f2)
            savenames.append(savename)
            i+=1

   # del finished_df_dict
    with open(cfg.pklsave, 'wb') as f2:
        pickle.dump(savenames, f2)

    print("\n\nsavenames=",savenames,"pickled to",cfg.pklsave,"\n\n df dict pickled\n\n")

    return 


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
   


def calculate_columns(X_df_dict,colnames,startdate,finishdate,f):

 #   df_list=[]
    tempcolsdict=dict({})
    tempcolnamesdict=dict({})

    df_colnames_dict=dict({})
    finished_df_dict=dict({})

    for key in X_df_dict:
        X_df_dict[key][0] = X_df_dict[key][0].set_index(pd.DatetimeIndex(pd.to_datetime(X_df_dict[key][0][cfg.column_zero_name], format="%Y/%m/%d %H:%M:%S",infer_datetime_format=True)))

        X_df_dict[key][0].drop(columns=[cfg.column_zero_name],inplace=True)

        X_df_dict[key][0].dropna(inplace=True)

        X_df_dict[key][0].sort_index(axis=0,ascending=[True],inplace=True)

        start = X_df_dict[key][0].index.searchsorted(startdate)    #dt.datetime(2013, 1, 2))
        finish= X_df_dict[key][0].index.searchsorted(finishdate)
        X_df_dict[key][0]=X_df_dict[key][0].iloc[start:finish]


##        print("Imported into pandas=\n",cfg.infilename,"shape:",X_df_dict[key].shape,"\n\n")    #head(10))
##
##        f.write("X_df=\n"+X_df_dict[key].to_string()+"\n\n")
##
##        f.write("Imported into pandas=\n"+str(cfg.infilename)+" shape:"+str(X_df_dict[key].shape)+"\n\n")    #head(10))


    
    


     #   print(X_df_dict) 

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


 #   key=1 # coles  (0 is WW)

##
##    print("X_df_dict[0]=\n",X_df_dict[0])
##    print("X_df_dict[1]=\n",X_df_dict[1])
##    print("colnames[0]=\n",colnames[0])
##    print("colnames[1]=\n",colnames[1])
##    

    # Build dafaframes for calculations
    ############################################################
    # 0) total units vs mainstream and premium in WW


##    tempcolsdict.setdefault(key, [])
##    tempcolnamesdict.setdefault(key,[])

####################################################

    key=0
    queryno=0
    

    tempcols,tempcolnames=temp_columns_by_col_no(X_df_dict[key][0],colnames[key],queryno,["1","15","29"])

    tempcolsdict.setdefault(key, [])
    tempcolnamesdict.setdefault(key,[])

    tempcolsdict[key].append(tempcols)
    tempcolnamesdict[key].append(tempcolnames)
    tempcolsdict[key].append(X_df_dict[key][1:5])

       
    #tempcolsdict.update({key:tempcols})
 #   tempcolnamesdict.update({key:tempcolnames})

    #df_list.append([tempcols])
   # print("\ndf colnames links=\n",key,":",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(key)+":"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")
 #   print("0tempcolsdict=\n",tempcolsdict)


    ######################
     #   1)Baseline and incr vs total category in coles
     
    key=1
    queryno=1
    tempcols,tempcolnames=temp_columns_by_col_no(X_df_dict[key][0],colnames[key],queryno,["48","62","44","58","54","55","68","69"])
    #df_list.append([tempcols])
    tempcolsdict.setdefault(key, [])
    tempcolnamesdict.setdefault(key,[])

    tempcolsdict[key].append(tempcols)
    tempcolnamesdict[key].append(tempcolnames)
    tempcolsdict[key].append(X_df_dict[key][1:5])


##    tempcolsdict.update({key:tempcols})
##    tempcolnamesdict.update({key:tempcolnames})

##    finished_df_dict.update({key : tempcols})
##    df_colnames_dict.update({key : tempcolnames})
   # print(tempcols)
  #  print("\ndf colnames links=\n",key,":",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(key)+":"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")


#    print("1tempcolsdict=\n",tempcolsdict)

    ######################
     #  2) Baseline and incr vs premium segment   

    key=1
    queryno=2
    tempcols,tempcolnames=temp_columns_by_col_no(X_df_dict[key][0],colnames[key],queryno,["43","47","57","61"])
    #df_list.append([tempcols])
    tempcolsdict.setdefault(key, [])
    tempcolnamesdict.setdefault(key,[])

    tempcolsdict[key].append(tempcols)
    tempcolnamesdict[key].append(tempcolnames)
    tempcolsdict[key].append(X_df_dict[key][1:5])

    
##    tempcolsdict.update({key:tempcols})
##    tempcolnamesdict.update({key:tempcolnames})

##    finished_df_dict.update({key : tempcols})
##    df_colnames_dict.update({key : tempcolnames})
  #  print(tempcols)
  #  print("\ndf colnames links=\n",key,":",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(key)+":"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")

 #   print("2tempcolsdict=\n",tempcolsdict)


    ######################
     #  3) sd vs BB strawberry jam 

    key=0
    queryno=3
    
    tempcols,tempcolnames=temp_columns_by_col_no(X_df_dict[key][0],colnames[key],queryno,["71","72","85","86","99","100"])
    
    tempcolsdict.setdefault(key, [])
    tempcolnamesdict.setdefault(key,[])

    tempcolsdict[key].append(tempcols)
    tempcolnamesdict[key].append(tempcolnames)
    tempcolsdict[key].append(X_df_dict[key][1:5])

    
##    tempcolsdict.update({key:tempcols})
##    tempcolnamesdict.update({key:tempcolnames})
##    finished_df_dict.update({key : tempcols})
##    df_colnames_dict.update({key : tempcolnames})

##    finished_df_dict.update({key : tempcols})
##    df_colnames_dict.update({key : tempcolnames})
  #  print(tempcols)
  #  print("\ndf colnames links=\n",key,":",json.dumps(tempcolnames,sort_keys=False,indent=4))
    f.write("\ndf Colnames links=\n"+str(key)+":"+str(json.dumps(tempcolnames,sort_keys=False,indent=4))+"\n\n")



#    print("3tempcolsdict=\n",tempcolsdict)

###########################################


    finished_df_dict.update(tempcolsdict)
    df_colnames_dict.update(tempcolnamesdict)



####################################################


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
##    df=df_list[key][-1]
##    df["smoothed_total_units"]=df["1"].rolling("50d",min_periods=3).mean()   # centre=True
##    df["smoothed_premium_units"]=df["29"].rolling("50d",min_periods=3).mean()   # centre=True
##
##  #  df["premium_percent"]=df["29"]/df["1"]
##    df["smoothed_premium_percent"]=df["smoothed_premium_units"]/df["smoothed_total_units"]
    
    #print(df)
  #  finished_df_dict.update({key:df_list})

    return finished_df_dict,df_colnames_dict          


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



##############################################################################################3333

def main():
  #  pd.set_option('display.expand_frame_repr', False)
  #  pd.set_option('display.max_rows', None)
    f=open(cfg.logfile,"w")
    print("\n\n")
    dfdict=load_IRI_xlsx(f)

  #  print("dfdict=\n",dfdict)

#################################################3
# create a list of dictionaries "colnames" of column names indexed by column number integer as a string

    row1_title=[]
    row1_count=[]
    unique_features_count=[]
    unique_measures_count=[]
    no_of_cols=[]
 #   colnamesrow=[]
 #   coltworow=[]
    colnames=[]
    coltwo=[]
    for key in dfdict.keys():
        row1_title.append(dfdict[key][1])
        row1_count.append(dfdict[key][2])
        unique_features_count.append(dfdict[key][3])
        unique_measures_count.append(dfdict[key][4])
      #  print("df deailts",row1_title[-1],row1_count[-1],unique_features_count[-1],unique_measures_count[-1])
        no_of_cols.append(row1_count[-1]*unique_features_count[-1]*unique_measures_count[-1])
        
        colnames.append(dfdict[key][0][0:3].to_dict("list"))  #other options : series records split index dict list
        coltwo.append(dfdict[key][0][1:2].to_dict("list"))

  #      print("colnamesrow=\n",colnamesrow)
  #      print("coltworow=\n",coltworow)            


        for col in range(no_of_cols[-1]+1):
           colnames[-1][str(col)][0]=row1_title[-1]

 #   print(colnames["0"][2])
        colnames[-1]["0"][1]=cfg.column_zero_name
        colnames[-1]["0"][2]=cfg.column_zero_name

        coltwoval=coltwo[-1]["1"][0]
  #  print("coltwoval=",coltwoval)
        for col in range(2,no_of_cols[-1]+1):
            if colnames[-1][str(col)][1]=="":
                colnames[-1][str(col)][1]=coltwoval
            else:
                coltwoval=coltwo[-1][str(col)][0]



        #colnames.append([colnamesrow])
        #coltwo.append([coltworow])
        
  #  print("colnames=\n",colnames)
  #  print("coltwo=\n",coltwo)            

        
   # print(colnames)
  #  print("\nColumn names=\n",json.dumps(colnames,sort_keys=False,indent=4))
    f.write("\nColumn names=\n"+str(json.dumps(colnames,sort_keys=False,indent=4))+"\n\n")

##################################################################
# create a dict grouping the common measures as values under integer index
    coldictlist=[]
    for key in dfdict.keys():
        coldict=dict()
        for col in range(1,unique_measures_count[key]+1):
            indexes=[]
            for u in range(unique_features_count[key]):
                indexes.append(str(col+u*unique_measures_count[key]))    #*unique_features_count) 
     #       print(indexes)
            coldict.update({str(col):[colnames[key][str(col)][2],indexes]})    #str(col+1*unique_measures_count),str(col+2*unique_measures_count)]})
        coldictlist.append(coldict)
##    print("\nColumn links=\n",json.dumps(coldictlist,sort_keys=False,indent=4))
##    f.write("\nColumn links=\n"+json.dumps(coldictlist,sort_keys=False,indent=4)+"\n\n")
    
################################################################33
# turn the raw excel import into a dataframe we can use
    X_df_dict=dict({})
    i=0
    for key in dfdict.keys():        
        X_df_dict.update({i:[dfdict[key][0].iloc[3:,:no_of_cols[key]+1],dfdict[key][1],dfdict[key][2],dfdict[key][3],dfdict[key][4]]})
        i+=1
        
    del dfdict  # clear memory 


#    print("X_df_dict=\n",X_df_dict)
    
##    for key in X_df_dict.keys():        
##        print("Imported into pandas=\n",X_df_dict[key],"\n\ncolumns=\n",X_df_dict[key].columns,"shape=\n",X_df_dict[key].shape)    #head(10))

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

#######################################################################

    #finished_df_dict=dict({})
    finished_df_dict,df_colnames_dict=calculate_columns(X_df_dict,colnames,startdate,finishdate,f)

    print("finished_df_dict=\n",finished_df_dict)
    




#################################################################
    #  save and load df in pickle


    save_df_dict(finished_df_dict)    

    del finished_df_dict

####################################################

    df_dict=load_df_dict(cfg.pklsave)
 





#########################################################################3
 #  print("dfdict=\n",dfdict)

 #   print("coldictlist=\n",coldictlist)

    print("\n\n\nnew finished df dict=\n",df_dict,"\n\n")
    
   # print("old finished df dict=\n",finished_df_dict,"\n\n")
   # print("\nfinished df dict=\n",json.dumps(finished_df_dict,sort_keys=False,indent=4))
  #  f.write("\nfinished df dict=\n"+pd.DataFrame.from_dict(finished_df_dict)+"\n\n")
   # f.write("\nfinished df dict=\n"+json.dumps(finished_df_dict,sort_keys=False,indent=4)+"\n\n")





 #   print("df_colnames_dict=\n",df_colnames_dict,"\n\n")
 #   print("\ndf colnames dict=\n",json.dumps(df_colnames_dict,sort_keys=False,indent=4))
 #   f.write("\ndf colnames dict=\n"+df_colnames_dict+"\n\n")
  #  f.write("\ndf colnames dict=\n"+json.dumps(df_colnames_dict,sort_keys=False,indent=4)+"\n\n")



    f.close()
    return 



if __name__ == '__main__':
    main()

