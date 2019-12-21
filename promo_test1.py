


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





def read_excel(filename,rows):
    xls = pd.read_excel(filename,sheet_name="Sheet1",header=0,index_col=None)    #'salestransslice1.xlsx')
    if rows==-1:
        return xls   #.parse(xls.sheet_names[0])
    else:        
        return xls.head(rows)


def main():
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    infilename="IRI_coles_jams_v7.xlsx"
    print(infilename)
##    f=open(cfg.outfile,"w")
##
##    print("\n\nBrand strength Index - By Anthony Paech 14/12/19\n")
###    print("Loading",sys.argv[1],"into pandas for processing.....")
##    print("Loading",cfg.infilename,"into pandas for processing.....")
##
##    print("Results Reported to",cfg.outfile)
##
##    f.write("\n\nBrand Strength Index - By Anthony Paech 14/12/19\n\n")
## #   f.write("Loading "+sys.argv[1]+" into pandas for processing.....\n")
##    f.write("Loading "+cfg.infilename+" into pandas for processing.....\n")
##
##    f.write("Results Reported to "+cfg.outfile+"\n")

##    df=read_excel(sys.argv[1],-1)  # -1 means all rows
##    if df.empty:
##        print(sys.argv[1],"Not found.")
##        sys.exit()
##
    df=read_excel(infilename,-1)  # -1 means all rows
    if df.empty:
        print(infilename,"Not found. Check brand_strength_cfg.py file")
        sys.exit()
##
##
##    X_df=df.iloc[:,:37]
##      
##    
##  #  del df  # clear memory 
##
##    print("Imported into pandas=\n",X_df.columns,"\n",X_df.shape)    #head(10))


    
##################################################33
##   # Remove extranous fields
##    b4=X_df.shape[1]
##    print("Remove columns not needed.")
###    X_df.drop(columns=["cat","code","costval","glset","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"],inplace=True)
##    X_df.drop(columns=cfg.excludecols,inplace=True)
##
##    print("Columns deleted:",b4-X_df.shape[1]) 
##
##    print("Delete rows for productgroup <10 or >15.")
##    b4=X_df.shape[0]
##    X_df.drop(X_df[(X_df["productgroup"]<10) | (X_df["productgroup"]>15)].index,inplace=True)
##    print("Rows deleted:",b4-X_df.shape[0])
##    
##
##    print("Prepare data....")
##    X_df.drop
    X_df=df.iloc[:,:45]
      
    
    del df  # clear memory 

    print("Imported into pandas=\n",X_df.columns,"\n",X_df.shape)    #head(10))


    
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
    X_df["date"]=pd.to_datetime(X_df["coles_scan_week"])
    X_df['date_encode'] = X_df['date'].map(dt.datetime.toordinal).astype(int)
  #  X_df['day_delta'] = (X_df.date-X_df.date.min()).dt.days.astype(int)
    X_df['day_delta'] = (X_df.date.max()-X_df.date).dt.days.astype(int)

    X_date=X_df["date"].to_numpy()

    X_df.drop(columns=["coles_scan_week","date_encode","bb_price","sd_price","c_price","bm_price","bb_promo_off","sd_promo_off","c_promo_off","bm_promo_off"],inplace=True)
 
    #print(sales)
    X_df.dropna(inplace=True)
 #   Xc_df.dropna(inplace=True)

################################################3


    X_df["bb_promo_disc"]=X_df["bb_promo_disc"].round(0)
    X_df["sd_promo_disc"]=X_df["sd_promo_disc"].round(0)
    X_df["c_promo_disc"]=X_df["c_promo_disc"].round(0)
    X_df["bm_promo_disc"]=X_df["bm_promo_disc"].round(0)

    # calculated totals and reorder remaining columns
##    X_df["bb_upw_baseline"]=X_df["bb_upspw_baseline"]*X_df["bb_dd"]
##    X_df["bb_upw_total"]=(X_df["bb_upspw_baseline"]+X_df["bb_upspw_incremental"])*X_df["bb_dd"]
##    X_df["sd_upw_baseline"]=X_df["sd_upspw_baseline"]*X_df["sd_dd"]
##    X_df["sd_upw_total"]=(X_df["sd_upspw_baseline"]+X_df["sd_upspw_incremental"])*X_df["sd_dd"]
##    X_df["c_upw_baseline"]=X_df["c_upspw_baseline"]*X_df["c_dd"]
##    X_df["c_upw_total"]=(X_df["c_upspw_baseline"]+X_df["c_upspw_incremental"])*X_df["c_dd"]
##    X_df["bm_upw_baseline"]=X_df["bm_upspw_baseline"]*X_df["bm_dd"]
##    X_df["bm_upw_total"]=(X_df["bm_upspw_baseline"]+X_df["bm_upspw_incremental"])*X_df["bm_dd"]

    X_df["promo_sum"]=X_df["bb_on_promo"]+X_df["sd_on_promo"]+X_df["c_on_promo"]+X_df["bm_on_promo"]

###################################
# rolling averages for baseline unit sales

# hang_over_weeks_ahead
    hang_over_weeks_ahead=2

    X_df.sort_values(by=["date"],axis=0,ascending=[True],inplace=True)


    X_df["bb_ave_baseline_units"]=X_df["bb_baseline_units"].rolling(6,min_periods=3,center=True).mean()
    X_df["sd_ave_baseline_units"]=X_df["sd_baseline_units"].rolling(6,min_periods=3,center=True).mean()
    X_df["c_ave_baseline_units"]=X_df["c_baseline_units"].rolling(6,min_periods=3,center=True).mean()
    X_df["bm_ave_baseline_units"]=X_df["bm_baseline_units"].rolling(6,min_periods=3,center=True).mean()

##    X_df["bb_baseline_units_next_3_weeks"]=X_df["bb_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
##    X_df["sd_baseline_units_next_3_weeks"]=X_df["sd_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
##    X_df["c_baseline_units_next_3_weeks"]=X_df["c_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()
##    X_df["bm_baseline_units_next_3_weeks"]=X_df["bm_baseline_units"].shift(periods=1).rolling(3,min_periods=1).mean()

    X_df["bb_baseline_units_next_week"]=X_df["bb_baseline_units"].shift(periods=-1).rolling(hang_over_weeks_ahead,min_periods=1).mean()#.shift(periods=-1)
    X_df["sd_baseline_units_next_week"]=X_df["sd_baseline_units"].shift(periods=-1).rolling(hang_over_weeks_ahead,min_periods=1).mean()#.shift(periods=-1)
    X_df["c_baseline_units_next_week"]=X_df["c_baseline_units"].shift(periods=-1).rolling(hang_over_weeks_ahead,min_periods=1).mean()#.shift(periods=-1)
    X_df["bm_baseline_units_next_week"]=X_df["bm_baseline_units"].shift(periods=-1).rolling(hang_over_weeks_ahead,min_periods=1).mean()#.shift(periods=-1)

##    X_df["sd_baseline_units_next_3_weeks"]=X_df["sd_baseline_units"].shift(periods=-1).rolling(3,min_periods=1).mean()
##    X_df["c_baseline_units_next_3_weeks"]=X_df["c_baseline_units"].shift(periods=-1).rolling(3,min_periods=1).mean()
##    X_df["bm_baseline_units_next_3_weeks"]=X_df["bm_baseline_units"].shift(periods=-1).rolling(3,min_periods=1).mean()


    X_df["bb_baseline_units_hangover"]=(X_df["bb_ave_baseline_units"]-X_df["bb_baseline_units_next_week"])*hang_over_weeks_ahead
    X_df["sd_baseline_units_hangover"]=(X_df["sd_ave_baseline_units"]-X_df["sd_baseline_units_next_week"])*hang_over_weeks_ahead
    X_df["c_baseline_units_hangover"]=(X_df["c_ave_baseline_units"]-X_df["c_baseline_units_next_week"])*hang_over_weeks_ahead
    X_df["bm_baseline_units_hangover"]=(X_df["bm_ave_baseline_units"]-X_df["bm_baseline_units_next_week"])*hang_over_weeks_ahead

 #   X_df["bb_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["bb_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)
 #   X_df["sd_baseline_units_hangover"]=X_df["sd_baseline_units"]-X_df["sd_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)
##    X_df["c_baseline_units_hangover"]=X_df["c_baseline_units"]-X_df["c_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)
##    X_df["bm_baseline_units_hangover"]=X_df["bm_baseline_units"]-X_df["bm_baseline_units"].shift(periods=-1).rolling(2,min_periods=1).mean().shift(periods=-1)


 #   X_df["bb_baseline_units_3wkave"]=X_df["bb_baseline_units_next_3_weeks"].rolling(3,min_periods=1).mean().shift(periods=-2)


##    X_df["bb_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["bb_baseline_units_next_3_weeks"]
##    X_df["sd_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["sd_baseline_units_next_3_weeks"]
##    X_df["c_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["c_baseline_units_next_3_weeks"]
##    X_df["bm_baseline_units_hangover"]=X_df["bb_baseline_units"]-X_df["bm_baseline_units_next_3_weeks"]
##


    X_df["bb_baseline_units_var"]=X_df["bb_baseline_units"]-X_df["bb_ave_baseline_units"]
    X_df["sd_baseline_units_var"]=X_df["sd_baseline_units"]-X_df["sd_ave_baseline_units"]
    X_df["c_baseline_units_var"]=X_df["c_baseline_units"]-X_df["c_ave_baseline_units"]
    X_df["bm_baseline_units_var"]=X_df["bm_baseline_units"]-X_df["bm_ave_baseline_units"]


    X_df["bb_gains_from_others"]=-(X_df["sd_baseline_units_var"]+X_df["c_baseline_units_var"]+X_df["bm_baseline_units_var"])
    X_df["sd_gains_from_others"]=-(X_df["bb_baseline_units_var"]+X_df["bm_baseline_units_var"])
    X_df["c_gains_from_others"]=-(X_df["bb_baseline_units_var"]+X_df["sd_baseline_units_var"]+X_df["bm_baseline_units_var"])
    X_df["bm_gains_from_others"]=-(X_df["bb_baseline_units_var"]+X_df["sd_baseline_units_var"]+X_df["c_baseline_units_var"])


   # print("\nBB only X_df=\n",X_df[["bb_baseline_units","bb_baseline_units_var","sd_baseline_units","sd_ave_baseline_units","sd_baseline_units_var","c_baseline_units","c_ave_baseline_units","c_baseline_units_var","bm_baseline_units","bm_ave_baseline_units","bm_baseline_units_var"]])

 #   print("\nBB only X_df=\n",X_df[["bb_baseline_units","bb_ave_baseline_units","bb_baseline_units_var","bb_gains_from_others","sd_baseline_units","sd_baseline_units_var","sd_gains_from_others","c_baseline_units","c_baseline_units_var","c_gains_from_others","bm_baseline_units","bm_baseline_units_var","bm_gains_from_others"]])



################################################333

    X_df=X_df[["bb_dd","date","bb_on_promo","sd_on_promo","c_on_promo","bm_on_promo","promo_sum","day_delta","bb_upspw_baseline","sd_upspw_baseline","c_upspw_baseline","bm_upspw_baseline","bb_upspw_incremental","sd_upspw_incremental","c_upspw_incremental","bm_upspw_incremental","bb_dd","sd_dd","c_dd","bm_dd","bb_promo_disc","sd_promo_disc","c_promo_disc","bm_promo_disc","bb_total_units","sd_total_units","c_total_units","bm_total_units","bb_baseline_units","sd_baseline_units","c_baseline_units","bm_baseline_units","bb_ave_baseline_units","sd_ave_baseline_units","c_ave_baseline_units","bm_ave_baseline_units","bb_baseline_units_var","sd_baseline_units_var","c_baseline_units_var","bm_baseline_units_var","bb_gains_from_others","sd_gains_from_others","c_gains_from_others","bm_gains_from_others","bb_incremental_units","sd_incremental_units","c_incremental_units","bm_incremental_units","bb_baseline_units_next_week","sd_baseline_units_next_week","c_baseline_units_next_week","bm_baseline_units_next_week","bb_baseline_units_hangover","sd_baseline_units_hangover","c_baseline_units_hangover","bm_baseline_units_hangover"]]

#   X_df.sort_values(by=["coles_scan_week","promo_sum","bb_on_promo","sd_on_promo","c_on_promo","bm_on_promo"],axis=0,ascending=[True,True,True,True,True,True],inplace=True)


# sd promo only with clear air
    mask=((X_df["sd_on_promo"]==1) & (X_df["sd_incremental_units"]>=1.0) & (X_df["promo_sum"]==1))
    #mask=((X_df["sd_incremental_units"]>=1.0) & (X_df["promo_sum"]==1))

    X_bb=X_df[mask]


    X_df.sort_values(by=["date"],axis=0,ascending=[True],inplace=True)

 #   print(X_bb.columns)
 #   print("\nSD only X_df=\n",X_bb[["bb_baseline_units","bb_baseline_units_var","bb_gains_from_others","sd_baseline_units","sd_baseline_units_var","sd_gains_from_others","c_baseline_units","c_baseline_units_var","c_gains_from_others","bm_baseline_units","bm_baseline_units_var","bm_gains_from_others",]])
    print("\nSD only X_df=\n",X_bb[["date","sd_promo_disc","bb_baseline_units","bb_baseline_units_next_week","sd_baseline_units","sd_ave_baseline_units","sd_baseline_units_next_week","sd_incremental_units","sd_baseline_units_hangover"]])

### bb promo with any other
##    mask=((X_df["bb_on_promo"]==1) & (X_df["promo_sum"]==2))
##    X_bb=X_df[mask]
##    print(X_bb.columns)
##    print("\nBB with one other X_df=\n",X_bb)
##
##
### sd promo only with clear air
##    mask=((X_df["sd_on_promo"]==1) & (X_df["promo_sum"]==1))
##    X_sd=X_df[mask]
##    print(X_sd.columns)
##    print("\nSd only X_df=\n",X_sd)
##
##
### sd promo with one other
##    mask=((X_df["sd_on_promo"]==1) & (X_df["promo_sum"]==2))
##    X_sd=X_df[mask]
##    print(X_sd.columns)
##    print("\nSd with one other X_df=\n",X_sd)
##
##
### c promo only with clear air
##    mask=((X_df["c_on_promo"]==1) & (X_df["promo_sum"]==1))
##    X_c=X_df[mask]
##    print(X_c.columns)
##    print("\nc only X_df=\n",X_c)
##
### c promo with one other
##    mask=((X_df["c_on_promo"]==1) & (X_df["promo_sum"]==2))
##    X_c=X_df[mask]
##    print(X_c.columns)
##    print("\nc with 1 other X_df=\n",X_c)
##
##
### bm promo only with clear air
##    mask=((X_df["bm_on_promo"]==1) & (X_df["promo_sum"]==1))
##    X_bm=X_df[mask]
##    print(X_bm.columns)
##    print("\nBM only X_df=\n",X_bm)
##
### bm promo with one other
##    mask=((X_df["bm_on_promo"]==1) & (X_df["promo_sum"]==2))
##    X_bm=X_df[mask]
##    print(X_bm.columns)
##    print("\nBM with one other X_df=\n",X_bm)
##
##









    return 



if __name__ == '__main__':
    main()

