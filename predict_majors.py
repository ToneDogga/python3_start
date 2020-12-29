#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 07:58:39 2020

@author: tonedogga
"""

import os
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


import glob
from pathlib import Path

import datetime as dt
from datetime import datetime
from datetime import date
from datetime import timedelta
import calendar
import xlsxwriter
import matplotlib.pyplot as plt


from matplotlib import pyplot, dates
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show

#import matplotlib.cm as cm
#import seaborn as sns
#import matplotlib.ticker as ticker
#from matplotlib.ticker import ScalarFormatter
#from matplotlib.ticker import FormatStrFormatter
#from matplotlib.ticker import StrMethodFormatter

#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
#from pandas.plotting import scatter_matrix

# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# tf.autograph.set_verbosity(0, False)
# import subprocess as sp

# from tensorflow import keras
# #from keras import backend as K


import dash2_dict as dd2



class predict_majors(object):
    def __init__(self):
        pass
    
 
             
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
    
 
    
  
    def _load_pickle(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            return pd.read_pickle(save_dir+savefile)
        else:
            print("load sales_df error.")
            return
      
        
      
    def load_data(self):
        scan_df=self._load_pickle(dd2.dash2_dict['scan']['save_dir'],dd2.dash2_dict['scan']['savefile'])  
        sales_df=self._load_pickle(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['raw_savefile']).copy()
     #  sales_df['specialpricecat']=sales_df['specialpricecat'].astype(str)
        sales_df.sort_index(axis=0,ascending=True,inplace=True)
        return sales_df,scan_df
  
    
#  can you predict orders from coles and woolworths based on weekly scan data?
#  the scan data for both drops on thursdays at 8am for the previous week
# this data is from all of wednesday to all of Tuesday
# so for example
# 
#  Wednesday 17/11 - tuesday night 24/11  - 4 weeks ago  
#  Wednesday 25/11 - tuesday night 1/12  -3 weeks
#   wed 2/12 - tuesday 8/12 -2 weeks
#   wed 9/12 - tuesday 15/12  data availble morning thursday 17/12  -1 week
#   wed 16/12 - tuesday 22/12  data available morning thursday 24/12
#
# this means you need to slice the sales df (this is the invoiced orders) for both coles (spc=12) and woolworths (spc=10) into resampled chunks on 1 week
# starting W-WED with an offset of -7 days
#
# so, coles and woolworths orders placed (y) wed 25/11 - tuesday 1/12 
# should match scan data (X) for wed 16/12- tuesday 22/12
# and also for every week prior
#
# present this visually to check
#
#  Critically, the scan data X for the week wed 16/12 - tuesday 22/12 
#  should predict the orders placed y by coles and woolworths on wednesday 13/1/21 - tuesday 19/1/21 
# this is the basis of the prediction system
# 
#  on thursday use a simple random forest regressor trained on all scan data (X) (including the scan data for this week wed - tues)  aligned against actual invoiced sales offset forward by 3 weeks (y)
# there should be a correlation
#  so X is the scan data, y is the target data, the orders
# the trained model should then be able to predict the next y (order) based on the new X (scan data)
# the next y will be the orders placed in 3 weeks
# so for scan data for wed 16/12 - tuesday 22/12, we are predicting orders for the week Wed 13/1/21-Tuesday 19/1/21
#
#
#
#   *  [there could be a overlap so we could approach it as a moving average of 3 weeks of orders including a week either side
#
 # then offset the invoiced sales forward by 3 weeks.   I think this is the delay between order placement and arrival and purchase in store.  this needs to be tested
#
#  either way, scan data for this week should predict the orders in 3 weeks time.  We have enough time to manufacture it.]
#
#
 

    def chunk_scan(self,scan_df):
      #  print("scan_df=\n",scan_df)
      #  scan_df.index.level(0).astype(np.int32)
        c_scan=scan_df.xs("71",level='plottype3',axis='index',drop_level=False)
 
        c_scan=c_scan.droplevel([0,2,5,6,7,8,9,12])

        c_scan['type']="scanned last week"
        c_scan['twe']=0
        c_scan['pwe']=0
     #   c_scan['rmse']=0
    #    print("cs=\n",c_scan)
     #   c_scan.columns.set_names('type', level=0,inplace=True)
        c_scan.set_index(["type",'twe','pwe'],append=True,inplace=True)
   #     print("c_scan=\n",c_scan)
   #    nr=nr.set_index(['product','type','retailer','sortorder','colname','productgroup','twe','pwe'])
        product_list=[(p,str(r),t,s,c,pg,twe,pwe) for p,r,t,s,c,pg,twe,pwe in zip(c_scan.index.get_level_values("product"),c_scan.index.get_level_values("retailer"),c_scan.index.get_level_values("type"),c_scan.index.get_level_values("sortorder"),c_scan.index.get_level_values("colname"),c_scan.index.get_level_values("productgroup"),c_scan.index.get_level_values("twe"),c_scan.index.get_level_values("pwe"))]
      #  product_group_list=[(str(r),str(pg)) for r,pg in zip(c_scan.index.get_level_values("retailer"),c_scan.index.get_level_values("productgroup"))]

    #     c_scan*=1000
        return c_scan.T,product_list   #,product_group_list


    def chunk_orders(self,sales_df,product_list,invoiced_sales_smoothed_over_weeks,offset):
       # prods=list(set([p[0] for p in product_list]))
       # rets=list(set([p[1] for p in product_list]))
       # print("prodcs and rets",prods,rets)
       # orders_df=sales_df[(sales_df['specialpricecat'].isin(rets)) & (sales_df['product'].isin(prods))].copy()
     #   print(sales_df)
        sales_df['specialpricecat']=sales_df['specialpricecat'].astype(int).astype(str)
     #   print(sales_df)
        orders_df=pd.DataFrame([])
        for p in product_list:
        #    new_df=sales_df[(sales_df['specialpricecat']==p[0]) & (sales_df['product']==p[1])].copy()
       #     print(p,"orders_df=\n",orders_df.shape)
            orders_df=pd.concat([orders_df,sales_df[(sales_df['specialpricecat']==p[1]) & (sales_df['product']==p[0])].copy()],axis=0)
      #  orders_df['specialpricecat']=orders_df['specialpricecat'].astype(np.int32)
       # print("orderdf=\n",orders_df)
 
    #    for pg in product_group_list:
        #    new_df=sales_df[(sales_df['specialpricecat']==p[0]) & (sales_df['product']==p[1])].copy()
       #     print(p,"orders_df=\n",orders_df.shape)
     #       orders_df=pd.concat([orders_df,sales_df[(sales_df['specialpricecat']==pg[0]) & (sales_df['productgroup']==pg[1])].copy()],axis=0)



        weekly_sdf=orders_df.groupby(['specialpricecat','product','productgroup',pd.Grouper(key='date', freq='W-TUE',label='right',closed='right',offset="7D")],as_index=True).agg({"qty":"sum"})
            #    weekly_sdf.reset_index(inplace=True)
      #
     #   print("2=\n",weekly_sdf)
        weekly_sdf.index = weekly_sdf.index.swaplevel(0, 3)
        weekly_sdf.index = weekly_sdf.index.swaplevel(1, 3)
        
        
        weekly_sdf.index.set_names(level='specialpricecat',names="retailer",inplace=True)
       # print(weekly_sdf.index.names[1])
    #    print("2",weekly_sdf)
        weekly_sdf = weekly_sdf.unstack(['retailer','productgroup','product'])  #.stack('date')

        weekly_sdf=weekly_sdf.T
        
        
        
     #   print("3",weekly_sdf)
        weekly_sdf=weekly_sdf.dropna(axis=0,how='all',thresh=6)
     #   weekly_sdf=weekly_sdf[(weekly_sdf.sum(axis=1)>50000)]   # units that make the product a national prduct
    
 

  #      print("4",weekly_sdf)
        weekly_sdf=weekly_sdf.T
        weekly_sdf.sort_index(axis=1,level=['retailer','productgroup','product'],inplace=True)
        colname=weekly_sdf.columns.get_level_values('product')
 #       colname2=list(weekly_sdf.columns.get_level_values('retailer').astype(np.int32).astype(str))  #.astype(np.int32).astype(str)

 
        weekly_sdf=weekly_sdf.T
      #  print("4",weekly_sdf)
     #  weekly_sdf.index.levels[0]=colname2  #.astype(str)   #weekly_sdf.index.set_levels(colname2,level='retailer')   #,level=0,inplace=True)

        weekly_sdf['sortorder']=np.arange(1,weekly_sdf.shape[0]+1)
        weekly_sdf['colname']=colname
        weekly_sdf['type']="invoiced ("+str(offset)+"wk smth and "+str(invoiced_sales_smoothed_over_weeks)+"wks right offset)"
        weekly_sdf['twe']=0
        weekly_sdf['pwe']=0
     #   weekly_sdf['rmse']=0
         
        weekly_sdf.set_index(["sortorder",'colname','type','twe','pwe'],append=True,inplace=True)
        
    #    weekly_df.index.set_level_values('retailer')=colname2
     #   print("5",weekly_sdf)
        weekly_sdf=weekly_sdf.droplevel([0])
        weekly_sdf/=1000
  #      print("weekly_sdf=\n",weekly_sdf)
  #      print(weekly_sdf.index.names)
        return weekly_sdf.T
        
    
    
    
    
    def _smooth_orders(self,weekly_sdf,*,smth_weeks):
        weekly_sdf.fillna(0,inplace=True)
        smoothed_weekly_sdf=weekly_sdf.rolling(smth_weeks,axis=0).mean()
    #    print("so=\n",smoothed_weekly_sdf)
        
        
        return smoothed_weekly_sdf   
        


    def _shift_and_join(self,weekly_sdf,weekly_scan_df,offset):
        #  Weekly sdf is the orders and the y
        # it needs to be offset by 3 weeks forward
       weekly_sdf=weekly_sdf.T 
       sales_product_list=list(set(list(weekly_sdf.index.get_level_values('product'))))
                               
     #  weekly_sdf=weekly_sdf.T 
       weekly_scan_df=weekly_scan_df.T 
       scan_product_list=list(set(list(weekly_scan_df.index.get_level_values('product'))))
     #  weekly_scan_df=weekly_scan_df.T 
 
    #   print("sales product names=",sales_product_list)
    #   print("scan product names=",scan_product_list)
       new_weekly_sdf=weekly_sdf[weekly_sdf.index.get_level_values('product').isin(scan_product_list)].copy()
     #  print("nwn",new_weekly_sdf.T)
           
       new_weekly_sdf=new_weekly_sdf.shift(periods=offset,axis='columns')    ## periods=dd2.dash2_dict['sales']['predictions']['invoiced_sales_weeks_offset']
       
       
    #   print("nwn2",new_weekly_sdf)
 
       new_weekly_sdf=new_weekly_sdf.T 
       weekly_scan_df=weekly_scan_df.T 
       joined_df=new_weekly_sdf.join(weekly_scan_df)
   #    print("shift and join=\n",joined_df.T)
      
       joined_df.columns = joined_df.columns.swaplevel(0, 2)
       joined_df.columns = joined_df.columns.swaplevel(5, 1)
   #    joined_df.columns = joined_df.columns.swaplevel(2, 0)
       joined_df.columns = joined_df.columns.swaplevel(2, 1)
   # joined_df.columns = joined_df.columns.swaplevel(2, 5)
  
   #    print(" shift and join joined_df=\n",joined_df)
  #  print(joined_df.T)

       joined_df.sort_index(level=[0,1],axis=1,sort_remaining=False,inplace=True)
       joined_df.fillna(0,inplace=True)
    #   print("final shift and join=\n",joined_df.T)
       return joined_df.T




    def _get_X_and_y(self,orders_df,training_window_offset_left_from_end,window_length,invoiced_sales_smoothed_over_weeks,offset):
        #  lets group all together first
        # so X will be 155 rows and 66 columns
        # y will be 66 columns
        # we can also test with by productgroup onlly, eg jams, dressings, sauces
        # we can also test by the two different retailers
        #
 #       X_df=orders_df.iloc[:,-64:-12].xs("scanned last week",level='type',drop_level=False)
        X_df=orders_df.iloc[:,-(training_window_offset_left_from_end+window_length):-(training_window_offset_left_from_end)].xs("scanned last week",level='type',drop_level=False)

     #   print("X-df=\n",X_df)
        oldX=X_df.T.to_numpy()
        X=np.nan_to_num(oldX,nan=0,copy=True)
     #   X=np.swapaxes(X,1,2)
     #   print("X=\n",X,X.shape)
        y_df=orders_df.iloc[:,-(training_window_offset_left_from_end+window_length):-(training_window_offset_left_from_end)].xs("invoiced ("+str(offset)+"wk smth and "+str(invoiced_sales_smoothed_over_weeks)+"wks right offset)",level='type',drop_level=False)
    #    print("Y_df=\n",y_df)
        oldy=y_df.to_numpy()
        y=np.nan_to_num(oldy[0],nan=0,copy=True)     #     print("y=\n",y,y.shape)
      #  print("y=\n",y,y.shape)   
      #  X=X.reshape(1,-1)
        return X,y



    def _rfr_model(self,orders_df,training_window_offset_left_from_end,window_length,invoiced_sales_smoothed_over_weeks,offset):
        X,y=self._get_X_and_y(orders_df,training_window_offset_left_from_end,window_length,invoiced_sales_smoothed_over_weeks,offset)
     #   print("X=\n",X,X.shape,"y=\n",y,y.shape)
    #    print("\nFit random forest Regressor...")     
        forest_reg=RandomForestRegressor(n_estimators=300,n_jobs=-1)
        forest_reg.fit(X,y)
        return forest_reg,X



    def _predict(self,joined_df,product_list,training_window_offset_left_from_end,prediction_window_offset_left_from_end,window_length,invoiced_sales_smoothed_over_weeks,offset,predict_back_weeks):
   #     new_row_df=pd.DataFrame([])
      #  print("jpredict df=\n",joined_df)
     #   print("predict product list=",product_list)
        t_list=[]
        
        for p in product_list:
            orders_df=joined_df.xs([p[0],p[1]],level=['product',"retailer"],drop_level=False).copy()
    
        #    if orders_df.shape[0]<2:
           #    orders_df=joined_df.xs([p[1],p[0]],level=['product',"retailer"],drop_level=False).copy()
         #      print("missing data")
         #      print("p=",p,"orders_df=\n",orders_df)
            
         #   else:
            for t in range(training_window_offset_left_from_end,8,-1):
                print("prediction window",-prediction_window_offset_left_from_end,"predict:",p[4],"on training window offset=",t,"                                  ",end="\r",flush=True)
                forest_reg,X=self._rfr_model(orders_df,t,window_length,invoiced_sales_smoothed_over_weeks,offset) 
            #    X_orders_df=joined_df.xs([p[1],"scanned last week",p[0]],level=['product','type',"retailer"],drop_level=False).copy()
               # print(X,X.shape,X[-53:-1,:],X[-53:-1,:].shape)
                prediction=forest_reg.predict(X[-(prediction_window_offset_left_from_end+window_length):-(prediction_window_offset_left_from_end),:])
              #  print("concat:\n",X_orders_df,"\npred=",prediction)  #,"new_y=",new_y,new_y.shape) 
                y_orders_df=joined_df.xs([p[0],p[1],"invoiced ("+str(offset)+"wk smth and "+str(invoiced_sales_smoothed_over_weeks)+"wks right offset)"],level=['product',"retailer",'type'],drop_level=False).copy()
    
                new_row=y_orders_df.copy()
                nr=new_row.reset_index()
             #   print("nr=\n",nr)
                nr['type']="pred wk="+str(-prediction_window_offset_left_from_end)+":trn win wkstoend"+str(t)
                nr['twe']=t
                nr['pwe']=-prediction_window_offset_left_from_end
               # print("pred=",prediction)
               # print(prediction.shape,"pred[-1]=",prediction[-1])
           #     print("before nr=\n",nr)
    
                pred=np.concatenate((np.array(np.atleast_1d(np.around(prediction[-1],3))),np.full((prediction_window_offset_left_from_end-1),np.nan)))
             #   print(prediction_window_offset_left_from_end,pred.shape,"pred=",pred)
                nr.iloc[:,-prediction_window_offset_left_from_end:]=pred
             #   print(nr.iloc[2,-1]=np.nan
                nr=nr.set_index(['product','retailer','type','sortorder','colname','productgroup','twe','pwe'])
               # print("after nr=\n",nr)
                t_list.append((p[0],p[1],p[2],p[3],p[4],p[5],t,-prediction_window_offset_left_from_end))
                joined_df=pd.concat((joined_df,nr),axis=0)
          #  rmse=self._RMSE(joined_df,p,t,predict_back_weeks)
          #  print("rmse",rmse)
          #  nr['rmse']=rmse


 
                
      #  twe = joined_df['twe']
      #  joined_df.drop(labels=['twe'], axis=1,inplace = True)
      #  joined_df.insert(3, 'twe', twe)   
      #  print("t_list=",t_list)
        print("\n")
        joined_df.sort_index(level=[0,1,2],axis=0,sort_remaining=False,inplace=True)
        return t_list,joined_df
    
        
  
    def _display(self,joined_df,product_list,rmse_list,plot_output_dir,display_window_length):
        joined_df=joined_df.droplevel([3,5])
        
        print("RFR majors prediction version2:\n")
        print(joined_df.iloc[:,-4:].to_string())
        for p in product_list:
            fig, ax = pyplot.subplots()
          #  fig.autofmt_xdate()
 
            orders_df=joined_df.xs([p[1],p[0]],level=['product',"retailer"],drop_level=False).T.copy()
          #  print(orders_df)
          #  plt.locator_params(axis='x', nbins=16)
          #  plt.xticks(np.arange(0, len(x)+1, 1))
 
            orders_df.iloc[-display_window_length:-1,[0]].plot(xlabel="",ylabel="units/week",use_index=True,grid=True,fontsize=7,style="b-",ax=ax)
            orders_df.iloc[-display_window_length:,1:-1].plot(xlabel="",ylabel="units/week",use_index=True,grid=True,fontsize=7,label='_nolegend_',style="r:",ax=ax)
            orders_df.iloc[-display_window_length:-1,[-1]].plot(xlabel="",ylabel="units/week",use_index=True,grid=True,fontsize=7,style="g-",ax=ax)

            #  ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d/%m/%Y'))
            fig.autofmt_xdate()
            plt.title(str(p)+" twe="+str(p[0])+" RMSE="+str(p[1]),fontsize=9)
            plt.legend(title="",fontsize=6,loc="upper left")
        #    plt.show()
            self._save_fig(str(p)+"_rfr_prediction_v2",plot_output_dir)
            plt.close()
      
        
      
 
    def _RMSE(self,joined_df,training_list,training_window_offset_left_from_end,predict_back_weeks):
       #  print("training list=",training_list)
     #    print("joineddf=\n",joined_df)
         rmse_list=[]
         for p in training_list:
             orders_df=joined_df.xs([p[0],p[1],p[6]],level=['product',"retailer","twe"],drop_level=False).T.copy()
     
        #     print("1orders_df=\n",orders_df)   #,"\n",orders_df.T)
     
             orders_df=orders_df.sort_index(level='pwe',axis=1,ascending=False,sort_remaining=False)
    
         #    print("2orders_df=\n",orders_df)   #,"\n",orders_df.T)
          #   invoiced=orders_df.iloc[-(predict_back_weeks):-1,[0]]
          #   print(p,"invoiced=\n",invoiced,"\n\n")
             predicted=orders_df.iloc[-(predict_back_weeks+2):-1]
          #   print(p,"predicted=\n",predicted,predicted.shape,"\n\n")
             ordered=np.flipud(predicted).diagonal(offset=0)[:-1][np.newaxis]   #, index=[predicted.index, predicted.columns])
          #   ordered=np.diagonal(predicted,offset=0)[:-1][np.newaxis]   #, index=[predicted.index, predicted.columns])
    
          #   print("ordered=",ordered)
             pred=np.flipud(predicted).diagonal(offset=1)[np.newaxis]   #, index=[predicted.index, predicted.columns])
    #             pred=np.diagonal(predicted,offset=1)[np.newaxis]   #, index=[predicted.index, predicted.columns])
     
          #   print("pred=",pred)
             joined=np.concatenate((ordered,pred),axis=0)
           #  print("joined=",joined)
          #   print("joined[0,:]",joined[0,:])
          #   print("joined[1,:]",joined[1,:])
          #   print("joined0-1",joined[0]-joined[1])
            # nr=nr.set_index(['product','type','retailer','sortorder','colname','productgroup','twe','pwe'])
             rmse_list.append((p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],int(round(np.sqrt(((joined[0]-joined[1])**2).mean()),0))))
         #rmse=int(round(np.sqrt(((joined[0]-joined[1])**2).mean()),0))
              #  joined_df.loc[]df.loc[(slice(​None), slice('price')), :]
    #    print("product list",product_list)    
     #   print("rmse=",rmse)
    #     joined_df['mse']=(joined_df['stock_count']-joined_df['ideal_stock_holdings'])**2
    #     rmse=round(np.sqrt(joined_df['mse'].mean()),0)
    #  #    print(product+" RMSE=",rmse)
         return rmse_list
    
   
    def _fill_rmse(self,joined_df,rmse_list):
         joined_df['rmse']=0
         joined_df['rmse_empty']=np.nan
       #  joined_df.sort_index(['product','retailer','twe','pwe'],inplace=True)
       #  rmse_temp=[]
       #  for r in rmse_list:
       #      rmse_temp.append(r[4])

            #    print(r)
         #    joined_df.xs([r[0],r[1],r[2],r[3]],level=['product','retailer','twe','pwe'],drop_level=False)['rmse']=r[4]
          #   e['rmse']=r[4]
          #   print(e)
         #    print(e.index.get_level_values('rmse'))
         #    joined_df.loc[(r[0],r[1],r[2],r[3]),(slice(None))] = 1
             # r[4]
         joined_df.set_index(['rmse'],append=True,inplace=True)
         print("joined_df before indexing=\n",joined_df)
      #   print("RMSE temp=",rmse_temp,len(rmse_temp))
    #    nr=nr.set_index(['product','type','retailer','sortorder','colname','productgroup','twe','pwe'])
         index_left = pd.MultiIndex.from_tuples(rmse_list,
                                        names=['product', 'retailer','type','sortorder','colname','productgroup','twe','pwe','rmse'])
         ind=pd.DataFrame(index=index_left,columns=['rmse_empty'])
         print("index left=\n",ind)
     #    joined_df=pd.merge((joined_df,ind), axis=1)
         #joined_df.index=index_left
         new_joined_df=joined_df.merge(ind, left_index=True, right_on=['product', 'retailer','type',"sortorder","colname",'productgroup','twe','pwe','rmse']).copy()
       #  joined_df.loc[(joined_df.index.level(6)!=0),'rmse']=rmse_temp
 #        joined_df.set_index(['product','type','retailer','colname','twe','pwe','rmse'],append=True,inplace=True)   
         print("after joined RMSE=\n",new_joined_df)
         new_joined_df.drop(['rmse_empty'],axis=1,inplace=True)
         print("2after joined RMSE=\n",new_joined_df)
         new_joined_df.set_index(['rmse'],append=True,inplace=True)    
         new_joined_df.sort_index(['rmse','product','retailer'],ascending=[True,True,True],inplace=True)
         print("3after joined RMSE=\n",new_joined_df)
         return new_joined_df
  
   #  def _RMSE(self,joined_df,product_list,training_window_offset_left_from_end,predict_back_weeks):
   #    #  print("RMSE pbw=",predict_back_weeks)
   #      rmse_list=[]
   #      for p in product_list:
   #          for t in range(training_window_offset_left_from_end,8,-1):
   #              orders_df=joined_df.xs([p[1],p[0],t],level=['product',"retailer","twe"],drop_level=False).T.copy()
 
   #              print("1orders_df=\n",orders_df)   #,"\n",orders_df.T)
 
   #              orders_df=orders_df.sort_index(level='pwe',axis=1,ascending=False,sort_remaining=False)

   #              print("2orders_df=\n",orders_df)   #,"\n",orders_df.T)
   #           #   invoiced=orders_df.iloc[-(predict_back_weeks):-1,[0]]
   #           #   print(p,"invoiced=\n",invoiced,"\n\n")
   #              predicted=orders_df.iloc[-(predict_back_weeks+2):-1]
   #              print(p,"predicted=\n",predicted,predicted.shape,"\n\n")
   #              ordered=np.flipud(predicted).diagonal(offset=0)[:-1][np.newaxis]   #, index=[predicted.index, predicted.columns])
   #           #   ordered=np.diagonal(predicted,offset=0)[:-1][np.newaxis]   #, index=[predicted.index, predicted.columns])

   #              print("ordered=",ordered)
   #              pred=np.flipud(predicted).diagonal(offset=1)[np.newaxis]   #, index=[predicted.index, predicted.columns])
   # #             pred=np.diagonal(predicted,offset=1)[np.newaxis]   #, index=[predicted.index, predicted.columns])
 
   #              print("pred=",pred)
   #              joined=np.concatenate((ordered,pred),axis=0)
   #              print("joined=",joined)
   #           #   print("joined[0,:]",joined[0,:])
   #           #   print("joined[1,:]",joined[1,:])
   #           #   print("joined0-1",joined[0]-joined[1])
   #              rmse_list.append((p[0],p[1],int(round(np.sqrt(((joined[0]-joined[1])**2).mean()),0)),t))
   #            #  rmse=int(round(np.sqrt(((joined[0]-joined[1])**2).mean()),0))
   #            #  joined_df.loc[]df.loc[(slice(​None), slice('price')), :]
   #  #    print("product list",product_list)    
   #   #   print("rmse=",rmse)
   #  #     joined_df['mse']=(joined_df['stock_count']-joined_df['ideal_stock_holdings'])**2
   #  #     rmse=round(np.sqrt(joined_df['mse'].mean()),0)
   #  #  #    print(product+" RMSE=",rmse)
   #      return rmse_list
    
    
    
    
    def predict_majors_orders(self,plot_output_dir):
       # os.chdir("/home/tonedogga/Documents/python_dev")
       # pm=predict_majors()
        window_length=52
        training_window_offset_left_from_end=12
        prediction_window_offset_left_from_end=1
        predict_back_weeks=3

        display_window_length=12
        invoiced_sales_smoothed_over_weeks=2
        offset=2
        
        print("\nPredict majors orders based on scan data.  Version 2..\n")
        sales_df,scan_df=self.load_data()
        weekly_scan_df,product_list=self.chunk_scan(scan_df)
      #  print("wsdf\n",weekly_scan_df)
      #  print("product list=",product_list)
        weekly_sdf=self.chunk_orders(sales_df,product_list,invoiced_sales_smoothed_over_weeks,offset)
        weekly_sdf=self._smooth_orders(weekly_sdf,smth_weeks=invoiced_sales_smoothed_over_weeks)
      #  print("weekly sdf=\n",weekly_sdf)
      #  print("product list",product_list)
     #   print("twe=",training_window_offset_left_from_end)
        joined_df=self._shift_and_join(weekly_sdf,weekly_scan_df,offset)
        training_list=[]
        for prediction_window_offset_left_from_end in range(1,predict_back_weeks+2,1):
         #   print("prediction point=",-prediction_window_offset_left_from_end,"weeks from end.") 
            t_list,joined_df=self._predict(joined_df,product_list,training_window_offset_left_from_end,prediction_window_offset_left_from_end,window_length,invoiced_sales_smoothed_over_weeks,offset,predict_back_weeks)
            training_list.append(t_list)
            #print("predict fin tl",training_list,"jdf=\n",joined_df.iloc[:70,-6:])
        joined_df*=1000
        
        flatten_list = [j for sub in training_list for j in sub] 
     #   print("training list=\n",flatten_list)
        print("joined_df=\n",joined_df)
        rmse_list=self._RMSE(joined_df,flatten_list,training_window_offset_left_from_end,predict_back_weeks)
 
      #  print("rmse list=",rmse_list)
        joined_df=self._fill_rmse(joined_df,rmse_list)
        print("after fill rmse joined_df=\n",joined_df)
    #    print("jshape=",joined_df.shape,"len rsme list",len(rmse_list))
        self._display(joined_df,product_list,rmse_list,plot_output_dir,display_window_length)

        joined_df.iloc[:,-4:].to_excel(plot_output_dir+"prediction_RFR_v2.xlsx",index=True)
        print("\nPredict majors orders v2 finished.")
        return


def main():
    os.chdir("/home/tonedogga/Documents/python_dev")
    pm=predict_majors()
    pm.predict_majors_orders("./")
main()

   