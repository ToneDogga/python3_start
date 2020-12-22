#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:26:13 2020

@author: tonedogga
"""

import glob
import os
import numpy as np
import pandas as pd
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

import matplotlib.cm as cm
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)   # hide fixed formatter warning in matplotlib





import dash2_dict as dd2
  




class stock_cost(object):    
   def _load_SOH(self):
       filenames=sorted(glob.glob(dd2.dash2_dict['production']['save_dir']+'SOH_[*.pkl')) 
       stock_level_dict={}
       for f in filenames:
           g=f.split('[')[1].split("]")[0]
           stock_level_dict[g]=pd.read_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['SOH_savefile'])
       return stock_level_dict   
       
 
       
#   def _load_PP(self):
#       return pd.read_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PP_savefile'])
       
     
       
   def _load_PM(self):
       return pd.read_pickle(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PM_savefile'])
       


   def _load_sales(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            return pd.read_pickle(save_dir+savefile)
        else:
            print("load sales_df error.")
            return
      
        
      
   def _load(self):
        stock_levels_df=self._load_SOH()
     #   print("stock levels=\n",stock_levels_df)
        product_made_df=self._load_PM()
       # print("product made=\n",product_made_df)
       
        sales_df=self._load_sales(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['raw_savefile'])
        sales_df.sort_index(axis=0,ascending=True,inplace=True)
        #sales_df=dash.sales._preprocess(sales_df,dd2.dash2_dict['sales']['rename_columns_dict'])
      #  print("sales=\n",sales_df)
        return stock_levels_df,product_made_df,sales_df



   def _bring_together(self,stock_levels_dict,product_made_df,sales_df):
        #print("stock levels=\n",stock_levels_dict)
     #   print("product made=\n",product_made_df)
        #print("sales=\n",sales_df)
        stocktake_key=max(stock_levels_dict.keys())
        stocktake_date=pd.to_datetime(stocktake_key)+pd.offsets.Day(0)

      #  print("most recent stocktake date:",stocktake_date)
 
        stock_levels=stock_levels_dict[stocktake_key][['code','qtyinstock','lastsalesdate']].copy()
        stock_levels.rename(columns={"code":"product","lastsalesdate":"stockqtydate"},inplace=True)
        
        stock_levels.set_index('product',inplace=True)
     #   new_stock_levels=stock_levels.rename({"code":"product"}).copy()
     #   stock_levels=new_stock_levels[['code','qtyinstock']]
       # stock_levels=stock_levels[['code','qtyinstock']].set_index('code')
     #   print(stock_levels)
        recent_product_made=product_made_df[product_made_df['to_date']>=stocktake_date].copy()   #.set_index('code')   #,drop=False)
        recent_product_made=recent_product_made.rename(columns={"code":"product","qtyunits":"qtymade","to_date":"made_date"})
      #  print("+product made since then:\n",recent_product_made)
        recent_product_group=recent_product_made[['product','qtymade','made_date']].groupby(['product']).agg({"qtymade":"sum","made_date":"min"})
     #   print("+product group since then:\n",recent_product_group)
        recent_sales_df=sales_df[sales_df.index>=stocktake_date].copy()
     #   print("recent sales=\n",recent_sales_df)
        recent_sales_group=recent_sales_df[['product','qty','date']].groupby(['product']).agg({"qty":"sum","date":"max"})
      #  print("product sold since then:\n",recent_sales_group)   #.to_string())
        rsg=recent_sales_group.rename(columns={"qty":"qtysold","date":"sold_date"})
      #  print("- stock sold\n",rsg)
        final=stock_levels.join([recent_product_group,rsg])
        final['qtyinstock']=final['qtyinstock'].replace(np.nan,0)
        final['qtymade']=final['qtymade'].replace(np.nan,0)
        final['qtysold']=final['qtysold'].replace(np.nan,0)
       # final['sold_date']=final['sold_date'].replace(np.nan,0)
     #   final.sort_index(ascending=True,axis=0,inplace=True)
        final['qty']=final['qtyinstock']+final['qtymade']-final['qtysold']
        final.sort_values(['qty'],axis=0,ascending=[True],inplace=True)
        return final



   def _add_pg_to_PM(self,product_made_df,sales_df):
        product_made_df.rename(columns={"code":"product"},inplace=True)
        product_made_df.set_index('product',inplace=True)
        sdf=sales_df.set_index('product')
      #  print(sdf)
      #  print(product_made_df)
      # use sales trans
  #      pdf=product_made_df.join(sdf,how='left').copy()
        pdf=product_made_df.join(sdf,how='inner')
 
      #  print(pdf)
       # return pdf[['to_date',"jobid","qtybatches","qtyunits","productgroup"]]
        rdf= pdf[['to_date',"jobid","qtybatches","qtyunits","productgroup"]].copy()
        rdf.drop_duplicates(subset=['to_date','jobid'],keep='last',inplace=True)
        rdf.reset_index(inplace=True)
        rdf.set_index('to_date',inplace=True,drop=True)
    #    rdf['pg_format_type']=rdf['productgroup']
        rdf['pg_format_type']=rdf['productgroup'].replace(dd2.dash2_dict['scheduler']['pg_format_type'])
        return rdf
       
          
    
   def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
    
    
   def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
   
     
   def align_yaxis(self,ax1, v1, ax2, v2):
        """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny+dy, maxy+dy)
    


   def _plot_mat_sales_vs_production(self,product,sales_df,product_made_df):  #,output_dir):
       
       # stock_df=pd.read_pickle(dd2.dash2_dict['production']['extended_SOH_savefile'])
       # product_made_df=pd.read_pickle(dd2.dash2_dict['production']['extended_PM_savefile'])
        

      #      mat_df=v.copy()
         #  mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
 #           loffset = '7D'
 #           weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
 #           weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 

         spdf=sales_df[sales_df['product']==product]
         pmdf=product_made_df[(product_made_df['product']==product) & (product_made_df['qtyunits']>0)]
      
         
         daily_spdf=spdf.resample('D',label='left').sum().round(0).copy()
         styles1 = ['b-','r-']
        # styles1 = ['bs-','ro:','y^-']
         linewidths = 1  # [2, 1, 4]
                 
         fig, ax1 = pyplot.subplots()
         ax2 = ax1.twinx()
         fig.autofmt_xdate()
         daily_spdf.plot(y='qty',label="unit sales",grid=True,fontsize=7,style="b-", lw=linewidths,ax=ax1)
         ax2=pmdf.plot(y='qtyunits',grid=True,label='unit production',fontsize=7,style="ro", ms=2,secondary_y=True,ax=ax2,legend=False)
          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       
         ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places
 
         ax1.set_title(str(product)+" sales vs production units daily",fontsize= 8)
         ax1.legend(title="",fontsize=7,loc="upper left")
         ax2.legend(title="",fontsize=7,loc="upper center")
         ax1.set_xlabel("",fontsize=7)
         ax1.set_ylabel("",fontsize=7)
         # ax.yaxis.set_major_formatter('${x:1.0f}')
         ax1.yaxis.set_tick_params(which='major', labelcolor='green',
                       labelleft=True, labelright=False)
         self.align_yaxis(ax1, 0, ax2, 0)
         plt.show() 
      #   self._save_fig(product+":sales_vs_production",output_dir)
         plt.close()
         return


 
 #   def _plot_mat_sales_vs_production_weekly_old(self,product,sales_df,product_made_df):  #,output_dir):
       
 #       # stock_df=pd.read_pickle(dd2.dash2_dict['production']['extended_SOH_savefile'])
 #       # product_made_df=pd.read_pickle(dd2.dash2_dict['production']['extended_PM_savefile'])
        

 #      #      mat_df=v.copy()
 #         #  mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
 # #           loffset = '7D'
 # #           weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
 # #           weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 

 #         spdf=sales_df[sales_df['product']==product]
 #         pmdf=product_made_df[(product_made_df['product']==product) & (product_made_df['qtyunits']>0)]
      
         
 #         daily_spdf=spdf.resample('W',label='left').sum().round(0).copy()
 #         styles1 = ['b-','r-']
 #        # styles1 = ['bs-','ro:','y^-']
 #         linewidths = 1  # [2, 1, 4]
                 
 #         fig, ax1 = pyplot.subplots()
 #         ax2 = ax1.twinx()
 #         fig.autofmt_xdate()
 #         daily_spdf.plot(y='qty',label="unit sales",grid=True,fontsize=7,style="b-", lw=linewidths,ax=ax1)
 #         ax2=pmdf.plot(y='qtyunits',grid=True,label='unit production',fontsize=7,style="ro", ms=2,secondary_y=True,ax=ax2,legend=False)
 #          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       
 #         ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places
 
 #         ax1.set_title(str(product)+" sales vs production units weekly",fontsize= 8)
 #         ax1.legend(title="",fontsize=7,loc="upper left")
 #         ax2.legend(title="",fontsize=7,loc="upper center")
 #         ax1.set_xlabel("",fontsize=7)
 #         ax1.set_ylabel("",fontsize=7)
 #         # ax.yaxis.set_major_formatter('${x:1.0f}')
 #         ax1.yaxis.set_tick_params(which='major', labelcolor='green',
 #                       labelleft=True, labelright=False)
 #         self.align_yaxis(ax1, 0, ax2, 0)
 #         plt.show() 
 #      #   self._save_fig(product+":sales_vs_production",output_dir)
 #         plt.close()
 #         return





   def product_RMSE(self,product):
         product_made_df=pd.read_pickle(dd2.dash2_dict['production']['extended_PM_savefile'])
         sales_df=self._load_sales(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['raw_savefile'])

         
         weeks_stock_to_hold=dd2.dash2_dict['scheduler']["RMSE_ideal_weeks_of_stocks"]
         
         spdf=sales_df[sales_df['product']==product][['qty']]
         pmdf=product_made_df[(product_made_df['product']==product) & (product_made_df['qtyunits']>0)][['qtyunits']]
     #    joined_df=spdf.join(pmdf,how='left')
     #    print(joined_df)
     #    weekly_jdf=joined_df.resample('2W',label='left').sum().round(0).copy()

         weekly_spdf=spdf.resample('1W',label='left').sum().round(0).copy()[['qty']]*-1
         weekly_pmdf=pmdf.resample('1W',label='left').sum().round(0).copy()[['qtyunits']]
         
        # print("ws=\n",weekly_spdf)
        # print("wp=\n",weekly_pmdf)
         
         joined_df=weekly_spdf.join(weekly_pmdf,how='left')
         joined_df.replace(np.nan,0,inplace=True)
         joined_df['stock_flow']=joined_df['qtyunits']+joined_df['qty']
         joined_df['empty']=0
         ideal_stock_holdings=-joined_df['qty'].mean()*weeks_stock_to_hold
         starting_units=int(ideal_stock_holdings)
         joined_df['stock_count']=starting_units+joined_df['stock_flow'].cumsum()
         joined_df['ideal_stock_holdings']=ideal_stock_holdings
         joined_df['mse']=(joined_df['stock_count']-joined_df['ideal_stock_holdings'])**2
         joined_df=joined_df[(joined_df.index>pd.to_datetime(joined_df.index[-1])+pd.offsets.Day(-365))]
     #    print(joined_df)
         return round(np.sqrt(joined_df['mse'].mean()),0)
   




   def productgroup_RMSE(self,productgroup):
         product_made_df=pd.read_pickle(dd2.dash2_dict['production']['extended_PM_savefile'])
         sales_df=self._load_sales(dd2.dash2_dict['sales']['save_dir'],dd2.dash2_dict['sales']['raw_savefile'])

         
         weeks_stock_to_hold=dd2.dash2_dict['scheduler']["RMSE_ideal_weeks_of_stocks"]
         
         spdf=sales_df[sales_df['productgroup']==productgroup][['qty']]
         pmdf=product_made_df[(product_made_df['productgroup']==productgroup) & (product_made_df['qtyunits']>0)][['qtyunits']]
     #    joined_df=spdf.join(pmdf,how='left')
     #    print(joined_df)
     #    weekly_jdf=joined_df.resample('2W',label='left').sum().round(0).copy()

         weekly_spdf=spdf.resample('1W',label='left').sum().round(0).copy()[['qty']]*-1
         weekly_pmdf=pmdf.resample('1W',label='left').sum().round(0).copy()[['qtyunits']]
         
        # print("ws=\n",weekly_spdf)
        # print("wp=\n",weekly_pmdf)
         
         joined_df=weekly_spdf.join(weekly_pmdf,how='left')
         joined_df.replace(np.nan,0,inplace=True)
         joined_df['stock_flow']=joined_df['qtyunits']+joined_df['qty']
         joined_df['empty']=0
         ideal_stock_holdings=-joined_df['qty'].mean()*weeks_stock_to_hold
         starting_units=int(ideal_stock_holdings)
         joined_df['stock_count']=starting_units+joined_df['stock_flow'].cumsum()
         joined_df['ideal_stock_holdings']=ideal_stock_holdings
         joined_df['mse']=(joined_df['stock_count']-joined_df['ideal_stock_holdings'])**2
         joined_df=joined_df[(joined_df.index>pd.to_datetime(joined_df.index[-1])+pd.offsets.Day(-365))]
     #    print(joined_df)
         return round(np.sqrt(joined_df['mse'].mean()),0)
   



 
   
   def product_plot_mat_sales_vs_production_weekly(self,product,sales_df,product_made_df,output_dir):
       
       # stock_df=pd.read_pickle(dd2.dash2_dict['production']['extended_SOH_savefile'])
       # product_made_df=pd.read_pickle(dd2.dash2_dict['production']['extended_PM_savefile'])
        

      #      mat_df=v.copy()
         #  mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
 #           loffset = '7D'
 #           weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
 #           weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
      #   starting_units=100000
         weeks_stock_to_hold=dd2.dash2_dict['scheduler']["RMSE_ideal_weeks_of_stocks"]
         day_freq=8
         
         spdf=sales_df[sales_df['product']==product][['qty']]
         pmdf=product_made_df[(product_made_df['product']==product) & (product_made_df['qtyunits']>0)][['qtyunits']]
     #    joined_df=spdf.join(pmdf,how='left')
     #    print(joined_df)
     #    weekly_jdf=joined_df.resample('2W',label='left').sum().round(0).copy()

         weekly_spdf=spdf.resample('1W',label='left').sum().round(0).copy()[['qty']]*-1
         weekly_pmdf=pmdf.resample('1W',label='left').sum().round(0).copy()[['qtyunits']]
         
        # print("ws=\n",weekly_spdf)
        # print("wp=\n",weekly_pmdf)
         
         joined_df=weekly_spdf.join(weekly_pmdf,how='left')
         joined_df.replace(np.nan,0,inplace=True)
         joined_df['stock_flow']=joined_df['qtyunits']+joined_df['qty']
         joined_df['empty']=0
         ideal_stock_holdings=-joined_df['qty'].mean()*weeks_stock_to_hold
         starting_units=int(ideal_stock_holdings)
         joined_df['stock_count']=starting_units+joined_df['stock_flow'].cumsum()
         joined_df['ideal_stock_holdings']=ideal_stock_holdings
         joined_df['mse']=(joined_df['stock_count']-joined_df['ideal_stock_holdings'])**2
         rmse=round(np.sqrt(joined_df['mse'].mean()),0)
     #    print(product+" RMSE=",rmse)
         
         joined_df.to_pickle(dd2.dash2_dict['production']['save_dir']+str(product)+":"+dd2.dash2_dict['production']['joined_sales_and_stock_savefile'],protocol=-1)
 
  #       print("jdf=\n",joined_df.to_string())
      #   print("ish=",ideal_stock_holdings)
           #      pdf=product_made_df.join(sdf,how='left').copy()
      #  pdf=product_made_df.join(sdf,how='inner')
         
         styles1 = ['b-','r-']
        # styles1 = ['bs-','ro:','y^-']
         linewidths = 1  # [2, 1, 4]
                 
         fig, ax = pyplot.subplots()  #   sharex=True)
       #  ax2 = ax1.twinx()
         fig.autofmt_xdate()
#         weekly_pmdf.iloc[:,0].plot(kind='line',label="unit production",grid=True,fontsize=7,style="r-",secondary_y=False,lw=linewidths,ax=ax)
   
 #        weekly_spdf.iloc[:,0].plot(kind='line',grid=False,label='unit sales',fontsize=7,color="blue",legend=False,ax=ax)
        # joined_df.iloc[:,4].plot(use_index=True,kind='line',rot=90,label="",style="k-",grid=False,legend=False,ax=ax)
 
         joined_df.iloc[:,0].plot(use_index=False,sharex=True,kind='line',grid=False,label='stock_sold',fontsize=7,style="b-",lw=1,legend=False,ax=ax)
         joined_df.iloc[:,1].plot(use_index=False,sharex=True,kind='bar',grid=False,label='units made',width=1.2,fontsize=7,color="red",legend=False,ax=ax)
         joined_df.plot(y="stock_count",sharex=True,use_index=False,kind='line',grid=False,label='calc stock count',fontsize=7,style="g-",lw=1,legend=False,ax=ax)  
         
         ax.hlines(y=ideal_stock_holdings, xmin=0, xmax=joined_df.shape[0],ls='--', color='black',label="ideal level "+str(weeks_stock_to_hold)+" weeks")   #color="black")
         joined_df.plot(y="empty",sharex=True,use_index=True,kind='line',grid=True,label='',fontsize=7,style="k-",lw=1,legend=False,ax=ax)  
         ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places

         #ax2 = ax1.twinx()
          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       

      #   ax2 = ax1.twinx()
 
 
         ax.set_title(str(product)+" calculated stock units. Starting units="+str(starting_units)+" RMSE="+str(rmse),fontsize= 8)
         ax.legend(title="",fontsize=7,loc="upper left")
       #  ax.legend(title="",fontsize=7,loc="upper center")
         ax.set_xlabel("",fontsize=7)
         ax.set_ylabel("",fontsize=7)
         # ax.yaxis.set_major_formatter('${x:1.0f}')
         ax.yaxis.set_tick_params(which='major', labelcolor='green',
                       labelleft=True, labelright=False)
 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         day_position=joined_df.index.to_list() 
     #    print(day_position)
         newdates = [i.toordinal() for i in day_position]
 
     #   print(newdates)
    
         new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
      #   print(new_labels)
         improved_labels = ['{}{}{}'.format(d,calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
      #   print(improved_labels) 

     #   # improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
     # #   improved_labels=improved_labels[:1]+improved_labels
     #  #   improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
     #  #   improved_labels=improved_labels[:1]+improved_labels


         improved_labels=improved_labels[:1]+improved_labels[::day_freq]
      #   improved_labels=improved_labels[::day_freq]
     
        
        
        
  
         ax.xaxis.set_major_locator(ticker.MultipleLocator(day_freq))
         ax.xaxis.set_tick_params(which='major', labelcolor='black',
                           pad=1.2)
      
        
        
         ax.set_xticklabels(improved_labels,fontsize=7,ha='right')
    
    
    
 #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    
    
    
    
  #       self.align_yaxis(ax1, 0, ax2, 0)
  #       plt.show() 
         self._save_fig("psoh_product:"+str(product)+":sales_vs_production_stock",output_dir)
         plt.close()
         return joined_df




 
   
   def productgroup_plot_mat_sales_vs_production_weekly(self,productgroup,sales_df,product_made_df,output_dir):
       
       # stock_df=pd.read_pickle(dd2.dash2_dict['production']['extended_SOH_savefile'])
       # product_made_df=pd.read_pickle(dd2.dash2_dict['production']['extended_PM_savefile'])
        

      #      mat_df=v.copy()
         #  mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
 #           loffset = '7D'
 #           weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
 #           weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
   #      starting_units=100000
         weeks_stock_to_hold=dd2.dash2_dict['scheduler']["RMSE_ideal_weeks_of_stocks"]
         day_freq=8
         
         spdf=sales_df[sales_df['productgroup']==productgroup][['qty']]
         pmdf=product_made_df[(product_made_df['productgroup']==productgroup) & (product_made_df['qtyunits']>0)][['qtyunits']]
     #    joined_df=spdf.join(pmdf,how='left')
     #    print(joined_df)
     #    weekly_jdf=joined_df.resample('2W',label='left').sum().round(0).copy()

         weekly_spdf=spdf.resample('1W',label='left').sum().round(0).copy()[['qty']]*-1
         weekly_pmdf=pmdf.resample('1W',label='left').sum().round(0).copy()[['qtyunits']]
         
        # print("ws=\n",weekly_spdf)
        # print("wp=\n",weekly_pmdf)
         
         joined_df=weekly_spdf.join(weekly_pmdf,how='left')
         joined_df.replace(np.nan,0,inplace=True)
         joined_df['stock_flow']=joined_df['qtyunits']+joined_df['qty']
         joined_df['empty']=0
         ideal_stock_holdings=-joined_df['qty'].mean()*weeks_stock_to_hold
         starting_units=int(ideal_stock_holdings)
         joined_df['stock_count']=starting_units+joined_df['stock_flow'].cumsum()
         joined_df['ideal_stock_holdings']=ideal_stock_holdings
         joined_df['mse']=(joined_df['stock_count']-joined_df['ideal_stock_holdings'])**2
         rmse=round(np.sqrt(joined_df['mse'].mean()),0)
    #     print(productgroup+" RMSE=",rmse)
    
         joined_df.to_pickle(dd2.dash2_dict['production']['save_dir']+str(productgroup)+":"+dd2.dash2_dict['production']['joined_sales_and_stock_savefile'],protocol=-1)
      #   print("jdf=\n",joined_df.to_string())
      #   print("ish=",ideal_stock_holdings)
           #      pdf=product_made_df.join(sdf,how='left').copy()
      #  pdf=product_made_df.join(sdf,how='inner')
         
         styles1 = ['b-','r-']
        # styles1 = ['bs-','ro:','y^-']
         linewidths = 1  # [2, 1, 4]
                 
         fig, ax = pyplot.subplots()  #   sharex=True)
       #  ax2 = ax1.twinx()
         fig.autofmt_xdate()
#         weekly_pmdf.iloc[:,0].plot(kind='line',label="unit production",grid=True,fontsize=7,style="r-",secondary_y=False,lw=linewidths,ax=ax)
   
 #        weekly_spdf.iloc[:,0].plot(kind='line',grid=False,label='unit sales',fontsize=7,color="blue",legend=False,ax=ax)
        # joined_df.iloc[:,4].plot(use_index=True,kind='line',rot=90,label="",style="k-",grid=False,legend=False,ax=ax)
 
         joined_df.iloc[:,0].plot(use_index=False,sharex=True,kind='line',grid=False,label='stock_sold',fontsize=7,style="b-",lw=1,legend=False,ax=ax)
         joined_df.iloc[:,1].plot(use_index=False,sharex=True,kind='bar',grid=False,label='units made',width=1.2,fontsize=7,color="red",legend=False,ax=ax)
         joined_df.plot(y="stock_count",sharex=True,use_index=False,kind='line',grid=False,label='calc stock count',fontsize=7,style="g-",lw=1,legend=False,ax=ax)  
         
         ax.hlines(y=ideal_stock_holdings, xmin=0, xmax=joined_df.shape[0],ls='--', color='black',label="ideal level "+str(weeks_stock_to_hold)+" weeks")   #color="black")
         joined_df.plot(y="empty",sharex=True,use_index=True,kind='line',grid=True,label='',fontsize=7,style="k-",lw=1,legend=False,ax=ax)  
         ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places

         #ax2 = ax1.twinx()
          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       

      #   ax2 = ax1.twinx()
 
 
         ax.set_title("Product group:"+str(productgroup)+" calculated stock units. Starting units="+str(starting_units)+" RMSE="+str(rmse),fontsize= 8)
         ax.legend(title="",fontsize=7,loc="upper left")
       #  ax.legend(title="",fontsize=7,loc="upper center")
         ax.set_xlabel("",fontsize=7)
         ax.set_ylabel("",fontsize=7)
         # ax.yaxis.set_major_formatter('${x:1.0f}')
         ax.yaxis.set_tick_params(which='major', labelcolor='green',
                       labelleft=True, labelright=False)
 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
         day_position=joined_df.index.to_list() 
     #    print(day_position)
         newdates = [i.toordinal() for i in day_position]
 
     #   print(newdates)
    
         new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
      #   print(new_labels)
         improved_labels = ['{}{}{}'.format(d,calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
      #   print(improved_labels) 

     #   # improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
     # #   improved_labels=improved_labels[:1]+improved_labels
     #  #   improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
     #  #   improved_labels=improved_labels[:1]+improved_labels


         improved_labels=improved_labels[:1]+improved_labels[::day_freq]
      #   improved_labels=improved_labels[::day_freq]
     
        
        
        
  
         ax.xaxis.set_major_locator(ticker.MultipleLocator(day_freq))
         ax.xaxis.set_tick_params(which='major', labelcolor='black',
                           pad=1.2)
      
        
        
         ax.set_xticklabels(improved_labels,fontsize=7,ha='right')
    
    
 #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   

    
  #       self.align_yaxis(ax1, 0, ax2, 0)
   #      plt.show() 
         self._save_fig("psoh_pg:"+str(productgroup)+":sales_vs_production_stock",output_dir)
         plt.close()
         return joined_df





   
#    def format_plot_mat_sales_vs_production_weekly(self,pg_format_type,sales_df,product_made_df):  #,output_dir):
       
#        # stock_df=pd.read_pickle(dd2.dash2_dict['production']['extended_SOH_savefile'])
#        # product_made_df=pd.read_pickle(dd2.dash2_dict['production']['extended_PM_savefile'])
        

#       #      mat_df=v.copy()
#          #  mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
#  #           loffset = '7D'
#  #           weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
#  #           weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
#          starting_units=100000
#          weeks_stock_to_hold=5
#          day_freq=8
         
#          spdf=sales_df[sales_df['productgroup']==productgroup][['qty']]
#          pmdf=product_made_df[(product_made_df['pg_format_type']==pg_format_type) & (product_made_df['qtyunits']>0)][['qtyunits']]
#      #    joined_df=spdf.join(pmdf,how='left')
#      #    print(joined_df)
#      #    weekly_jdf=joined_df.resample('2W',label='left').sum().round(0).copy()

#          weekly_spdf=spdf.resample('1W',label='left').sum().round(0).copy()[['qty']]*-1
#          weekly_pmdf=pmdf.resample('1W',label='left').sum().round(0).copy()[['qtyunits']]
         
#         # print("ws=\n",weekly_spdf)
#         # print("wp=\n",weekly_pmdf)
         
#          joined_df=weekly_spdf.join(weekly_pmdf,how='left')
#          joined_df.replace(np.nan,0,inplace=True)
#          joined_df['stock_flow']=joined_df['qtyunits']+joined_df['qty']
#          joined_df['stock_count']=starting_units+joined_df['stock_flow'].cumsum()
#          joined_df['empty']=0
#          ideal_stock_holdings=-joined_df['qty'].mean()*weeks_stock_to_hold
#          print("jdf=\n",joined_df.to_string())
#       #   print("ish=",ideal_stock_holdings)
#            #      pdf=product_made_df.join(sdf,how='left').copy()
#       #  pdf=product_made_df.join(sdf,how='inner')
         
#          styles1 = ['b-','r-']
#         # styles1 = ['bs-','ro:','y^-']
#          linewidths = 1  # [2, 1, 4]
                 
#          fig, ax = pyplot.subplots()  #   sharex=True)
#        #  ax2 = ax1.twinx()
#          fig.autofmt_xdate()
# #         weekly_pmdf.iloc[:,0].plot(kind='line',label="unit production",grid=True,fontsize=7,style="r-",secondary_y=False,lw=linewidths,ax=ax)
   
#  #        weekly_spdf.iloc[:,0].plot(kind='line',grid=False,label='unit sales',fontsize=7,color="blue",legend=False,ax=ax)
#         # joined_df.iloc[:,4].plot(use_index=True,kind='line',rot=90,label="",style="k-",grid=False,legend=False,ax=ax)
 
#          joined_df.iloc[:,0].plot(use_index=False,sharex=True,kind='line',grid=False,label='stock_sold',fontsize=7,style="b-",lw=1,legend=False,ax=ax)
#          joined_df.iloc[:,1].plot(use_index=False,sharex=True,kind='bar',grid=False,label='units made',width=1.2,fontsize=7,color="red",legend=False,ax=ax)
#          joined_df.plot(y="stock_count",sharex=True,use_index=False,kind='line',grid=False,label='calc stock count',fontsize=7,style="g-",lw=1,legend=False,ax=ax)  
         
#          ax.hlines(y=ideal_stock_holdings, xmin=0, xmax=joined_df.shape[0],ls='--', color='black',label="ideal level "+str(weeks_stock_to_hold)+" weeks")   #color="black")
#          joined_df.plot(y="empty",sharex=True,use_index=True,kind='line',grid=True,label='',fontsize=7,style="k-",lw=1,legend=False,ax=ax)  
#          ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places

#          #ax2 = ax1.twinx()
#           #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       

#       #   ax2 = ax1.twinx()
 
 
#          ax.set_title(str(productgroup)+" calculated stock units. Starting units="+str(starting_units),fontsize= 8)
#          ax.legend(title="",fontsize=7,loc="upper left")
#        #  ax.legend(title="",fontsize=7,loc="upper center")
#          ax.set_xlabel("",fontsize=7)
#          ax.set_ylabel("",fontsize=7)
#          # ax.yaxis.set_major_formatter('${x:1.0f}')
#          ax.yaxis.set_tick_params(which='major', labelcolor='green',
#                        labelleft=True, labelright=False)
 
#     #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#          day_position=joined_df.index.to_list() 
#      #    print(day_position)
#          newdates = [i.toordinal() for i in day_position]
 
#      #   print(newdates)
    
#          new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
#       #   print(new_labels)
#          improved_labels = ['{}{}{}'.format(d,calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
#       #   print(improved_labels) 

#      #   # improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
#      # #   improved_labels=improved_labels[:1]+improved_labels
#      #  #   improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
#      #  #   improved_labels=improved_labels[:1]+improved_labels


#          improved_labels=improved_labels[:1]+improved_labels[::day_freq]
#       #   improved_labels=improved_labels[::day_freq]
     
        
        
        
  
#          ax.xaxis.set_major_locator(ticker.MultipleLocator(day_freq))
#          ax.xaxis.set_tick_params(which='major', labelcolor='black',
#                            pad=1.2)
      
        
        
#          ax.set_xticklabels(improved_labels,fontsize=7,ha='right')
    
    
    
#  #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    
    
    
    
    
    
    
    
    
    
    
#   #       self.align_yaxis(ax1, 0, ax2, 0)
#      #   plt.show() 
#       #   self._save_fig(product+":sales_vs_production",output_dir)
#          plt.close()
#          return







   def stock_summary(self,plot_output_dir):
        stock_levels_dict,product_made_df,sales_df=self._load()

        stock_df=self._bring_together(stock_levels_dict,product_made_df,sales_df)
        stock_df.to_pickle(dd2.dash2_dict['production']['extended_SOH_savefile'],protocol=-1)
 
        print("\n",stock_df.to_string(),"\n")
      #  print(stock_df.to_string())
        product_made_df=self._add_pg_to_PM(product_made_df,sales_df)
        product_made_df.to_pickle(dd2.dash2_dict['production']['extended_PM_savefile'],protocol=-1)
        print("\n",product_made_df,"\n")
  
        for p in dd2.dash2_dict['production']["products_to_plot_stock_levels"]:
            print("plotting product:",p,"stock levels")
            self.product_plot_mat_sales_vs_production_weekly(p,sales_df,product_made_df,plot_output_dir)
            r=self.product_RMSE(p)  
            print("product",p,"the last 365 days of accuracy of scheduling against ideal stock levels",dd2.dash2_dict['scheduler']["RMSE_ideal_weeks_of_stocks"],"weeks. RMSE=",r)
    
        for pg in dd2.dash2_dict['production']["productgroups_to_plot_stock_levels"]:
            print("plotting product group:",pg,"stock levels")
            self.productgroup_plot_mat_sales_vs_production_weekly(pg,sales_df,product_made_df,plot_output_dir)
            r=self.productgroup_RMSE(pg)
            print("productgroup",pg,"the last 365 days of accuracy of scheduling against ideal stock levels",dd2.dash2_dict['scheduler']["RMSE_ideal_weeks_of_stocks"],"weeks. RMSE=",r)    
   
      #  


  #      print(self.product_RMSE("SJ300"))
  #      print(self.productgroup_RMSE("11"))

        print("\n")
        return #sales_df,product_made_df







#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# def main():
#     os.chdir("/home/tonedogga/Documents/python_dev")
#     sc=stock_cost()
#    # sales_df,product_made_df=sc.stock_summary("./")
#     sc.stock_summary("./")
# #    sc.product_plot_mat_sales_vs_production_weekly("SJ300",sales_df,product_made_df,plot_output_dir)
# #    sc.productgroup_plot_mat_sales_vs_production_weekly("10",sales_df,product_made_df,plot_output_dir)
    
#  #   sc.calculate_all_products_scheduling_accuracy()
    
    
    
#     print(sc.product_RMSE("SJ300"))
#     print(sc.productgroup_RMSE("11"))
#   #   sc.format_plot_mat_sales_vs_production_weekly(1,sales_df,product_made_df)

#   #   print(product_made_df)   #.to_string())
#     # product_made_df=sc._add_pg_to_PM(product_made_df,sales_df)

    
#   #  print(product_made_df)
# main()    
