#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:02:59 2020

@author: tonedogga
"""

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import os
import datetime as dt

import matplotlib.pyplot as plt  
import dash2_dict as dd2 
 

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)   # hide fixed formatter warning in matplotlib

#import os
#import numpy as np
#import pandas as pd
from pandas.tseries.frequencies import to_offset

from datetime import datetime,timedelta

#import datetime as dt
from datetime import date
#from datetime import timedelta
import calendar
import xlsxwriter
#import xlrd


#import matplotlib.pyplot as plt
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

#import PIL
import glob
import imageio
import cv2
  
from p_tqdm import p_map,p_umap

       

   

class scheduler(object):
    def __init__(self):
                
        self.urgency_maxx=10
        self.urgency_maxy=2
        self.urgency_cross=4
        
        
        self.efficiency_maxx=50
        self.efficiency_maxy=2.1
        self.efficiency_cross=7
        
        self.total_days_in_year=366  # one extra as index
        self.total_maxx=101
        self.total_graph_xpoints=10
        return
        
        
        
  
    
     
    def area_under_curve(self,curve,startx,endx):
        return np.trapz(curve[startx:endx+1])
    
    
    
    def urgency(self,z,maxx,maxy,urgency_cross):
        return maxy-(maxy / (1+np.exp(-z+(urgency_cross))))
    
    
    def efficiency(self,z,maxx,maxy,efficiency_cross):
        return (maxy / (1 + np.exp(-z+(efficiency_cross))))
    
    
    #--------------------------------------------------------------------------------------------------
    
    
    def flat_base(self,z,*,length,y):
        return np.full((length),y)
    
    
    def add_a_curve(self,z,*,start,end,xoffset,yoffset,curveoffset,wl,amp,y_shift):  #,start,end,xoffset,yoffset,wl,amp):
        return np.add(xoffset+np.sin(((z[start:end]+curveoffset)/wl)+yoffset)/amp,y_shift)
    
    
    def adjust_to_average_one(self,curve,y):
        return np.add(curve,y)
    
    #-----------------------------------------------------------------------------------------------
    
    
    
     
    def add_xmas_bump(self,z,*,length,donor_curve):
        start=210   # start day of seasonal increase out of 365 days
        xmas_up=self.add_a_curve(z,start=start,end=length,xoffset=1.1,yoffset=250,curveoffset=2,wl=1.3,amp=5,y_shift=0)
        return np.r_[donor_curve[:start],xmas_up,donor_curve[length:]]
    
    
    def add_easter_bump(self,z,*,start,end,donor_curve):
      #  start=60   # start day of seasonal increase out of 365 days
        easter_up=self.add_a_curve(z,start=start,end=end,xoffset=1.1,yoffset=200,curveoffset=2,wl=0.1,amp=27,y_shift=-0.083)
        return np.r_[donor_curve[:start],easter_up,donor_curve[end:]]
        
    
    def add_long_weekend_bump(self,z,*,start,end,donor_curve):
        lw_up=self.add_a_curve(z,start=start,end=end,xoffset=1.1,yoffset=250,curveoffset=2.9,wl=0.11,amp=5,y_shift=0)
        return np.r_[donor_curve[:start],lw_up,donor_curve[end:]]
        
    
    def add_winter_rise(self,z,*,start,end,donor_curve):
        w_up=self.add_a_curve(z,start=start,end=end,xoffset=1.1,yoffset=250,curveoffset=1.8,wl=0.7,amp=5,y_shift=0.087)
        return np.r_[donor_curve[:start],w_up,donor_curve[end:]]
    
    
    def add_winter_fall(self,z,*,start,end,donor_curve):
        s_up=self.add_a_curve(z,start=start,end=end,xoffset=1.1,yoffset=250,curveoffset=1.8,wl=0.7,amp=5,y_shift=0.11)
        return np.r_[donor_curve[:start],2-s_up,donor_curve[end:]]
    
    
    
     
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
          
 
    def create_curves(self): 
        
        z = np.linspace(0, self.total_graph_xpoints, self.total_maxx)
  
        urgency_curve=self.urgency(z,self.urgency_maxx,self.urgency_maxy,self.urgency_cross)
        efficiency_curve=self.efficiency(z,self.efficiency_maxx,self.efficiency_maxy,self.efficiency_cross)
        
        urgency_curve=np.r_[urgency_curve,np.full((self.total_days_in_year-self.total_maxx), 0)]
        efficiency_curve=np.r_[efficiency_curve,np.full((self.total_days_in_year-self.total_maxx), 2)]
        
        z2 = np.linspace(0, self.urgency_maxx, self.total_days_in_year)
       
      #  non_seasonal_base_curve=self.flat_base(z,length=self.total_days_in_year,y=1.0)   #np.ones((total_days_in_year))
        seasonal_base_curve=self.flat_base(z,length=self.total_days_in_year,y=0.932)   #np.ones((total_days_in_year))
 
        base_curve_plus_xmas=self.add_xmas_bump(z2,length=self.total_days_in_year,donor_curve=seasonal_base_curve)   #np.ones((total_days_in_year))
    
        base_curve_plus_xmas_and_easter=self.add_easter_bump(z2,start=66,end=85,donor_curve=base_curve_plus_xmas)
    
        base_curve_plus_xmas_and_easter_and_lw=self.add_long_weekend_bump(z2,start=250,end=272,donor_curve=base_curve_plus_xmas_and_easter)
    
    
        jams_condiments_and_mealbases_curve=self.adjust_to_average_one(self.add_winter_rise(z2,start=85,end=250,donor_curve=base_curve_plus_xmas_and_easter_and_lw),-0.13)
    
        sauces_and_dressings_curve=self.adjust_to_average_one(self.add_winter_fall(z2,start=85,end=250,donor_curve=base_curve_plus_xmas_and_easter_and_lw),0.045)
    
        curves= np.c_[urgency_curve,efficiency_curve,jams_condiments_and_mealbases_curve,sauces_and_dressings_curve]
        return curves
        #return pd.DataFrame(curves,columns=["urgency_factor","efficiency_factor","winter_demand","summer_demand"]),curves
 
    
 
           
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
     
 
    
    
    def display_curves(self,curves,plot_output_dir):
        print("blue area under curve (to 100 days) urgency",self.area_under_curve(curves[:,0],0,100))
        print("red area under curve (to 100 days) efficiency",self.area_under_curve(curves[:,1],0,100))
        print("magenta area under curve sauces, dressings",self.area_under_curve(curves[:,2],0,366))
        print("green area under curve jams, condiments and mealbases",self.area_under_curve(curves[:,3],0,366))
        
        plt.plot(curves[:,0], "b-", linewidth=2)
        plt.plot(curves[:,1], "r-", linewidth=2)
        plt.plot(curves[:,2], "g-", linewidth=2)
        plt.plot(curves[:,3], "m-", linewidth=2)
        plt.grid(True)
        plt.title("Urgency (blue) vs efficiency (red) plus sales rate jams(green), dressings(mag)", fontsize=10)
        plt.xlabel("days to out of stock / % of ideal manu run / days of year",fontsize=10)
        self._save_fig("scheduler_curves",plot_output_dir)
        #plt.show()
        
        return
    
     
    
    def _load_SOH(self,filename):
         stock_on_hand_df=pd.read_pickle(filename)  
         stock_on_hand_df['pg']=stock_on_hand_df['productgroup']
         stock_on_hand_df["pg"].replace(dd2.dash2_dict['scheduler']['productgroup_dict'], inplace=True)
         stock_on_hand_df=stock_on_hand_df[stock_on_hand_df['pg'].isin(dd2.dash2_dict['scheduler']['productgroup_mask'])]
         stock_on_hand_df['pg_type']=stock_on_hand_df['pg'] 
         stock_on_hand_df["pg_type"].replace(dd2.dash2_dict['scheduler']['pg_type'],inplace=True)
         stock_on_hand_df['format_type']=stock_on_hand_df['productgroup'] 
         stock_on_hand_df["format_type"].replace(dd2.dash2_dict['scheduler']['format_type'],inplace=True)
         stock_on_hand_df.rename(columns={'code':"product"},inplace=True)
         return stock_on_hand_df[["product","format_type","pg_type","productgroup","pg","lastsalesdate","qtyinstock"]]


         
    def _load_PP(self,filename):
         return pd.read_pickle(filename)  
        # return stock_on_hand_df[["product","format_type","pg_type","productgroup","pg","lastsalesdate","qtyinstock"]]

         
    def _load_PM(self,filename):
         return pd.read_pickle(filename)  
        # return stock_on_hand_df[["product","format_type","pg_type","productgroup","pg","lastsalesdate","qtyinstock"]]
        
  
    
    
    def load_average_weekly_sales(self,filename):
        df=pd.read_pickle(filename)
        latest_date=df.index[0]
    #    print("latest date",latest_date)
        df=df[(df.index>pd.to_datetime(latest_date)+pd.offsets.Day(-365)) & (df.index<=pd.to_datetime(latest_date))]
        
      #  df.reset_index(drop=True, inplace=True)
        
        
       # df["total_units_for_last_365_days"]=df['qty'].groupby(df["product"]).transform(sum)
      #  new_df=df[['product','qty']].resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
       # print(new_df)
       # print(len(list(set(list(df['product'])))))
        #df['average_units_per_week']=df['qty'].groupby(df["product"]).transform(sum)/52
        return df['qty'].groupby(df["product"]).sum()/52,latest_date  
        
       # return df[["product","average_units_per_week","total_units_for_last_365_days"]]     
    
    
    def calc_urgency(self,day_no,est_days_to_make,curves):
        # the day no is the days to stock out, not the day it will be manufactured
        if est_days_to_make<0:
            est_days_to_make=0
        if est_days_to_make>200:
            est_days_to_make=200
        if est_days_to_make>day_no:
            day_no=1
        
        if day_no-est_days_to_make>364:
            day_no=364
        if day_no-est_days_to_make<0:
            day_no=1
            
    #    print("dayno",day_no,"est_days_to_make",est_days_to_make)    
        return np.mean(curves[day_no-est_days_to_make:day_no+2],axis=0).copy()
    #    print("u=",c)
    #    return c
 
    
    def calc_efficiency(self,stock_to_make,minimum_ideal_volume,curves):
        # the day no is the days to stock out, not the day it will be manufactured
        percent_efficiency=int(round(100*stock_to_make/minimum_ideal_volume,0))  #.astype(np.int32)
        if percent_efficiency>100:
            percent_efficiency=100
        if percent_efficiency<=0:
            percent_efficiency=0
    #    print("dayno",day_no,"est_days_to_make",est_days_to_make)    
        return curves[percent_efficiency,1].copy()
    #    print("e=",percent_efficiency,"%",c)
     #   return c
 
  
   
    def calc_demand(self,days_to_wait,curves):
        # the day no is the days to stock out, not the day it will be manufactured
        date = pd.to_datetime('today')
        new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
        day_of_the_year = (date - new_year_day).days + 1
        
      #  print("calc demand day of year",day_of_the_year)    
    #    print("dayno",day_no,"est_days_to_make",est_days_to_make)    
        return curves[day_of_the_year,2:].copy()
   #     print("f=",day_of_the_year,"%",c)
     #   return c




    def _find_latest_manu_date(self,production_made_df):
        pmd=list(set(list(production_made_df['code'])))    # product code    
        latest_date_dict={}
        for p in pmd: 
             w=production_made_df[production_made_df['code']==p]
          #   print("w=\n",w)
             latest_date_dict[p]=w['to_date'].max(axis=0)
     #   print("latest_date_dict=",latest_date_dict)
    #    stocks["format_priority"].replace(fp, inplace=True)
        return latest_date_dict



    def _replace_latest_manu_date(self,latest_date_dict,s):
        ps=list(set(list(s['product'])))    
        stocks=s.copy()
        for p in ps: 
             stocks['last_manu_date'][stocks['product']==p]=latest_date_dict[p]
             
       # print("fp=",fp)
        stocks["weeks_since_last_made"]=(pd.to_datetime('today')-pd.to_datetime(stocks['last_manu_date'])).dt.days/7    #round("D")
        return stocks

    
    
            
    def display_schedule(self,output_dir):
     #   os.chdir("/home/tonedogga/Documents/python_dev")
     #   cwdpath = os.getcwd()
       # s=schedule()    
       # sc=scheduler_curves()
        
        target_weeks_of_stocks=5
        minimum_weeks_stock_holdings=2
        estimated_days_to_make_from_scheduling=1
        urgency_weight=1.0
        efficiency_weight=1.0
        minimum_onehundred_percent_run=40000
        
        curves=self.create_curves()
      #  print(curves)
        self.display_curves(curves,output_dir)
       # print("curves_df=\n",curves_df)
        
        stock_on_hand_df=self._load_SOH(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['SOH_savefile'])
      #  print("SOH=\n",stock_on_hand_df)  

        production_made_df=self._load_PM(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PM_savefile'])
      #  print("PM=\n",production_made_df)  

        latest_date_dict=self._find_latest_manu_date(production_made_df)      


        weekly_sales_df,latest_date=self.load_average_weekly_sales(dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['savefile'])
        #print("weekly sales df=\n",weekly_sales_df)
        stocks=pd.merge(stock_on_hand_df,weekly_sales_df,on=["product"])   #['product']=='product']]
        stocks.rename(columns={'qty':"ave_qty_sold_per_week"},inplace=True)
    
        stocks['no_weeks_sales_as_target']=target_weeks_of_stocks
        stocks['minimum_weeks_stock_holdings']=minimum_weeks_stock_holdings
        
        stocks['target_holdings']=round(stocks['ave_qty_sold_per_week']*target_weeks_of_stocks,0).astype(np.int32)
        stocks['stock_needed']=stocks['target_holdings']-stocks['qtyinstock']
        stocks['weeks_to_stock_out']=round(stocks['qtyinstock'] / ((stocks['ave_qty_sold_per_week'])),1)
        stocks['weeks_to_minimum']=round(stocks['qtyinstock'] / stocks['ave_qty_sold_per_week'],1)-stocks['minimum_weeks_stock_holdings']
 
        stocks['urgency']=0.0
        stocks['effic']=0.0
        stocks['winter_demand']=0.0  #*(1-stocks['pg_type'])+curv[3]*stocks['pg_type']
        stocks['summer_demand']=0.0
        stocks['last_manu_date']=pd.to_datetime("today")
        stocks['weeks_since_last_made']=0
    #    print("stocks before:=\n",stocks.to_string())
        
        latest_date_dict=self._find_latest_manu_date(production_made_df)      
        stocks=self._replace_latest_manu_date(latest_date_dict,stocks)
    
    
     #   print("stocks after:=\n",stocks.to_string())
        # stocks['effic']=0.0
        #for row in range(0,stocks.shape[0]):
        weeks_no_list=stocks['weeks_to_stock_out']
        for i,w in enumerate(weeks_no_list):   
         #  print("urgency",i,d) 
           curv=self.calc_urgency(int(w/7),estimated_days_to_make_from_scheduling,curves)
           stocks['urgency'].iloc[i]=curv[0]
           
           
        sn_list=stocks['stock_needed']
        for i,e in enumerate(sn_list):   
         #  print("effic",i,e) 
           curv2=self.calc_efficiency(e,minimum_onehundred_percent_run,curves)
           stocks['effic'].iloc[i]=curv2
          
    #    sd={}    
    #    product_list=list(set(list(stocks['product']))) 
      #  print("product_list=",product_list)
    #    for p in product_list:
    #        stocks[stocks['product']==p].sort_index(ascending=True)
    #        sd[p]=5  #(pd.to_datetime("today")-pd.to_datetime(stocks['lastsalesdate'].iloc[-1]))    #.strftime("%d/%m/%Y")
            
          
        curv3=self.calc_demand(estimated_days_to_make_from_scheduling,curves)  
        stocks['winter_demand']=curv3[0]  #*(1-stocks['pg_type'])+curv[3]*stocks['pg_type']
        stocks['summer_demand']=curv3[1]
    
      
        stocks['demand_factor']=stocks['winter_demand']*(1-stocks['pg_type'])+stocks['summer_demand']*stocks['pg_type']
        stocks=stocks.drop(['winter_demand','summer_demand'],axis=1) 
        stocks["adjusted_ave_qty_sold_per_week"]=stocks['ave_qty_sold_per_week']*stocks['demand_factor']
        
        stocks['target_holdings']=round(stocks['adjusted_ave_qty_sold_per_week']*target_weeks_of_stocks,0).astype(np.int32)
        
        stocks['priority']=(stocks['effic']*efficiency_weight+stocks['urgency']*urgency_weight)
        stocks=stocks[stocks['stock_needed']>0]
        stocks['format_priority']=stocks['pg']
        stocks['format_type_priority']=stocks['pg_type']
        pgs=list(set(list(stocks['pg'])))
        fp={}
        for pg in pgs:
            v=stocks[stocks['pg']==pg]
       
            #print("v",v,np.mean(v['priority']))
            fp[pg]=np.mean(v['priority'])
            
        pgts=list(set(list(stocks['pg_type'])))    
        ft={}
        for pgt in pgts: 
             w=stocks[stocks['pg_type']==pgt]
             ft[pgt]=np.mean(w['priority'])
     #   print("fp=",fp)
        stocks["format_priority"].replace(fp, inplace=True)
        stocks["format_type_priority"].replace(ft, inplace=True)
       # stocks['days_since_last_manu'].replace(sd,inplace=True)
        stocks=stocks.sort_values(['format_type_priority','format_priority','priority','weeks_to_stock_out'],axis=0,ascending=[False,False,False,True])
        print("schedule at",latest_date.strftime("%d/%m/%Y"),"=\n",stocks.to_string())  #,"\n",stocks.T)
        stocks.to_pickle(dd2.dash2_dict['scheduler']['schedule_savedir']+dd2.dash2_dict['scheduler']['schedule_savefile'],protocol=-1)
        
        
      #  print(stocks[["product","qtyinstock","no_weeks_sales_as_target","stock_needed","days_to_stock_out","format_type","format_priority","priority"]].to_string())
        # stock to make = target_weeks_of_SOH*historic_weekly_sales_rate*seasonality_factor- Current SOH
        #  days to stock out=Current SOH /  (historic_weekly_sales_rate/7*seasonality_factor)
        print("\n===============================================================================================\n")
        return



    
    # def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
    #     os.makedirs(output_dir, exist_ok=True)
    #     path = os.path.join(output_dir, fig_id + "." + fig_extension)
    #   #  print("Saving figure", fig_id)
    #     if tight_layout:
    #         plt.tight_layout()
    #     plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
    #     return
    
    
    def _align_yaxis(self,ax1, v1, ax2, v2):
        """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny+dy, maxy+dy)  
      
    
    
    def _plot_engine(self,plot_dict):
     
    #  pass a dictionary - plot_dict
#  {"df":df,"plot_list":plot_list,"title":title,"lefty_title":...,"righty_title":....,"bottomx_title":....,"topx_title":...}        

        df=plot_dict['df']
      #  day_position=df.index.get_level_values('day_position').to_list()
        plot_len=(plot_dict['end_date']-plot_dict['start_date']).days
        
        if plot_len<=80:
            day_freq=1
        elif ((plot_len>80) & (plot_len<=160)): 
            day_freq=2
        elif ((plot_len>160) & (plot_len<=240)): 
            day_freq=3
        elif ((plot_len>240) & (plot_len<=320)): 
            day_freq=4
        else:
            day_freq=8
            
  
        if plot_dict['recommended']:
            message="Recommended"
        else:
            message="Example"
  
        newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
        day_position=df.index.to_list()

       # min_stock=df['minimum_stock']    
        minimum_days_stock_holdings=plot_dict['minimum_weeks_stock_holdings']*7*plot_dict['adjusted_average_sales_rate_per_day']
        no_days_sales_as_target=plot_dict['no_weeks_sales_as_target']*7*plot_dict['adjusted_average_sales_rate_per_day']


        fig, ax = plt.subplots()
   
        fig.autofmt_xdate()
        fig.subplots_adjust(top=2,bottom=1.9)
        ax.ticklabel_format(style='plain')
        
        ax.axvline(-day_position[0], ls='--', color='black',label='today')   #color="black")
    #    ax.vlines(-day_position[0], ymin=df.index[-1],ymax=df.index[0],ls='--', color='black', label="today")   #color="black")

       # ax.hlines(y=minimum_days_stock_holdings, xmin=0, xmax=7,data="test",ls='--', color='red')   #color="black")
       # plt.axhline(y=no_days_sales_as_target, ls='--', color='green')   #color="black")
  
        
        df[['stock_units']].plot(xlabel="",use_index=False,kind='bar',align='edge',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
        ax.set_ylabel('Units',fontsize=7)
    
    
    
    
     #   line1=df[['stock_as_batches']].plot(use_index=False,grid=False,xlabel="",kind='line',rot=90,style=["r-","k-"],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
        ax2=df[['stock_as_batches','stock_as_approx_days_left']].plot(use_index=False,grid=True,xlabel="",kind='line',rot=90,style=["m-","k-"],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
 
   
        ax.hlines(y=minimum_days_stock_holdings, xmin=0, xmax=df.shape[0],ls='--', color='red',label="min level ("+str(plot_dict['minimum_weeks_stock_holdings'])+" wks stock)")   #color="black")
        ax.hlines(y=no_days_sales_as_target, xmin=0, xmax=df.shape[0],ls='--', color='green',label="target level ("+str(plot_dict['no_weeks_sales_as_target'])+" wks stock)")   #color="black")
        
      #  ax.right_ax.set_ylabel('batches/ approx days left',fontsize=6)
      #  ax.right_ax.tick_params(axis='y',which='major',labelsize=6)
 
        ax2.set_ylabel('batches/ approx days left',fontsize=6)
        ax2.tick_params(axis='y',which='major',labelsize=6)
       
        
        future_sched_date=plot_dict['todays_date']+timedelta(plot_dict['x'])
 
        fig.legend(title="Priority:"+str(plot_dict["priority_number"])+":"+message+" for Product:"+plot_dict['product_name']+":Today=["+plot_dict['todays_date'].strftime("%A %d/%m/%Y")+"]\n("+str(plot_dict['y'])+" batches scheduled for ["+str(future_sched_date.strftime("%A %d/%m/%Y"))+"] (in today+"+str(plot_dict['x'])+" days, yield/batch="+str(plot_dict['units_per_batch'])+")",ncol=2,title_fontsize=7,fontsize=6,loc='upper center', bbox_to_anchor=(0.5, 1.1))
      
        ax.tick_params(axis='y', which='major', labelsize=6)
     #   ax.tick_params(axis='y', which='right', labelsize=6)
      
        
        new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
        improved_labels = ['{}{}{}'.format(d,calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
        

       # improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
     #   improved_labels=improved_labels[:1]+improved_labels
        improved_labels=[str(day_position[i])+">"+improved_labels[i] if df['date_weekday'].iloc[i] else str(day_position[i]) for i in range(0,df.shape[0])]
        improved_labels=improved_labels[:1]+improved_labels


      #  improved_labels=improved_labels[:1]+improved_labels[::day_freq]
        improved_labels=improved_labels[::day_freq]
     
        
        
        
  
        ax.xaxis.set_major_locator(ticker.MultipleLocator(day_freq))
        ax.xaxis.set_tick_params(which='major', labelcolor='black',
                         pad=1.2)
      
        
        
        ax.set_xticklabels(improved_labels,fontsize=4.5,ha='center')
        self._align_yaxis(ax, 0, ax2, 0)
       # plt.autoscale()
        fig.tight_layout()
        return plot_dict        
    
     



    def make_stock(self,batches,day_position,plot_dict):
        qty=batches*plot_dict['units_per_batch']
        plot_dict['df'].loc[day_position:,'stock_units']+=qty
        plot_dict['df'].loc[day_position:,'stock_as_batches']+=qty/plot_dict['units_per_batch']
        plot_dict['df'].loc[day_position:,'stock_as_approx_days_left']+=qty/plot_dict['adjusted_average_sales_rate_per_day']
        return plot_dict



    def sell_stock(self,qty,day_position,plot_dict):
        plot_dict['df'].loc[day_position:,'stock_units']-=qty
        plot_dict['df'].loc[day_position:,'stock_as_batches']-=qty/plot_dict['units_per_batch']
        plot_dict['df'].loc[day_position:,'stock_as_approx_days_left']-=qty/plot_dict['adjusted_average_sales_rate_per_day']

        return plot_dict
    
  
       # for use in setting up stock levels prior to day zero
    def _make_stock(self,qty,day_position,plot_dict):
        plot_dict['df'].loc[:day_position,'stock_units']+=qty
        plot_dict['df'].loc[:day_position,'stock_as_batches']+=qty/plot_dict['units_per_batch']
        plot_dict['df'].loc[:day_position,'stock_as_approx_days_left']+=qty/plot_dict['adjusted_average_sales_rate_per_day']

        return plot_dict

  
    
 
    def _apply_sales_curve(self,sales_curve,plot_dict):
        df=plot_dict['df']
        average_sales_rate_per_day=plot_dict['adjusted_average_sales_rate_per_day']
        day_position=df.index.to_list()
        curve=sales_curve[:,plot_dict['curve_column']]
     #   print("curve=\n",curve,curve.shape)
        days_of_year=df['day_of_year'].to_list()
      #  print("dp=",day_position,len(day_position))
      #  print("\ndy=",days_of_year,len(days_of_year))
      #  print("days of year",days_of_year)
        for d,x in zip(day_position,days_of_year):
            if d>0:
                plot_dict=self.sell_stock(curve[x-1]*average_sales_rate_per_day,d,plot_dict)
                
        for d,x in zip(day_position[::-1],days_of_year[::-1]):
            if d<0:
                plot_dict=self._make_stock(curve[x-1]*average_sales_rate_per_day,d,plot_dict)
        
        
        return plot_dict









     
    def _create_day_position(self,*,todays_date,start_date,end_date):
        if todays_date < start_date:
            print("todays date < start date error")
            return []
        elif todays_date > end_date:
            print("todays date > end date error")
            return []
        
        right_len=(end_date-todays_date).days
        left_len=(todays_date-start_date).days
      #  print(start_date,todays_date,end_date)
        day_count_list=list(range(-left_len,0))+list(range(0,right_len+1))
      #  print(day_count_list)
        return day_count_list
    

    
    def _create_plot_dict(self,*,product_name,units_per_batch,stock_at_day_zero,adjusted_average_sales_rate,curve_column,minimum_weeks_stock_holdings,no_weeks_sales_as_target,start_date,end_date,todays_date):
        day_count_list=self._create_day_position(todays_date=todays_date,start_date=start_date,end_date=end_date)
        
        df = pd.DataFrame({'date': pd.date_range(start_date, end_date,normalize=True),'day_position':day_count_list,'stock_units':0})
     #   df['date'] = pd.to_datetime(df['dates'].dt.normalize())
        df['date_weekday']=~((df['date'].dt.dayofweek==5) | (df['date'].dt.dayofweek==6))
        df['day_of_year']=df['date'].dt.dayofyear
        df.set_index(['day_position'],inplace=True)
   #     df.drop(['dates'],axis=1,inplace=True)
        #print("df1=\n",df)
        
        
        df['stock_units']=stock_at_day_zero
        df['stock_as_batches']=stock_at_day_zero/units_per_batch
        df['stock_as_approx_days_left']=stock_at_day_zero/adjusted_average_sales_rate        
            
        
        
        plot_dict={}
        plot_dict['df']=df
        plot_dict['product_name']=product_name
        plot_dict['units_per_batch']=units_per_batch
        plot_dict['adjusted_average_sales_rate_per_day']=adjusted_average_sales_rate
        plot_dict['curve_column']=curve_column
        plot_dict['minimum_weeks_stock_holdings']=minimum_weeks_stock_holdings
        plot_dict['no_weeks_sales_as_target']=no_weeks_sales_as_target
        plot_dict['plot_list']=[]
        plot_dict['todays_date']=todays_date
        plot_dict['start_date']=start_date
        plot_dict['end_date']=end_date
   
        return plot_dict
          



#-------------------------------------------------------------------------------
 
    
    def _multiplot(self,plot_dict):
        x=plot_dict['x']
        y=plot_dict['y']
        if plot_dict['recommended']:
            message="_recommended"   
        else:
            message="_example"
      
        plot_dict=self.make_stock(y,x,plot_dict)
    
        plot_dict=self._plot_engine(plot_dict)
        filename=str(plot_dict['priority_number']).zfill(4)+"_"+str(plot_dict['product_name'])+"_"+str(x).zfill(4)+"_"+str(y).zfill(4)+message
        self._save_fig(filename,dd2.dash2_dict['scheduler']['schedule_savedir_plots'])
       # filename_list.append(filename+".png")
        #plt.close()
     #   plot_dict=stock.make_stock(-y,x,plot_dict)
        return #filename_list
        
    
    
    def _resize(self,filename):
        fname = os.path.basename(filename)
        input_image  = imageio.imread(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+fname, format='PNG-FI')
        output_image = cv2.resize(input_image, (944,800))   #(1888, 1600))   #(256, 256))
        imageio.imwrite(dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+fname, output_image, format='PNG-FI')   # 48 bit
        return



    
    def animate_plots(self,*,gif_duration,mp4_fps,plot_output_dir):
        # turn the plots into a animated gif, mp4
        # turn to recommended ones into a gif
     
        filenames=sorted(glob.glob(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+'*recommended.png'))   #, key=os.path.getctime)
        #sorted(glob.glob('*.png'), key=os.path.getmtime)
        #print("f=",filenames)
        print("\n")
        print("Creating gif...")
        images = []
      #  i=1
        lf=len(filenames)
      #  for filename in filenames:
      #      print("Creating gif of",i,"/",lf,"plots....",end="\r",flush=True)
      #      images.append(imageio.imread(filename))        
      #      i+=1
        p_map(self._resize,filenames)
        images=p_map(imageio.imread,filenames)    
            
        imageio.mimsave(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'], images,duration=gif_duration)
        print(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'],"completed\n")
        imageio.mimsave(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'], images,duration=gif_duration)
        print(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'],"completed\n")
        
        
    #---------------------------------------------------
    # only need once
       # imageio.plugins.freeimage.download()
    #---------------------------------------------------------------------
    
        filenames=sorted(glob.glob(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+'*example.png'))   #, key=os.path.getctime)
        lf=len(filenames)
        print("Creating",dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4'])
        # i=0
        # for filename in filenames:
        #     print("Resizing plots of",i,"/",lf,"plots....",end="\r",flush=True)
        #     fname = os.path.basename(filename)
        #     input_image  = imageio.imread(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+fname, format='PNG-FI')
        #     output_image = cv2.resize(input_image, (1888, 1600))   #(256, 256))
        #     imageio.imwrite(dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+fname, output_image, format='PNG-FI')   # 48 bit
        #     i+=1
            
        p_map(self._resize,filenames)
        
       # print("\nCreating test.mp4")
        writer = imageio.get_writer(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4'], fps=mp4_fps)
        writer2 = imageio.get_writer(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4'], fps=mp4_fps)

        i=1
      #  print("Creating gif of",i,"/",lf,"plots....",end="\r",flush=True)
     
        for filename in filenames:
             print("Creating mp4 of",i,"/",lf,"plots....",end="\r",flush=True)
             fname = os.path.basename(filename)
             writer.append_data(imageio.imread(dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+fname))
             writer2.append_data(imageio.imread(dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+fname))

             i+=1
        
      #  writer=p_map(_mp4_write,filenames)
        
        writer.close()
        writer2.close()
        print("\n")
        print(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4']+" completed")
        print(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4']+" completed")

        print("\n")
        return
    


    
    
    def stacker(self,df):
    # take a df of the schedule and fit it efficiency into each day using dd2.dash2_dict['scheduler']['stacker_productivity']  values
    
     #   df['new_dates']=df.index.get_level_values(0)
     #   print("stacker=\n",df)
        # for r1 in df.index:
        #     start_batch=df.iloc[r1,'batches']
        #     print(df)
        #     for r2 in df.index[1:]:
        #         tot_batches=df.loc[r2,'batches']+start_batch
        #         print("r2",r2,tot_batches)
    
                
                
        
        return df

   
    
    
    def display_and_export_final_schedule(self,df,start_schedule_date):
        df['priority']=np.arange(1,df.shape[0]+1)
        df['stacked_date_pos']=np.arange(0,df.shape[0])
        df['stacked_date']=df['scheduled_date'].iloc[0]
        final_schedule=df[['priority','scheduled_date','product','batches','new_stock_will_be_approx','stacked_date','stacked_date_pos']]
     #   final_schedule.drop(["scheduled_date","priority"],axis=1,inplace=True)
        final_schedule.to_pickle("fs.pkl",protocol=-1)
   
       # final_schedule.set_index(['scheduled_date','priority'],drop=False,inplace=True)
    
 
        
        print("\nBefore stacking - scheduling start date=",start_schedule_date.strftime("%A %d/%m/%Y"))            
        print(final_schedule)
        
        final_schedule=self.stacker(final_schedule)
          
        print("\nAfter stacking - scheduling start date=",start_schedule_date.strftime("%A %d/%m/%Y"))            
        print(final_schedule)
        
        
        sheet_name = 'Sheet1'
        writer = pd.ExcelWriter(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_save_excel'],engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')
     
        final_schedule.to_excel(writer,sheet_name=sheet_name,index=False)
        
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
      #   money_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': False})
      #   total_fmt = workbook.add_format({'num_format': '$#,##0', 'bold': True})
     
       #  worksheet.set_column('E:ZZ', 12, money_fmt)
       #  worksheet.set_column('D:D', 12, total_fmt)
       #  worksheet.set_row(3, 12, total_fmt)
     
             # Apply a conditional format to the cell range.
    #     worksheet.conditional_format('B2:B8', {'type': '3_color_scale'})
       #  worksheet.conditional_format('E5:ZZ1000', {'type': '3_color_scale'})
     
         # Close the Pandas Excel writer and output the Excel file.
        writer.save()      
        
       
        return 



#def plot_combinations(plot_dict):
    #print(schedule[['product','last_manu_date','qtyinstock','adjusted_ave_qty_sold_per_week','stock_needed','weeks_since_last_made']].to_string(),"\n") 
    
    #     visualise schedule
    # turn the stocks df into a list of list of matplotlib figures that can be scrolled in 2D
    # of manu date and manu quant (as batches)
    #
    #
    
    # product                                        SJ300  ...                                  BDC170
    # format_type                                        0  ...                                       2
    # pg_type                                            0  ...                                       1
    # productgroup                    Jams 250ml glass jar  ...  Traditional condiments 150ml glass jar
    # pg                                                10  ...                                      16
    # lastsalesdate                    2020-12-03 00:00:00  ...                     2020-12-02 00:00:00
    # qtyinstock                                     20138  ...                                     632
    # ave_qty_sold_per_week                        15090.4  ...                                 148.827
    # no_weeks_sales_as_target                           5  ...                                       5
    # minimum_weeks_stock_holdings
    # target_holdings                                86591  ...                                     984
    # stock_needed                                   55314  ...                                     112
    # weeks_to_stock_out                               1.3  ...                                     4.2
    # weeks_to_minimum                                   2
    # urgency                                      1.96019  ...                                 1.96019
    # effic                                        2.00041  ...                              0.00191321
    # last_manu_date                   2020-11-26 00:00:00  ...                     2020-10-06 00:00:00
    # weeks_since_last_made                        1.28571  ...                                 8.57143
    # demand_factor                                1.14763  ...                                 1.32263
    # adjusted_ave_qty_sold_per_week               17318.2  ...                                 196.843
    # priority                                      3.9606  ...                                 1.96211
    # format_priority                              2.28389  ...                                 1.96221
    # format_type_priority                         2.17208  ...                                 1.97562
    
    #   To visualise, we need 4 pieces of info and a timeline from last manufacturing date, today in       elif ((plot_len>80) & (plot_len<=120)): 
    # 1 . stock level at the moment
    # 2.  the average unit sales rate      declining with a x axis of dates until it intersects with the x axis at zero stock
    # 3.  how much is being made
    # 4.  When it is being made
    #
    # from this we can see the new x axis intecept date
    # 
    # ultimately, I want to be able to use pyglet to create a slider that allows an individual product or product group
    # manufacturing run to be adjusted with the mouse and updated in real time to see the effect 
    #
    # some formulas:
    #   stocks(t) = stocks(today)-sales_rate_per_day(t.days - today.days)
    #   plus any manufactured
    #   
    
         
    
    
    #--------------------------------------------------------------------------------------------------------------
    
    
       
    
    def visualise_schedule(self,plot_output_dir):
    #    os.chdir("/home/tonedogga/Documents/python_dev")
     #   cwdpath = os.getcwd()
        
      #  sv=stock_plot_starter()
      #  stock=plot_schedule()
      #  sc=scheduler.scheduler_curves()
                
        curves=self.create_curves()
     
       
        todays_date = dt.date.today()   #time.now()  #.normalize()   #dt.date.today()
        other_todays_date=dt.datetime.now()
        maximum_days_into_schedule_future=80
       # start_schedule_day_offset=0
         
        date_choices=pd.bdate_range(todays_date,todays_date+timedelta(maximum_days_into_schedule_future),normalize=True)
        date_choices_dict={}
        
        for d in date_choices:
            i=int((d-other_todays_date).days+1)
            date_choices_dict[i]=d
 
        if True: #we are testing
            start_schedule_day_offset=5
            start_schedule_date=date_choices_dict[int(start_schedule_day_offset)]
        else:    
            #print("dcd=",date_choices_dict)    
            incorrect=True
            while incorrect:    
                start_schedule_day_offset=input("Days ahead from today to start scheduling manufacturing?")
                if isinstance(start_schedule_day_offset,str):
                    try:
                        start_schedule_day_offset=int(start_schedule_day_offset)
                    except:
                        pass
                    else:
                        if isinstance(start_schedule_day_offset,int):
                            try:
                                start_schedule_date=date_choices_dict[int(start_schedule_day_offset)]
                            except KeyError:
                                print("start date is weekend or invalid.")
                                pass
                            else:
                                incorrect=False
              
            
          
            
        start_schedule_day_offset= int(start_schedule_day_offset)  
        saved_start_schedule_date= start_schedule_date   
       # print("scheduling start day offset=",start_schedule_day_offset)            
        print("\nscheduling start date=",start_schedule_date.strftime("%A %d/%m/%Y"))            
     
        days_back_from_today_if_last_manu_is_too_long_ago=10
        days_to_potentially_schedule_ahead=40
        
      #  output_dir=cwdpath
        schedule=pd.read_pickle(dd2.dash2_dict['scheduler']['schedule_savedir']+dd2.dash2_dict['scheduler']['schedule_savefile'])
       # schedule=read_schedule.copy()
        #print(stock_line['pg'])
        # stock level equation is    x (dates) = starting stock - sales rate/day * x
        
        #print(schedule)
        
        product_code_priority=schedule['product'].to_list()
        pcp_len=len(product_code_priority)
        print("\n")  #"ppp",product_code_priority)
        
       # schedule['days_to_manu']=np.arange(start_schedule_day_offset,start_schedule_day_offset+schedule.shape[0])
        
        schedule['yield_per_batch']=0
       # pg_list=dd2.dash2_dict['scheduler']['pg_yield'][schedule['pg']].to_list()
       # print("pg_list",pg_list)
     #   print("schedule before0=\n",schedule)
     #   schedule_day=start_schedule_day_offset
      #  start_schedule_date=todays_date+timedelta(schedule_day) 
        for i in schedule.index:
      #      print("r=",r)
            stock_line=schedule.loc[i,:]
            schedule.loc[i,'yield_per_batch']=dd2.dash2_dict['scheduler']['pg_yield'][stock_line['pg']]
     
      #  schedule['yield_per_batch']=dd2.dash2_dict['scheduler']['pg_yield'][str(schedule['pg'])]
        
     
      #  print("schedule before1=\n",schedule)
        
    
      #  schedule['batches']
        schedule['scheduled_date']=pd.bdate_range(start_schedule_date,start_schedule_date+timedelta(round(pcp_len*(7/5),0)+1),normalize=True)[:pcp_len] 
        schedule['days_to_manu']=(pd.to_datetime(schedule['scheduled_date'])-pd.to_datetime(todays_date)).dt.days.astype('int16')   #pd.to_datetime(todays_date)
        schedule['day_of_week_no']=schedule['scheduled_date'].dt.dayofweek
      #  print("schedule before2=\n",schedule)
    
        schedule['ideal_batch_size']=(schedule['stock_needed']+((schedule['adjusted_ave_qty_sold_per_week']/7)*schedule['days_to_manu']))/schedule['yield_per_batch']
        schedule['batches']=((np.around((schedule['ideal_batch_size']/5),0)+1)*5).astype(np.int32)
        schedule['new_stock_will_be_approx']=np.around(schedule['qtyinstock']+schedule['batches']*schedule['yield_per_batch']-((schedule['adjusted_ave_qty_sold_per_week']/7)*schedule['days_to_manu']),0).astype(np.int32)
      #  schedule['day_of_week']=schedule['day_of_week_no'].format("%W")
      #  print("schedule before3=\n",schedule)
        
        
        
        
        
        schedule_day=start_schedule_day_offset
        row=0
        plot_dict_list=[]
        for p in product_code_priority:
        
            stock_line=schedule.iloc[row]   #[schedule['product']==p]
            print("Visualising scheduling choices for product:",p," (",row+1,"/",len(product_code_priority),")                                  ",end="\r",flush=True)
            sales_rate_per_day=stock_line['adjusted_ave_qty_sold_per_week']/7
            
            recommended_batch_size=stock_line['batches']
            recommended_manu_day_offset=int(stock_line['days_to_manu'])
            
            start_date=stock_line['last_manu_date'].date()  #.normalize()    #.strftime("%d/%m/%Y")
            #print(start_date)
            #end_date=start_date+pd.offsets.Day(36)
            end_date=start_date+timedelta(maximum_days_into_schedule_future)
    
            if todays_date>(end_date-timedelta(days_to_potentially_schedule_ahead)):
                start_date=todays_date-timedelta(days_back_from_today_if_last_manu_is_too_long_ago)
                end_date=start_date+timedelta(maximum_days_into_schedule_future)
            
                
            #print(curves[:,2:])
            
            #print(day_count_list,len(day_count_list))
            # for jams
            
            plot_dict=self._create_plot_dict(product_name=p,units_per_batch=dd2.dash2_dict['scheduler']['pg_yield'][stock_line['pg']],stock_at_day_zero=stock_line['qtyinstock'],adjusted_average_sales_rate=sales_rate_per_day,curve_column=dd2.dash2_dict['scheduler']['pg_type'][stock_line['pg']],minimum_weeks_stock_holdings=stock_line['minimum_weeks_stock_holdings'],no_weeks_sales_as_target=stock_line['no_weeks_sales_as_target'],start_date=start_date,end_date=end_date,todays_date=todays_date)
            
            plot_dict=self._apply_sales_curve(curves[:,2:],plot_dict)
    
    
          #  for days_in_future in range(6,36,6) :  #plot_dict['df'].shape[0]):
              
            for batch_size in [5,10,20,30,40,50,60,70,80,90,100,110,120,recommended_batch_size]:
                start_schedule_date=dt.date.today()+timedelta(schedule_day)
                while ((start_schedule_date.weekday==5) | (start_schedule_date.weekday==6)):
                    # avoid weekends
                    schedule_day+=1
                    start_schedule_date=dt.date.today()+timedelta(schedule_day)
     
                try:
                    if plot_dict['df'].loc[schedule_day,'date_weekday']:
                      #  print("building a plot list:",p,x,y) 
                        if batch_size==recommended_batch_size:
                            plot_dict['x']=recommended_manu_day_offset
                            plot_dict['recommended']=True
                        else:    
                            plot_dict['x']=schedule_day
                            plot_dict['recommended']=False
                        plot_dict['y']=batch_size
                        
                        plot_dict['priority_number']=row+1
                   #     plot_dict['product_code']=p
                        plot_dict_list.append(plot_dict.copy())
                    else:
                        pass
                       # print("weekend",x,y)
                except:
                    pass
        
            schedule_day+=1
            row+=1        
            
       
      
        print("\nFinished.")
        
       # recommended=schedule[['product','days_to_manu','batches']].copy()   #.to_list()
       # print("recommended",recommended)
      
   #+-----------------------------------------------------------------------------------------------------------
             
      
        # clear old plots
        files = glob.glob(dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+'*.png')
      #  print("1files to del:",files)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("file delete Error: %s : %s" % (f, e.strerror))
                
                
           # clear old plots
        files = glob.glob(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+'*.png')
      #  print("2files to del:",files)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("file delete Error: %s : %s" % (f, e.strerror))
        
                
      #+-----------------------------------------------------------------------------------------------------------
              
                
        
        print("\nPlotting...")
        p_map(self._multiplot,plot_dict_list)                
        plt.close('all')
        
    
    
      
        
        self.display_and_export_final_schedule(schedule,saved_start_schedule_date)
        
        self.animate_plots(gif_duration=4,mp4_fps=2,plot_output_dir=plot_output_dir)
        
        
        
        
        
        
        return
    
    #---------------------------------------------------------------------------------------------------------------
    
    
      
