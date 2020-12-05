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


class scheduler_curves(object):
    def __init__(self):
                
        self.urgency_maxx=10
        self.urgency_maxy=2
        self.urgency_cross=4
        
        
        self.efficiency_maxx=50
        self.efficiency_maxy=2.1
        self.efficiency_cross=7
        
        self.total_days_in_year=366
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
    
   
###########################################################################################################################    
   
    
  
    
  
class schedule(object):
    def __init__(self):           
         pass
    
    
    def _load_SOH(self,filename):
         stock_on_hand_df=pd.read_pickle(filename)  
         stock_on_hand_df['pg']=stock_on_hand_df['productgroup']
         stock_on_hand_df["pg"].replace(dd2.dash2_dict['scheduler']['productgroup_dict'], inplace=True)
         stock_on_hand_df=stock_on_hand_df[stock_on_hand_df['pg'].isin(dd2.dash2_dict['scheduler']['productgroup_mask'])]
         stock_on_hand_df['pg_type']=stock_on_hand_df['productgroup'] 
         stock_on_hand_df["pg_type"].replace(dd2.dash2_dict['scheduler']['productgroup_type'],inplace=True)
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
        s=schedule()    
        sc=scheduler_curves()
        
        target_weeks_of_stocks=5
        estimated_days_to_make_from_scheduling=1
        urgency_weight=1.0
        efficiency_weight=1.0
        minimum_onehundred_percent_run=40000
        
        curves=sc.create_curves()
      #  print(curves)
        sc.display_curves(curves,output_dir)
       # print("curves_df=\n",curves_df)
        
        stock_on_hand_df=self._load_SOH(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['SOH_savefile'])
      #  print("SOH=\n",stock_on_hand_df)  

        production_made_df=self._load_PM(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['PM_savefile'])
      #  print("PM=\n",production_made_df)  

        latest_date_dict=self._find_latest_manu_date(production_made_df)      


        weekly_sales_df,latest_date=s.load_average_weekly_sales(dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['savefile'])
        #print("weekly sales df=\n",weekly_sales_df)
        stocks=pd.merge(stock_on_hand_df,weekly_sales_df,on=["product"])   #['product']=='product']]
        stocks.rename(columns={'qty':"ave_qty_sold_per_week"},inplace=True)
    
        stocks['no_weeks_sales_as_target']=target_weeks_of_stocks
        stocks['target_holdings']=round(stocks['ave_qty_sold_per_week']*target_weeks_of_stocks,0).astype(np.int32)
        stocks['stock_needed']=stocks['target_holdings']-stocks['qtyinstock']
        stocks['weeks_to_stock_out']=round(stocks['qtyinstock'] / ((stocks['ave_qty_sold_per_week'])),1)
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
           curv=s.calc_urgency(int(w/7),estimated_days_to_make_from_scheduling,curves)
           stocks['urgency'].iloc[i]=curv[0]
           
           
        sn_list=stocks['stock_needed']
        for i,e in enumerate(sn_list):   
         #  print("effic",i,e) 
           curv2=s.calc_efficiency(e,minimum_onehundred_percent_run,curves)
           stocks['effic'].iloc[i]=curv2
          
        sd={}    
        product_list=list(set(list(stocks['product']))) 
      #  print("product_list=",product_list)
        for p in product_list:
            stocks[stocks['product']==p].sort_index(ascending=True)
            sd[p]=5  #(pd.to_datetime("today")-pd.to_datetime(stocks['lastsalesdate'].iloc[-1]))    #.strftime("%d/%m/%Y")
            
          
        curv3=s.calc_demand(estimated_days_to_make_from_scheduling,curves)  
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


  