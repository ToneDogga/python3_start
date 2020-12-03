#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:02:59 2020

@author: tonedogga
"""

import numpy as np
import pandas as pd
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
    
        return np.c_[urgency_curve,efficiency_curve,jams_condiments_and_mealbases_curve,sauces_and_dressings_curve]
 
    
    
    
    def display_curves(self,curves):
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
        plt.show()
        
        return
    
   
###########################################################################################################################    
   
    
  
    
  
class schedule(object):
    def __init__(self):
        
        self.product_group_mask=["10","11","12","13","14","15","16","17"]
        self.productgroup_dict={
                    
                   # 0 for winter product group
                   # 1 for summer product group 
                   # that means they use different demand curves
                   
                   "10":("Jams 250ml glass jar",0),
                   "11":("Sauce 300ml glass bottle",1),
                   "12":("Dressings 300ml glass bottle",1),
                   "13":("Condiments 250ml glass jar",0),
                   "14":("Meal bases 250ml glass jar",0),
                   "15":("Condiments for cheese 150ml glass jar",1),
                   "16":("Traditional condiments 150ml glass jar",1),
                   "17":("Mustards 150ml glass jar",0)
                  }


              
              
        pass
    
    
    def load_SOH(self,filename):
         return pd.read_pickle(filename)        
    
    
    "RFR_order_predict_model_savefile":"RFR_order_predict_model.pkl",
        pass
    
    
    
    
    
    def load_average_weekly_sales(self,filename):
        pass
    
    
    def _winter_or_summer_product(self,pg):
        if pg==
    
        
  
    
s=schedule()    
sc=scheduler_curves()


curves=sc.create_curves()
sc.display_curves(curves)
s.load_SOH(dd2.dash2_dict['production']['save_dir']+dd2.dash2_dict['production']['SOH_savefile'])