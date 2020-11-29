#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:11:50 2020

@author: tonedogga
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

import datetime as dt
from datetime import date
from datetime import timedelta
import calendar
import xlsxwriter

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
import matplotlib.pyplot as plt


from p_tqdm import p_map,p_umap

import dash2_dict as dd2



class scan_class(object):
   def __init__(self):
       #self.scan_init="scan_init called"
      # print(self.scan_init)
       pass
   
 
    
    
   def _write_excel(self,df,filename):
        sheet_name = 'Sheet1'
        writer = pd.ExcelWriter(filename,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')   #excel_file, engine='xlsxwriter')     
        df.to_excel(writer,sheet_name=sheet_name,header=False,index=False)    #,engine='xlsxwriter',datetime_format='dd/mm/yyyy',date_format='dd/mm/yyyy')     
        writer.save()
        return

  
   
   
   def load_scan_data_from_excel(self,in_dir,scan_data_files,scan_data_filesT):
        scan_data_files=[in_dir+f for f in scan_data_files]
        scan_data_filesT=[in_dir+f for f in scan_data_filesT]
 
        print("load scan data",scan_data_files,"\n",scan_data_filesT)
       # df=pd.DataFrame([])
             
        
        count=1
        for scan_file,scan_fileT in zip(scan_data_files,scan_data_filesT):
          #  column_count=pd.read_excel(scan_file,-1).shape[1]   #count(axis='columns')
            #if dd.dash_verbose:
            print("Loading...",scan_file)   #,scan_fileT)   #,"->",column_count,"columns")
          
           # convert_dict={col: np.float64 for col in range(1,column_count-1)}   #1619
           # convert_dict['index']=np.datetime64
        
            if count==1:
     #           df=pd.read_excel(scan_file,-1,dtype=convert_dict,index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
                dfT=pd.read_excel(scan_file,-1,header=None,engine='xlrd',dtype=object)    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
    
                self._write_excel(dfT.T,scan_fileT)   #,engine='xlsxwriter')  #dd2.dash2_dict['scan']['save_dir'])
    
                df=pd.read_excel(scan_fileT,-1,header=None,index_col=[1,2,3,4,5,6,7,8,9,10,11,12,13],engine='xlrd',dtype=object)  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
            else:
           #     print(convert_dict)
             #   del df2
                dfT=pd.read_excel(scan_file,-1,header=None,engine='xlrd',dtype=object)    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
                
                self._write_excel(dfT.T,scan_fileT)   #,engine='xlsxwriter')  #dd2.dash2_dict['scan']['save_dir'])
    
         
                df2=pd.read_excel(scan_fileT,-1,header=None,index_col=[1,2,3,4,5,6,7,8,9,10,11,12,13],engine='xlrd',dtype=object) #,na_values={"nan":0}) 
            
                df=pd.concat([df,df2],axis=0)   #,ignore_index=True)   #levels=['plotnumber','retailer','brand','productgroup','product','variety','plottype','yaxis','stacked'])   #,keys=[df['index']])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
              #  del df2
           # print(df)
            count+=1 
        df.index.set_names('plotnumber', level=0,inplace=True)
        df.index.set_names('retailer', level=1,inplace=True)
        df.index.set_names('brand', level=2,inplace=True)
        df.index.set_names('productgroup', level=3,inplace=True)
        df.index.set_names('product', level=4,inplace=True)
        df.index.set_names('variety', level=5,inplace=True)
        df.index.set_names('plottype', level=6,inplace=True)
        df.index.set_names('plottype1', level=7,inplace=True)
        df.index.set_names('plottype2', level=8,inplace=True)
        df.index.set_names('plottype3', level=9,inplace=True)
        df.index.set_names('sortorder', level=10,inplace=True)
        df.index.set_names('colname', level=11,inplace=True)
        df.index.set_names('measure', level=12,inplace=True)
       
        
       # a = df.index.get_level_values(0).astype(str)
       # b = df.index.get_level_values(6).astype(str)
    
       # df.index = [a,b]
        
       
         
        df=df.T
     #   print("df0=\n",df)
        df['date']=df.iloc[:,1]
     #   print("df1=\n",df)
        colnames=df.columns.levels[0].tolist()
      #  print("colnames=",colnames)
        
    
        colnames = colnames[-1:] + colnames[:-3]
        df = df[colnames]
     #   print("df2=\n",df)
    
        df = df[df.index != 0]
      #  print("df3=\n",df)
    
        df.set_index('date',drop=False,append=True,inplace=True)
        df=df.reorder_levels([1,0])
       # print("df4=\n",df)
     
        df=df.droplevel(1)
        #print("df5=\n",df)
        df=df.drop('date',level='plotnumber',axis=1)
       # print("df6=\n",df)
    
        df.fillna(0.0,inplace=True)
         #   df=df.drop('date',level='plotnumber')
    
       # print("df6.cols=\n",df.columns)
        df=df.T
        df.sort_index(axis=0,inplace=True,level='sortorder',ascending=True)
     #   write_excel2(df,"testdf.xlsx")
        
     #   print("df4=\n",df,"\n",df.T)
        return df
    

     

    
   def load_scan_monthly_data_from_excel(self,in_dir,scan_data_files,weeks_back):
        #scan_data_files=[in_dir+f for f in filename_list]
        print("Load scan monthly data from excel",in_dir,scan_data_files)
        count=1
        for scan_file in scan_data_files:
           #  column_count=pd.read_excel(scan_file,-1).shape[1]   #count(axis='columns')
         #if dd.dash_verbose:
            print("Load scan monthly data.  Loading...",scan_file)   #,scan_fileT)   #,"->",column_count,"columns")
         
         # convert_dict={col: np.float64 for col in range(1,column_count-1)}   #1619
         # convert_dict['index']=np.datetime64
         
            if count==1:
              #           df=pd.read_excel(scan_file,-1,dtype=convert_dict,index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
                 dfT=pd.read_excel(in_dir+scan_file,-1,header=None,engine='xlrd',dtype=object)   #,converters = {'date': lambda x: pd.to_datetime(x, errors='coerce')})    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
            
                 self._write_excel(dfT.T,dd2.dash2_dict['scan']["save_dir"]+"T"+scan_file)  #,dd2.dash2_dict['scan']["save_dir"])
            
                 df=pd.read_excel(dd2.dash2_dict['scan']["save_dir"]+"T"+scan_file,-1,header=None,index_col=[0,1,2],engine='xlrd',dtype=object)   #,converters = {0: lambda x: pd.to_datetime(x, format="%d/%m/%Y", errors='coerce')})  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
           #      df=pd.read_excel(dd2.dash2_dict['scan']["save_dir"]+"T"+scan_file,-1,header=None,engine='xlrd',dtype=object,converters = {'date': lambda x: pd.to_datetime(x, format="%d/%m/%Y", errors='coerce')})  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
     
            else:
             #     print(convert_dict)
              #   del df2
                 dfT=pd.read_excel(in_dir+scan_file,-1,header=None,engine='xlrd',dtype=object)   #,converters = {'date': lambda x: pd.to_datetime(x, errors='coerce')})    #,index_col=[1,2,3,4,5,6,7,8,9,10,11])  #,na_values={"nan":0})   #index_col=0)   #,header=[0,1,2])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
                 
                 self._write_excel(dfT.T,dd2.dash2_dict['scan']["save_dir"]+"T"+scan_file)   #,dd2.dash2_dict['scan']["save_dir"])
            
                 df2=pd.read_excel(dd2.dash2_dict['scan']["save_dir"]+"T"+scan_file,-1,header=None,index_col=[0,1,2],engine='xlrd',dtype=object)  #,converters = {0: lambda x: pd.to_datetime(x, format="%d/%m/%Y", errors='coerce')}) #,na_values={"nan":0}) 
           
   
           
            
              #   df2=pd.read_excel(dd2.dash2_dict['scan']["save_dir"]+"T"+scan_file,-1,header=None,engine='xlrd',dtype=object,converters = {'date': lambda x: pd.to_datetime(x, format="%d/%m/%Y", errors='coerce')}) #,na_values={"nan":0}) 
                 df2=df2.iloc[1:,:]
             #    print("df2=\n",df2)
                 df=pd.concat([df,df2],axis=0)   #,ignore_index=True)   #levels=['plotnumber','retailer','brand','productgroup','product','variety','plottype','yaxis','stacked'])   #,keys=[df['index']])  #,skip_rows=3)  #[column_list]   #,names=column_list)   #,sheet_name="AttacheBI_sales_trans",use_cols=range(0,16),verbose=True)  # -1 means all rows   #print(df)
               #  del df2
            # print(df)
            count+=1 
      
        
      #  print("2df=\n",df)
        
        df.index.set_names('retailer', level=0,inplace=True)
        df.index.set_names('product', level=1,inplace=True)
        df.index.set_names('measure', level=2,inplace=True)
        df=df.T
        df = df.loc[:,~df.columns.duplicated()]
      #  print("df2=\n",df)
        df.index=df.iloc[:,0]
        df.index.set_names('date',inplace=True)
    
         
        df = df.iloc[-weeks_back:,1:]
     
        df.fillna(0.0,inplace=True)
      #  print("lsmd=\n",df)
        df=df[(df.index.isnull()==False)]
        df=df.loc[:, (df != 0.0).any(axis=0)]
        return df
    
    
        
    
     
       
     
   def _plot_chart(self,scan_pass,count,extra_names,output_dir):
        scan_df=scan_pass.copy(deep=True)
       #print("sd=\n",scan_df,"\n",scan_df.T)
        week_freq=8
      #  scan_df['changedate']=pd.to_datetime(scan_df['date']).strftime("%Y-%m").to_list()
     #   scan_df['date']=pd.to_datetime(scan_df.index).strftime("%Y-%m").to_list()
      #  df=df[(df.date.isnull()==False)]
        scan_df['date']=pd.to_datetime(scan_df.index,format="%Y-%m",exact=False).to_list()
       # scan_df['date']=pd.to_datetime(scan_df.index,format="%Y-%m",exact=True).to_list()
    
    
        newdates = pd.to_datetime(scan_df['date']).apply(lambda date: date.toordinal()).to_list()
      #  print("nd=",newdates)   
     
        
        fig, ax = pyplot.subplots()
        fig.autofmt_xdate()
        ax.ticklabel_format(style='plain')
        scan_df=scan_df.iloc[:,:-1]
        scan_df.plot(xlabel="",grid=True,ylabel="UPSPW index smoothed",ax=ax)
    
        plt.legend(loc='upper left',title="",fontsize=6,title_fontsize=5, bbox_to_anchor=(0.3, 1.1))
        plt.title("Scanned_"+extra_names+"_"+str(count), x=0, y=1)
        new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #[::week_freq] ]  #ax.get_xticks()]
      #  print("new labels=",new_labels)
        improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
        
      #  improved_labels=improved_labels[:1]+improved_labels[::week_freq]
      #  print("scan_df=\n",scan_df)
    
        improved_labels=improved_labels[:1]+improved_labels[week_freq+1::week_freq]
     #   print("improived labels=",improved_labels)
       
        self._save_fig("Scanned_monthly_"+extra_names+"_"+str(count),output_dir)
      
     #   ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
     #   ax.set_xticklabels(improved_labels,fontsize=7)
      #  plt.show()
        plt.close()
     #   plt.close()       
       # plt.tight_layout()
    
    
        return
    
    
    
   def _scale(self,scan_df):
     #   print(scan_df)
        smooth_weeks=6
        scan_df.replace(0,np.nan,inplace=True)
        scan_df=scan_df.rolling(smooth_weeks,axis=0).mean()
      #  print(scan_df)
        scaling=(100/scan_df.iloc[smooth_weeks-1,:]).to_list() 
        i=0
        for s in scaling:
            scan_df.iloc[:,i]*=s
            i+=1
     #   scan_df.iloc[:,column]=scan_df.iloc[:,column]*scaling
     #       column+=1
     #   print("s;=:",scaling,len(scaling))
        return scan_df
    
    
    
    


    
   def plot_scan_monthly_data(self,scan_monthly_df,output_dir):
        
        print("Plotting UPSPW indexes for all scanned products to",output_dir)
        jump=5
        for r in range(0,scan_monthly_df.shape[1],jump):
            self._plot_chart(scan_monthly_df.iloc[:,r:r+jump],int(r/jump),"",output_dir)
     #   print("Finished plotting absolute.")
        
        scale_df=self._scale(scan_monthly_df)
        jump=5
        for r in range(0,scale_df.shape[1],jump):
            self._plot_chart(scale_df.iloc[:,r:r+jump],int(r/jump)+10000,"",output_dir)
     #   print("Finished plotting relative.")
        return   
   
    
 
 

    
   def plot_scan_monthly_dict(self,scan_monthly_dict,output_dir):
        
        for p in scan_monthly_dict.keys():
            print("Plotting scan data monthly for",p,"scan query to",output_dir)
            scan_monthly_df=scan_monthly_dict[p].copy()
            jump=5
            for r in range(0,scan_monthly_df.shape[1],jump):
                self._plot_chart(scan_monthly_df.iloc[:,r:r+jump],int(r/jump),p,output_dir)
     #       print("Finished plotting absolute.")
            
            scale_df=self._scale(scan_monthly_df)
            jump=5
            for r in range(0,scale_df.shape[1],jump):
                self._plot_chart(scale_df.iloc[:,r:r+jump],int(r/jump)+10000,p,output_dir)
      #      print("Finished plotting relative.")
        return   
   
 
    
 
        
   def preprocess_monthly(self,df):
       # rename columns
     #  print("preprocess scan monthly df")
   #    df=df.T
   #    df=df.assign(retailer_number=0).set_index('retailer_number', append=True)
   #    df=df.assign(brand_number=0).set_index('brand_number', append=True)
   #    df=df.assign(product_number=0).set_index('product_number', append=True)
   #    df=df.reorder_levels(['retailer_number','product_number','brand_number','retailer','product','measure'],axis=0)
   #    df=df.T
    #    available_products=list(set(df.columns.get_level_values('product')))
    #    available_retailers=list(set(df.columns.get_level_values('retailer')))
    #    available_brands=list(set(df.columns.get_level_values('brand_number')))       
    # #      available_measures=list(set(df.columns.get_level_values('measure'))))
       df=self._scan_monthly_name_finder(df)
       return df    #,available_retailers,available_products,available_brands
    
  
     

       
  
    
   def save(self,scan_df,save_dir,savefile):
       print("save scan_df to ",save_dir+savefile)
       os.makedirs(save_dir, exist_ok=True)
       if isinstance(scan_df,pd.DataFrame):
           if not scan_df.empty:
          #     scan_df=pd.DataFrame([])
          # else:    
               scan_df.to_pickle(save_dir+savefile,protocol=-1)  
               return True
           else:
               return False
       return False
           
 
  #     return("price save outfile")
       
   def load(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
        print("load scan data from ",save_dir+savefile)

        my_file = Path(save_dir+savefile)
        if my_file.is_file():
            df=pd.read_pickle(save_dir+savefile)
            return df
        else:
           print("load scan data error.")
           return


            
    
   def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
   
    
   def _scan_monthly_name_finder(self,df):
      # look at the product names and deduce the variety type and brand number
       print("scan monthly name finder")
    #   print("1",df) 
       df=df.rename(dd2.retailers_name_dict,level='retailer',axis='columns')
       lvr=df.columns.get_level_values('retailer')
       lvp=df.columns.get_level_values('product')
       lvm=df.columns.get_level_values('measure')
       
       df=df.T
       df=df.assign(retailer_number=lvr).set_index('retailer_number', append=True)
       df=df.T 
  
       df=df.rename(dd2.retailers_number_dict,level='retailer_number',axis='columns')
 
       new_lvb=[dd2.brand_number_dict[k.lower()[:6]] if (k.lower()[:6] in dd2.brand_number_dict) else 0 for k in lvp]
       new_lvm=[dd2.measure_number_dict[k] if k in dd2.measure_number_dict else 0 for k in lvm]
 
        # new_lvp=[dd2.variety_number_dict[k.lower()[:6]] if k.lower()[:6] in dd2.brand_number_dict else 0 for k in lvp]
     
     #  print("new_lvb",new_lvb,len(new_lvb))
    #   print("new_lvm",new_lvm,len(new_lvm))
       
       
       df=df.T
       df=df.assign(brand_number=new_lvb).set_index('brand_number', append=True)
     #  df=df.assign(product_number=lvp).set_index('product_number', append=True)
      # df=df.T
  
    #   df=df.T
       df=df.assign(measure_number=new_lvm).set_index('measure_number', append=True)
     #  df=df.assign(product_number=lvp).set_index('product_number', append=True)
       df=df.T
       df=df.rename(dd2.measure_number_dict,level='measure_number',axis='columns')
 
  
    
       df=df.reorder_levels(['retailer_number','retailer','brand_number','product','measure_number','measure'],axis=1)
 
     #  print("new df\n",df)
       
       
       available_products=list(set(df.columns.get_level_values('product')))
       available_retailers=list(set(df.columns.get_level_values('retailer')))
       available_brands=list(set(df.columns.get_level_values('brand_number')))   
       available_meaures=list(set(df.columns.get_level_values('measure')))   
      
    #   print("available retailers=",available_retailers,"\navailable products=",available_products)   #,"\navailable brands",available_brands)

   
     #  print("scan_df=\n",df,"\n",df.T)
   
       return df
   
       
       
    
   def _multiple_slice_scandata(self,df,query):
        new_df=df.copy(deep=True)
        for q in query:
            
            criteria=q[1]
         #   print("key=",key)
         #   print("criteria=",criteria)
            ix = new_df.index.get_level_values(criteria).isin(q)
            new_df=new_df[ix]    #.loc[:,(slice(None),(criteria))]
        new_df=new_df.sort_index(level=['sortorder'],ascending=[True],sort_remaining=True)   #,axis=1)
    
      #  write_excel2(new_df,"testdf2.xlsx")
        return new_df
            
    
   # def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
   #      os.makedirs(output_dir, exist_ok=True)
   #      path = os.path.join(output_dir, fig_id + "." + fig_extension)
   #    #  print("Saving figure", fig_id)
   #      if tight_layout:
   #          plt.tight_layout()
   #      plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
   #      return
    
    
    
   def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
   
      
        
   def _plot_brand_index(self,tdf,output_dir,y_col,col_and_hue,savename):    
        tdf=tdf.astype(np.float64)
     
        tdf=self._add_trues_and_falses(tdf,col_and_hue[0])
        tdf=self._add_trues_and_falses(tdf,col_and_hue[1])
        
        date=pd.to_datetime(tdf.index).strftime("%Y-%m-%d").to_list()
     #   print("date=",date)
        tdf['date']=date
        tdf['dates'] = pd.to_datetime(tdf['date']).apply(lambda date: date.toordinal())
        fig, ax = pyplot.subplots()
        ax.set_xlabel("",fontsize=8)
        sns.set(font_scale=0.6)
     #   sns.lmplot(x='date_ordinal', y='coles_beerenberg_jams_off_promo_scanned', col='coles_bonne_maman_jams_on_promo',hue='coles_st_dalfour_jams_on_promo',data=tdf)   #,color="green",label="")
        sns.lmplot(x='dates', y=y_col, col=col_and_hue[0],hue=col_and_hue[1],data=tdf,legend=False)   #,color="green",label="")
        ax=plt.gca()
        #sns.regplot(x='date_ordinal', y='coles_beerenberg_jams_off_promo_scanned',data=tdf,color="green",marker=".",label="")
        ax.set_xlabel("",fontsize=8)
        plt.legend(loc='upper left',title=col_and_hue[1],fontsize=8,title_fontsize=8)
        new_labels = [dt.date.fromordinal(int(item)) for item in ax.get_xticks()]
      #  print("new_labels=",new_labels)
        improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
      #  print("improved labels",improved_labels)
        ax.set_xticklabels(improved_labels,fontsize=8)
        self._save_fig(savename,output_dir) 
        return
           
       
   def _add_a_week(self,df):
        last_date=df.columns[-1]
     #   print("last date",last_date)
     #   new_date=pd.Timestamp("2015-01-01") + pd.offsets.Day(7))
        new_date=df.columns[-1] + pd.offsets.Day(7)
     #   print("new_date=",new_date)
        df[new_date]=np.nan
      #  print("df=\n",df,"\n",df.T)
        
        return df
    
    
    
    
   def _take_out_zeros(self,df,cols):
        # cols of a list of column names
        df[cols]=df[cols].clip(lower=1000.0,axis=1)
        df[cols]=df[cols].replace(1000.0, np.nan)
        return df
    
    
   def _add_trues_and_falses(self,df,cols):
        df[cols]=df[cols].replace(1,True)
        df[cols]=df[cols].replace(0,False)
        return df
    
     
         
    
   def _add_notes(self,df,rows):
        y_text=round(np.nanmax(df.iloc[0:rows-1].to_numpy())/2.0,0)
        note_df=df[df.index.get_level_values('measure')=='notes']
     #   note_df=note_df.droplevel(['colname','measure'])
        note_df=note_df.droplevel(['colname'])
    
       # print("notes1=\n",notes) 
        note_df.sort_index(axis=1,ascending=True,inplace=True)
      
        note_df=note_df.T
        note_df['weekno']=np.arange(0,note_df.shape[0])
      #  print("notes4=\n",note_df) 
        note_df.set_index('weekno',inplace=True)
        note_df.dropna(subset = ["notes"], inplace=True)
        if note_df.shape[0]>0:
            note_df.reset_index(inplace=True)
            # number of labels max on a graph is 10
            increment_y_text=round(y_text/10,0)
            for i in range(0,note_df.shape[0]):
                 #  plt.axvline(note_df['weekno'].iloc[i], ls='--', color="black")
                   plt.text(note_df['weekno'].iloc[i],y_text, note_df['notes'].iloc[i], fontsize=8)
               #    print("note_df['notes'].iloc[i]",i,y_text,note_df['notes'].iloc[i]) 
                   y_text-=increment_y_text
        return
    
    
        
  
       
   def brand_index(self,pdf,output_dir):
           ##################################################################3
    #  jams brand index    Beerenberg vs st Dalfour (and Bonne Maman) 
     
        print("brand_index=\n",pdf.shape)
        new_pdf=self._multiple_slice_scandata(pdf,query=[('9','plottype3')])
         
        new_pdf=new_pdf.droplevel([0,1,2,3,4,5,6,7,8,9,10])
        column_names=['-'.join(tup) for tup in new_pdf.index]
        new_pdf = new_pdf.reset_index(level=[0,1],drop=True)  #'sortorder'])
        new_pdf=new_pdf.T
        newcols_dict={k:v for k,v in zip(new_pdf.columns,column_names)}
        new_pdf.rename(columns=newcols_dict, inplace=True)
    
    #   plot_brand_index(new_pdf,y_col=('Coles Beerenberg all jams','Units (000) Sold off Promotion >= 5 % 6 wks'),col_and_hue=[('Coles Bonne Maman all jams','Wks on Promotion >= 5 % 6 wks'),('Coles St Dalfour all jams','Wks on Promotion >= 5 % 6 wks')],savename="coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
        self._plot_brand_index(new_pdf,output_dir,y_col='Coles Beerenberg all jams-Units Sold off Promotion >= 5 % 6 wks',col_and_hue=['Coles Bonne Maman all jams-Wks on Promotion >= 5 % 6 wks','Coles St Dalfour all jams-Wks on Promotion >= 5 % 6 wks'],savename="Brand index jams coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
    
      #  plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo'],savename="coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
        self._plot_brand_index(new_pdf,output_dir,y_col='Woolworths Beerenberg all jams-Units Sold off Promotion >= 5 % 6 wks',col_and_hue=['Woolworths Bonne Maman all jams-Wks on Promotion >= 5 % 6 wks','Woolworths St Dalfour all jams-Wks on Promotion >= 5 % 6 wks'],savename="Brand index jams woolworths1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
    
    
    
    
     
     ##################################################################3
    #  condiments brand index   Beerenberg vs Baxters (and Whitlock or Jills)
     
       # print("pdf=\n",pdf)
        new_pdf=self._multiple_slice_scandata(pdf,query=[('10','plottype1')])
         
        new_pdf=new_pdf.droplevel([0,1,2,3,4,5,6,7,8,9,10])
        column_names=['-'.join(tup) for tup in new_pdf.index]
        new_pdf = new_pdf.reset_index(level=[0,1],drop=True)  #'sortorder'])
        new_pdf=new_pdf.T
        newcols_dict={k:v for k,v in zip(new_pdf.columns,column_names)}
        new_pdf.rename(columns=newcols_dict, inplace=True)
    
    #   plot_brand_index(new_pdf,y_col=('Coles Beerenberg all jams','Units (000) Sold off Promotion >= 5 % 6 wks'),col_and_hue=[('Coles Bonne Maman all jams','Wks on Promotion >= 5 % 6 wks'),('Coles St Dalfour all jams','Wks on Promotion >= 5 % 6 wks')],savename="coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
        self._plot_brand_index(new_pdf,output_dir,y_col='Coles Beerenberg Tomato chutney 260g-Units Sold off Promotion >= 5 % 6 wks',col_and_hue=['Coles Jills Tomato chutney 400g-Wks on Promotion >= 5 % 6 wks','Coles Baxters Tomato chutney 225g-Wks on Promotion >= 5 % 6 wks'],savename="Brand index Tomato chutney coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
    
      #  plot_brand_index(get_xs_name(df,("jams",3)).iloc[24:],y_col='coles_beerenberg_jams_off_promo_scanned',col_and_hue=['coles_bonne_maman_jams_on_promo','coles_st_dalfour_jams_on_promo'],savename="coles1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
        self._plot_brand_index(new_pdf,output_dir,y_col='Woolworths Beerenberg Tomato chutney 260g-Units Sold off Promotion >= 5 % 6 wks',col_and_hue=['Woolworths Whitlock Tomato chutney 275g-Wks on Promotion >= 5 % 6 wks','Woolworths Baxters Tomato chutney 225g-Wks on Promotion >= 5 % 6 wks'],savename="Brand index Tomato chutney woolworths1")   # miss first 22 weeks of jam data bacuase no national ranging in Coles
        print("brand index finished")
        return





    
    
    
    
    
   def _plot_type1(self,df):
        # first column is unit sales off proro  (stacked)
        # second column is unit sales on promo  (stacked)
        # third is price (second y axis)
        # fourth is notes
       
      #  print("plot type 1 =\n",df)
        weeks_back=80  
        week_freq=8
       # print("plot type1 df=\n",df)
        df=df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
        df=df.iloc[:,-weeks_back:]
        df=df.T
        df['date']=pd.to_datetime(df.index).strftime("%Y-%m").to_list()
        newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
        df=df.T
        df.iloc[0:2]*=1000
        
        #print("plot type1 df=\n",df)
        fig, ax = pyplot.subplots()
        fig.autofmt_xdate()
        ax.ticklabel_format(style='plain')
        
        self._add_notes(df,3)
        
      #  weekno=22
      #  plt.axvline(weekno, ls='--', color="black")
      #  plt.text(weekno,1, "Target\nsparsity1", fontsize=9)
     
        
     
        
        df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
        ax.set_ylabel('Units/week',fontsize=9)
    
        line=df.iloc[2].T.plot(use_index=False,xlabel="",kind='line',rot=0,style=["g-","k-"],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
        ax.right_ax.set_ylabel('$ price',fontsize=9)
        fig.legend(title="Units/week vs $ price",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.3, 1.1))
      #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
        new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
        improved_labels = ['{}\n{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
        
     #   print("improived labels=",improved_labels[0])
        improved_labels=improved_labels[:1]+improved_labels[::week_freq]
        
      #  ax.axvline(-10, ls='--', color="black")
       # ax.annotate("test1",xy=(0,0))
      #  plt.plot([1, 1], [0, 0.3], "g:")
      #  plt.text(0.05, 0.32, "Target\nsparsity", fontsize=9)
     
      
        ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
        ax.set_xticklabels(improved_labels,fontsize=6)
        
     
        
        
       # ax.axvline(10, ls='--')
        return
    
    
    
    
   def _plot_type2(self,df,this_year_df,last_year_df):
        # first column is total units sales
        # second column is distribution 
        # third is notes
        
      #  print("plot type 2 before df=\n",df)
       #3 print("plotdf.T=\n",df.T)
       # pv = pd.pivot_table(df.T, index=df.index, columns=df.index,
       #                 values='value', aggfunc='sum')
       # print("pv=\n",pv)
       # pv.plot()
       # plt.show()
          
        week_freq=8
       # print("plot type1 df=\n",df)
        this_year_df=this_year_df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
        last_year_df=last_year_df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
        df=df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
       
      #  df=df.T
      #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
      #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
      #  df=df.T
        this_year_df.iloc[:1]*=1000
        last_year_df.iloc[:1]*=1000
    
    
       # print("plot type 2 after df=\n",df)
    
    
        #print("plot type1 df=\n",df)
        fig, ax = pyplot.subplots()
        ax.ticklabel_format(style='plain')
       
        #add_notes(df,2)
        
    
    #    fig = plt.figure()
    #ax1 = fig.add_subplot(111)
        ax2 = ax.twiny()
    
    
    
        fig.autofmt_xdate()
     
     #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
        ax.set_ylabel('Units/week this year vs LY',fontsize=9)
     #  ax.annotate("test2",xy=(0,0))
     #   ax.plot([0.5,0.41], [0, 0.3], "k:")
      #  plt.axvline(12, ls='--', color="black")
      #  plt.text(0.55, 0.82, "Target\nsparsity2", fontsize=7)
      
        line=this_year_df.iloc[:1].T.plot(use_index=True,grid=True,xlabel="",kind='line',style=["r-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
       # current_handles0, current_labels0 = ax.get_legend_handles_labels()
    
        line=last_year_df.iloc[:1].T.plot(use_index=True,grid=False,xlabel="",kind='line',style=["r:"],secondary_y=False,fontsize=9,legend=False,ax=ax2)   #,ax=ax2)
       # current_handles, current_labels = plt.gca().get_legend_handles_labels()
      #  current_handles1, current_labels1 = ax2.get_legend_handles_labels()
    
        #if this_year_df.shape[0]>=2:
         #   line=last_year_df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b:'],secondary_y=False,fontsize=9,legend=False,ax=ax2)   #,ax=ax2)
        line=this_year_df.iloc[1:2].T.plot(use_index=True,grid=False,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
        current_handles2, current_labels2 = ax.get_legend_handles_labels()
   #     print("\r",current_labels2,"     ")
       # print(current_handles2[0].current_labels2[0])
        # if df.shape[0]>=3:
       #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
     
    #  ax.set_ylabel('Units/week',fontsize=9)
      #  ax.axvline(10, ls='--', color="black")
        ax.right_ax.set_ylabel('Distribution this year',fontsize=9)
     #   ax.axvline(-10, ls='--', color="yellow")
      #  ax.annotate("test2",xy=(0,0))
      #  plt.plot([1, 1], [0, 0.3], "k:")
      #  plt.text(0.05, 0.32, "Target\nsparsity", fontsize=9)
      #  current_handles, current_labels = plt.gca().get_legend_handles_labels()
       # print("cl=",current_labels,line)
       # current_labels=current_labels+" Last year"
    # sort or reorder the labels and handles
    #reversed_handles = list(reversed(current_handles))
    #reversed_labels = list(reversed(current_labels))
    
    # call plt.legend() with the new values
    #plt.legend(reversed_handles,reversed_labels)
        
            
            
        fig.legend([current_labels2[0],current_labels2[0]+" last year","Distribution (sold)"],title="Units/week TY vs LY",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.2, 1.1))
     #   ax.axvline(12, ls='--', color="yellow")
      #  ax.annotate("test2",xy=(0,0))
    
        #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
      #  new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
      #  improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
      #  improved_labels=improved_labels[::week_freq]
      
      #  ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
      #  ax.set_xticklabels(improved_labels,fontsize=8)
      #  plt.axvline(12, ls='--', color="black")
      #  plt.text(0.55, 0.82, "Target\nsparsity2", fontsize=7)
     
        return
    
    
    
    
    
   def _plot_type3(self,df):
           # first column is total units sales
        # second column is distribution 
        
      #  print("plot type 3 df=\n",df)
     
          
        week_freq=8
       # print("plot type1 df=\n",df)
        df=df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
        
      #  df=df.T
      #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
      #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
      #  df=df.T
        #df.iloc[:1]*=1000
        df*=1000
     #   print("plot type3 df=\n",df)
        fig, ax = pyplot.subplots()
        fig.autofmt_xdate()
        ax.ticklabel_format(style='plain')
        
     #   add_notes(df,1)
        
    
     #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
        ax.set_ylabel('Total units/week',fontsize=9)
    
        line=df.T.plot(use_index=True,xlabel="",kind='line',style=["g-","r-","b-","k-","c-","m-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
      #  if df.shape[0]>=2:
       # line=df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
       # if df.shape[0]>=3:
       #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
        
    #  ax.set_ylabel('Units/week',fontsize=9)
    
     #   ax.right_ax.set_ylabel('Units/week',fontsize=9)
        fig.legend(title="Total units/week",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.3, 1.1))
      #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
      #  new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
      #  improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
      #  improved_labels=improved_labels[::week_freq]
      
      #  ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
      #  ax.set_xticklabels(improved_labels,fontsize=8)
    
       # return
    
    
       # print("plot 3")
        return
    
    
    
    
   def _plot_type4(self,df):
              # first column is total units sales
        # second column is distribution 
        
        return
          
    #     week_freq=8
    #    # print("plot type1 df=\n",df)
    #     df=df.droplevel([0,1,2,3,4,5,6,7,8,9,10])
        
    #   #  df=df.T
    #   #  df['date']=pd.to_datetime(df.index).strftime("%Y-%m-%d").to_list()
    #   #  newdates = pd.to_datetime(df['date']).apply(lambda date: date.toordinal()).to_list()
    #   #  df=df.T
    #     df.iloc[:]*=1000
    #  #   print("plot type3 df=\n",df)
    #     fig, ax = pyplot.subplots()
    #     fig.autofmt_xdate()
    #     ax.ticklabel_format(style='plain')
       
    #  #   df.iloc[0:2].T.plot(xlabel="",use_index=False,kind='bar',color=['blue','red'],secondary_y=False,stacked=True,fontsize=9,ax=ax,legend=False)
    #     ax.set_ylabel('Units/week',fontsize=9)
    
    #     line=df.T.plot(use_index=True,xlabel="",kind='line',style=["b-","r-","g-","k-","c-","m-"],secondary_y=False,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
    #   #  if df.shape[0]>=2:
    #   #  line=df.iloc[1:2].T.plot(use_index=True,xlabel="",kind='line',style=['b-'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
    
    #    # if df.shape[0]>=3:
    #    #     line=df.iloc[2:3].T.plot(use_index=True,xlabel="",kind='line',style=['g:'],secondary_y=True,fontsize=9,legend=False,ax=ax)   #,ax=ax2)
        
    # #  ax.set_ylabel('Units/week',fontsize=9)
    
    #   #  ax.right_ax.set_ylabel('Units/week',fontsize=9)
    #     fig.legend(title="Units/week",title_fontsize=9,fontsize=7,loc='upper center', bbox_to_anchor=(0.3, 1.1))
    #   #  print(df.shape,"xticks=",ax.get_xticks(),df.iloc[:,ax.get_xticks()])
    #   #  new_labels = [dt.date.fromordinal(int(item)) for item in newdates]   #ax.get_xticks()]
    #   #  improved_labels = ['{}-{}'.format(calendar.month_abbr[int(m)],y) for y, m , d in map(lambda x: str(x).split('-'), new_labels)]
    #   #  improved_labels=improved_labels[::week_freq]
      
    #   #  ax.xaxis.set_major_locator(ticker.MultipleLocator(week_freq))
    #   #  ax.set_xticklabels(improved_labels,fontsize=8)
    
    #    # return
    
    
    
    #     return
    
    
    
    
    
    
    
    
   def plot_scan_weekly(self,df,output_dir):
     #   df.replace(0.0,np.nan,inplace=True)
        print("Plot scan weekly data to",output_dir)    
          #   print(new_df)
        plottypes=list(set(list(set(df.index.get_level_values('plottype').astype(str).tolist()))+list(set(df.index.get_level_values('plottype1').astype(str).tolist()))))   #+list(set(df.index.get_level_values('plottype2').astype(str).tolist()))+list(set(df.index.get_level_values('plottype3').astype(str).tolist()))))
       #     plottypes=list(set([p for p in plottypes if p!='0']))
       #     print("plotypes=",plottypes)
        for pt in plottypes:  
            plotnumbers=list(set(df.index.get_level_values('plotnumber').astype(str).tolist()))
        #    colnames=list(set(df.index.get_level_values('colname').astype(str).tolist()))
        #    retailer=list(set(df.index.get_level_values('retailer').astype(str).tolist()))
        #    variety=list(set(df.index.get_level_values('variety').astype(str).tolist()))
    
            new_df=pd.concat((self._multiple_slice_scandata(df,[(pt,'plottype')]) ,self._multiple_slice_scandata(df,[(pt,'plottype1')])),axis=0)   #,(pt,'plottype1')])
     
        #    colnames=list(set(new_df.index.get_level_values('colname').astype(str).tolist()))
        #    retailer=list(set(new_df.index.get_level_values('retailer').astype(str).tolist()))
        #    variety=list(set(new_df.index.get_level_values('variety').astype(str).tolist()))
        #    brand=list(set(new_df.index.get_level_values('brand').astype(str).tolist()))
    
    
       #     print("pt=",pt,colnames,retailer,variety,brand)
    
    
            if (pt=='3') :  #| (pt=='4') | (pt=='5') | (pt=='9'):
                
                plot_df=new_df.replace(0.0,np.nan)
      
                colnames=list(set(plot_df.index.get_level_values('colname').astype(str).tolist()))
                retailer=list(set(plot_df.index.get_level_values('retailer').astype(str).tolist()))
                variety=list(set(plot_df.index.get_level_values('variety').astype(str).tolist()))
                brand=list(set(plot_df.index.get_level_values('brand').astype(str).tolist()))
    
                self._plot_type3(plot_df)
                self._save_fig("Scan_weekly_data_"+str(pt)+"_plot_"+str(retailer[0])+"_"+str(variety[0])+"_"+str(brand[0])+"_"+str(colnames[0])+"_3",output_dir)
             #   plt.close()
     
            else:
    
           #     print("pt=",pt)   #,"plotnumbdsers",plotnumbers)
                for pn in plotnumbers:
                    plot_df=self._multiple_slice_scandata(new_df,[(pn,'plotnumber')])
                    plot_df.replace(0.0,np.nan,inplace=True)
                     
                    colnames=list(set(plot_df.index.get_level_values('colname').astype(str).tolist()))
                    retailer=list(set(plot_df.index.get_level_values('retailer').astype(str).tolist()))
                    variety=list(set(plot_df.index.get_level_values('variety').astype(str).tolist()))
                    brand=list(set(plot_df.index.get_level_values('brand').astype(str).tolist()))
    
                    
    
    
                    last_year_plot_df=plot_df.iloc[:,-(dd2.dash2_dict['scan']['e_scandata_number_of_weeks']+52):-(dd2.dash2_dict['scan']['e_scandata_number_of_weeks']-1)]
                    this_year_plot_df=plot_df.iloc[:,-dd2.dash2_dict['scan']['e_scandata_number_of_weeks']:]    
    
            #        print("pn",pn)
        
                 #   print("plot_df=\n",plot_df)
                #   print("this year plot df=",this_year_plot_df)
                 #   print("last year plot df=",last_year_plot_df)
                    if str(pt)=='1':   #standard plot type
                        self._plot_type1(plot_df)
                        self._save_fig("Scan_weekly_data_"+str(pt)+"_plot_"+str(retailer[0])+"_"+str(variety[0])+"_"+str(brand[0])+"_"+str(colnames[0])+"_"+str(pn)+"_"+str(pt),output_dir)
                        plt.close()    
                    elif str(pt)=='2':   #stacked bars plus right axis price
                        self._plot_type2(df,this_year_plot_df,last_year_plot_df)
                        self._save_fig("Scan_weekly_data_"+str(pt)+"_plot_"+str(retailer[0])+"_"+str(variety[0])+"_"+str(brand[0])+"_"+str(colnames[0])+"_"+str(pn)+"_"+str(pt),output_dir)
                        plt.close()
         
                    else:    
                        pass
                #elif str(pt)=='4':   #unused 
                #    plot_type4(plot_df)
                #elif str(pt)=='0':
                #    pass
       #         save_fig("ZZ_scandata_plot_"+str(colnames[0])+"_"+str(pt)+"_"+pn)
          #      plt.show()
                
                 
        plt.close('all')
        return








        
        
   def _scan_monthly_query_df(self,new_df,query_name):
# =============================================================================
#         
#         #   query of AND's - input a list of tuples.  ["AND",(field_name1,value1) and (field_name2,value2) and ...]
#             the first element is the type of query  -"&"-AND, "|"-OR, "!"-NOT, "B"-between
# #            return a slice of the df as a copy
# # 
# #        a query of OR 's  -  input a list of tuples.  ["OR",(field_name1,value1) or (field_name2,value2) or ...]
# #            return a slice of the df as a copy
# #
# #        a query_between is only a triple tuple  ["BD",(fieldname,startvalue,endvalue)]
#                "BD" for between dates, "B" for between numbers or strings
# # 
# #        a query_not is only a single triple tuple ["NOT",(fieldname,value)]   
# 
#         
# =========================================================================
  #   print("query_df df=\n",df,"query_name=",query_name)  
     if (query_name==[]) | (new_df.shape[0]==0):
           return new_df.copy(deep=True) 
     else :   
           if (query_name[0]=="AND") | (query_name[0]=='OR') | (query_name[0]=="BD")| (query_name[0]=="B") | (query_name[0]=="NOT"):
                operator=query_name[0]
              #  print("valid operator",operator)
                query_list=query_name[1:]
             #   print("quwery",query_list)
       #         new_df=df.copy()
                if operator=="AND":
                    for q in query_list:    
                  #      new_df=new_df[(new_df[q[0]]==q[1])].copy(deep=True) 
                 #       print("scan monthly query q=",q)
                        new_df=new_df.xs(q[1],level=q[0],axis='columns',drop_level=False).copy(deep=True)
                #       print("AND new_df=\n",new_df)
                        
                    #    print("AND query=",q,"&",new_df.shape) 
                 #   print("new_df=\n",new_df)    
                elif operator=="OR":
                    new_df_list=[]
                    for q in query_list:    
                 #       new_df_list.append(new_df[(new_df[q[0]]==q[1])].copy(deep=True)) 
                         new_df_list.append(new_df.xs(q[1],level=q[0],axis='columns',drop_level=False).copy(deep=True))
                 #   print("OR query=",q,"|",new_df_list[-1].shape)
                    new_df=new_df_list[0]    
                    for i in range(1,len(query_list)):    
                        new_df=pd.concat((new_df,new_df_list[i]),axis=0)   
                  #  print("before drop",new_df.shape)    
                    new_df.drop_duplicates(keep="first",inplace=True)   
                  #  print("after drop",new_df.shape)
                elif operator=="NOT":
                    for q in query_list:    
                  #      new_df=new_df[(new_df[q[0]]!=q[1])].copy(deep=True) 
                     #    quq='"'+str(q[0])+' != '+ "'"+str(q[1])+"'"+'"'
                         new_df=new_df.T
                         new_df=new_df[new_df.index.get_level_values(q[0]) != q[1]]
                         new_df=new_df.T
                      #   print("query quq=",quq)
                       #  new_df=new_df.query(quq)
                     #    new_df.xs(not q[1],level=q[0],axis='columns',drop_level=False).copy(deep=True)
                   
                elif operator=="BD":  # betwwen dates
                  #  if (len(query_list[0])==3):
                    for q in query_list:
                    #    print("between ql=",q[1],q[2])
                        start=q[1]
                        end=q[2]
                        new_df=new_df[(pd.to_datetime(new_df[q[0]])>=pd.to_datetime(q[1])) & (pd.to_datetime(new_df[q[0]])<=pd.to_datetime(q[2]))].copy(deep=True) 
                     #       print("Beeterm AND query=",q,"&",new_df.shape) 
                   # else:
                   #     print("Error in between statement")
                elif operator=="B":  # btween numbers or strings
                  #  if (len(query_list[0])==3):
                    for q in query_list:
                    #    print("between ql=",q[1],q[2])
                        start=q[1]
                        end=q[2]
                        new_df=new_df[(new_df[q[0]]>=q[1]) & (new_df[q[0]]<=q[2])].copy(deep=True) 
                     #       print("Beeterm AND query=",q,"&",new_df.shape) 
                   # else:
                   #     print("Error in between statement")
     
                else:
                    print("operator not found\n")
                        
                return new_df.copy(deep=True)
                      
           else:
                print("invalid operator")
                return pd.DataFrame([])
    
  
      
   def _build_a_scan_monthly_query_dict(self,query_name):
     #   print("build an entry query_name",query_name)
     #   print("query name=",query_name[1])
     #   print("filesave name=",query_name[0])
        #queries=query_name[1]
      #  query_name=qd.queries[q]
        new_df=dd2.dash2_dict['scan']['query_df']
      #  new_df=query_df.copy(deep=True)
        for qn in query_name[1]:  
        #    print("build a query dict qn=",qn)
            q_df=self._scan_monthly_query_df(new_df,qn)
            new_df=q_df.copy()
        q_df.drop_duplicates(keep="first",inplace=True)    
       # q_df=smooth(q_df)
    #    self.save(q_df,dd2.dash2_dict['scan']['save_dir'],query_name[0])   
        return q_df
    
           
    
   def scan_monthly_queries(self,qdf):
      #  self.query=sales_query_class()
      
        query_df=qdf.copy()
        dd2.dash2_dict['scan']['query_df']=query_df
        if query_df.shape[0]>0:
         #   df=df.rename(columns=qd.rename_columns_dict)  
          #  query_handles=[]
            query_filenames=[]
            query_filenames.append(p_map(self._build_a_scan_monthly_query_dict,dd2.dash2_dict['scan']['queries'].items()))   #st.save_query(q_df,query_name,root=False)   
         #   query_filenames=[q[:250] for q in query_handles[0]]  # if len(q)>249]
         #   print("build a query dict query filenames",query_filenames)
            return {k: v for k, v in zip(dd2.dash2_dict['scan']['queries'].keys(),query_filenames[0])}     #,{k: v for k, v in zip(qd.queries.keys(),query_filenames)}
        else:
            print("scan monthly df empty",query_df)
            return {}
 