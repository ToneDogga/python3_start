#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:40:09 2020

@author: tonedogga
"""

import os
import numpy as np
import pandas as pd
from p_tqdm import p_map,p_umap
from pathlib import Path


import dash2_dict as dd2






class price_class(object):
   def __init__(self):
       #self.price_init="price_init called"
       #print(self.price_init)
       pass
       
   def load_from_excel(self,in_dir,filename):
       print("load prices from",in_dir+filename)   #dd2.dash2_dict['price']['in_file_dir']+dd2.dash2_dict['price']['in_file'])      
       price_df=pd.read_excel(in_dir+filename,sheet_name="prices",usecols=range(0,dd2.dash2_dict['price']['price_width']),header=0,skiprows=[0,2,3],index_col=0,verbose=False)  # -1 means all rows  
       price_df=price_df.iloc[:-2]
       price_df = price_df.rename_axis("product")
       return price_df

   
       
   def preprocess(self,price_df):
      # print("preprocess price df")
       price_df.replace({'productgroup':dd2.productgroup_dict},inplace=True)
       price_df.replace({'productgroup':dd2.productgroups_dict},inplace=True)
       price_df.replace({'specialpricecat':dd2.spc_dict},inplace=True)
       price_df.replace({'specialpricecat':dd2.spcs_dict},inplace=True)
       price_df.replace({'salesrep':dd2.salesrep_dict},inplace=True)
       return price_df
    
   

       
   def save(self,price_df,save_dir,savefile):
       print("save price_df to ",save_dir+savefile)
       os.makedirs(save_dir, exist_ok=True)
       if isinstance(price_df,pd.DataFrame):
           if not price_df.empty:
               #price_df=pd.DataFrame([])
               price_df.to_pickle(save_dir+savefile,protocol=-1)
               return True
           else:
               return False
       else:
           return False
    
  #     return("price save outfile")
       
   def load(self,save_dir,savefile):
       # os.makedirs(save_dir, exist_ok=True)
       my_file = Path(save_dir+savefile)
       if my_file.is_file():
           return pd.read_pickle(save_dir+savefile)
       else:
           print("load price_df error.")
           return
  
     
    
           
   def _promo_flags(self,s_sales_df,price_df):
        
            
        price_df.reset_index(inplace=True)
        #print(price_df,price_df.shape)
        
        new_price_df= pd.melt(price_df, 
                    id_vars='product', 
                    value_vars=list(price_df.columns[1:]), # list of days of the week
                    var_name='spc', 
                    value_name='price_sb')
        #print("npdf cols=",pd.unique(new_price_df['specialpricecat']))
        
        #print(new_price_df)
        new_price_df['spc']=new_price_df['spc'].astype(np.float32)
        #print("npdf cols2=",pd.unique(new_price_df['specialpricecat']))
        
        #print("npdf1=\n",new_price_df)
        
        
        new_price_df = new_price_df.set_index(['spc','product'],drop=False)
        
        #print("npdf2=\n",new_price_df)
        #print(sales_df.columns.dtype)
        s_sales_df['spc']=s_sales_df['spc'].astype(np.float32)
        s_sales_df['product']=s_sales_df['product'].astype(np.str)
        
        #print("nsdf1=\n",sales_df)
        
        
        s_sales_df.loc[:,'price']=np.around(s_sales_df.loc[:,'salesval']/s_sales_df.loc[:,'qty'],2)
        #print("nsdf`=\n",sales_df)
        
        
        
        new_sales_df = s_sales_df.set_index(['spc','product'],drop=False)
        
        #print("nsdf2=\n",new_sales_df)
        
        test_df=new_sales_df.join(new_price_df,how='inner',lsuffix="l",rsuffix='r')   #,sort=True)
        #test_df=pd.concat((new_sales_df,new_price_df),axis=1,join='outer')   #keys=('specialpricecat','product'))   #,on=['specialpricecat','product'])
        
        #print("tdf1=\n",test_df)
        
        test_df.drop(["productr",'spcr'],axis=1,inplace=True)
        test_df=test_df.rename(columns={'productl':'product','spcl':'spc'})
        test_df.set_index('date',drop=False,inplace=True)
        
         
        test_df['discrep']=np.round(test_df['price_sb']-test_df['price'],2)
      #  test_df['on_promo_guess']=False
        test_df['on_promo']=(((test_df['spc']==88) & (test_df['discrep']>0.09)) & ((test_df['pg']=='10') | (test_df['pg']=='11') | (test_df['pg']=='12') | (test_df['pg']=='13') | (test_df['pg']=='14') | (test_df['pg']=='15') |(test_df['pg']=='16') |(test_df['pg']=='17')))
        test_df.sort_index(ascending=False,inplace=True)
  
        return test_df
      
   
    
   def flag_promotions(self,in_sales_df,price_df,output_dir):
        print("Augment sales_df with promotion flags...")   
        os.makedirs(output_dir, exist_ok=True)
        complete_augmented_sales_df=self._promo_flags(in_sales_df,price_df)
     #   complete_augmented_sales_df.to_pickle(dd.sales_df_complete_augmented_savename,protocol=-1)          
    
        end_date=in_sales_df['date'].iloc[-1]- pd.Timedelta(30, unit='d')
        #print(end_date)
        #print("ysdf=",sales_df)
      #  recent_sales_df=sales_df[sales_df['date']>end_date]
      #  augmented_sales_df=promo_flags(recent_sales_df,price_df)
     #   augmented_sales_df.to_pickle(dd.sales_df_augmented_savename,protocol=-1)          
        on_promo_sales_df=complete_augmented_sales_df[complete_augmented_sales_df['on_promo']==True]    #.copy(deep=True)
        
      #  print(on_promo_sales_df)
        on_promo_sales_df["month"] = pd.to_datetime(on_promo_sales_df['date']).dt.strftime('%b')
        on_promo_sales_df["monthno"] = pd.to_datetime(on_promo_sales_df['date']).dt.strftime('%m') 
        on_promo_sales_df["year"] = pd.to_datetime(on_promo_sales_df['date']).dt.strftime('%Y')
        
        on_promo_sales_df.replace({'pg':dd2.productgroup_dict},inplace=True)
        on_promo_sales_df.replace({'pg':dd2.productgroups_dict},inplace=True)
        on_promo_sales_df.replace({'spc':dd2.spc_dict},inplace=True)
        on_promo_sales_df.replace({'salesrep':dd2.salesrep_dict},inplace=True)
    
        promo_pivot_df=pd.pivot_table(on_promo_sales_df, values='discrep',index=['salesrep','code','pg','product'], columns=['year','monthno','month'],aggfunc=np.sum, margins=True,dropna=True)  # fill_value=0)
      #  print(promo_pivot_df) 
        promo_pivot_df.to_excel(output_dir+"price 088 promotions summary4.xlsx") 
        
        promo_pivot_df=pd.pivot_table(on_promo_sales_df, values='discrep',index=['salesrep','code','pg'], columns=['year','monthno','month'],aggfunc=np.sum, margins=True,dropna=True)  # fill_value=0)
      #  print(promo_pivot_df) 
        promo_pivot_df.to_excel(output_dir+"price 088 promotions summary3.xlsx") 
    
        promo_pivot_df=pd.pivot_table(on_promo_sales_df, values='discrep',index=['salesrep','code'], columns=['year','monthno','month'],aggfunc=np.sum, margins=True,dropna=True)  # fill_value=0)
      #  print(promo_pivot_df) 
        promo_pivot_df.to_excel(output_dir+"price 088 promotions summary2.xlsx") 
     
        promo_pivot_df=pd.pivot_table(on_promo_sales_df, values='discrep',index=['salesrep'], columns=['year','monthno','month'],aggfunc=np.sum, margins=True,dropna=True)  # fill_value=0)
      #  print("Promotional retail spending in SA stores:")
      #  print(promo_pivot_df.iloc[:,-5:]) 
        promo_pivot_df.to_excel(output_dir+"price 088 promotions summary1.xlsx") 
    
    
       # print("\nPromotions flagged",promo_pivot_df.shape,"to 088 promotions summary 1 to 4.xlsx")
        return complete_augmented_sales_df,promo_pivot_df
    
     
    
    
   def report(self,test_df,promo_pivot_df,output_dir):
        print("Promotional retail spending in SA stores:")
        print(promo_pivot_df.iloc[:,-5:]) 
        print("\nPromotions flagged",promo_pivot_df.shape,"to 088 promotions summary 1 to 4.xlsx")
      #  test_df=pd.read_pickle(dd.sales_df_augmented_savename)
       # print("augmented testdf=\n",test_df)
           # print("tdf2=\n",test_df)
        
        # test_df=test_df[(test_df['discrep']!=0)] # & (test_df['productgroup']=='10')]
        # #print("tdf3=\n",test_df)
        
        test_df.dropna(axis=0,subset=['price','price_sb'],inplace=True)
        test_df.sort_values(inplace=True,ascending=False,by='discrep')
        # #print("tdf4=\n",test_df)
        
        summ_df=pd.pivot_table(test_df, values='discrep',index='code',columns='product',aggfunc=np.sum, margins=True,dropna=True,observed=True)
        summ_df=summ_df.dropna(axis=0,how='all')
        summ_df=summ_df.dropna(axis=1,how='all')
    
        summ_df.fillna(0,inplace=True)
        summ_df = summ_df.sort_values('All', axis=1, ascending=False)
        summ_df = summ_df.sort_values('All', axis=0, ascending=False)
     #   print("Sample of last 30 days underpriced summary, check excel report:\n",summ_df.iloc[10:20,10:20])
        print('Underpriced?? summary report completed:',output_dir+dd2.dash2_dict['price']['price_discrepencies_summary'],"\n") 
        os.makedirs(output_dir, exist_ok=True)
        summ_df.to_excel(output_dir+dd2.dash2_dict['price']['price_discrepencies_summary'])
        
        print("============================================================================\n")  
 
        return
    
