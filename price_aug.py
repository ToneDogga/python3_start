#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 07:27:32 2020

@author: tonedogga
"""

import pandas as pd
import numpy as np
import datetime as dt
import BB_data_dict as dd



def distribution_report_counts(days_back_to_start,days_back_to_end):
    
    sales_df=pd.read_pickle('sales_trans_df.pkl')    #.head(40000)
    #print(sales_df.shape)
    
    print("\nCreating distribution count table from sales",sales_df.shape)
    
    end_date=sales_df.index[0]- pd.Timedelta(days_back_to_end, unit='d')
    startend_date=sales_df.index[0]- pd.Timedelta(days_back_to_start, unit='d')
    
    #print(startend_date,end_date)
    
    sales_df = sales_df.drop(sales_df[(sales_df['productgroup']==0)].index)
    
    sales_df["month"] = pd.to_datetime(sales_df["date"]).dt.strftime('%m-%b')
    sales_df['quarter'] = sales_df['date'].dt.quarter
    
    #sales_df["qtr"] = pd.to_datetime(sales_df["date"]).dt.strftime('%m-%b')
    sales_df["year"] = pd.to_datetime(sales_df["date"]).dt.strftime('%Y')
    
    new_sales_df=sales_df[(sales_df.index<end_date) & (sales_df.index>=startend_date)]
    #year_sales_df.sort_values(['date'],ascending=[True],inplace=True)
    
    new_sales_df=new_sales_df[new_sales_df['productgroup'].isin(dd.product_groups_only) & new_sales_df['specialpricecat'].isin(dd.spc_only)]   
    new_sales_df=new_sales_df[(new_sales_df['qty']>0) & (new_sales_df['salesval']>0)]   
     
    new_sales_df.replace({'productgroup':dd.productgroup_dict},inplace=True)
    new_sales_df.replace({'productgroup':dd.productgroups_dict},inplace=True)
    new_sales_df.replace({'specialpricecat':dd.spc_dict},inplace=True)
    new_sales_df.replace({'salesrep':dd.salesrep_dict},inplace=True)
    
    pivot_df=pd.pivot_table(new_sales_df, values=['product'],index=['salesrep','code','productgroup'], columns=['year','quarter'],aggfunc=pd.Series.nunique, margins=True,dropna=True,fill_value=0)
    
    pivot_df.to_excel("distribution_report_counts.xlsx") 
    print("Distribution report count completed",pivot_df.shape,"\n")
        # dd.report_dict[dd.report(name,6,"_*","_*")]=pivot_df
    return



#sales_df=pd.read_pickle('sales_trans_df.pkl')    #.head(40000)
#copy_sales_df=sales_df.copy(deep=True)
distribution_report_counts(days_back_to_start=732,days_back_to_end=0)


#print(sales_df.shape)





# distdollars_df=pd.read_pickle("dd_df.pkl") 
# print("did=\n",distdollars_df)
 
# #tot_cust=distdollars_df.index.get_level_values(2)[1:]
# #tot_prod=distdollars_df.columns.get_level_values(1)[1:]
# distdollars_df=distdollars_df.droplevel([0],axis=1)
# distdollars_df=distdollars_df.droplevel([0,1],axis=0)

# cust_tot=distdollars_df['total']
# distdollars_df=distdollars_df.T
# prod_tot=distdollars_df['total']
# #print(cust_tot)
# prod_tot[1:60].to_frame().plot.bar(ylabel="$",fontsize=6)
# cust_tot[1:60].to_frame().plot.bar(ylabel="$",fontsize=6)
# #pareto=pd.DataFrame(columns={"prod":prod_tot,"cust":cust_tot})
# #print("td=",cust_tot,prod_tot)

# #pareto[['cust']].plot.bar()
  
 
  

# #print(pd.unique(sales_df['product']))

# price_df=pd.read_pickle('price_df.pkl')
# #print(price_df,price_df.shape)

# #price_df=price_df.T
# price_df.reset_index(inplace=True)
# #print(price_df,price_df.shape)

# new_price_df= pd.melt(price_df, 
#             id_vars='product', 
#             value_vars=list(price_df.columns[1:]), # list of days of the week
#             var_name='specialpricecat', 
#             value_name='price_sb')
# #print("npdf cols=",pd.unique(new_price_df['specialpricecat']))

# #print(new_price_df)
# new_price_df['specialpricecat']=new_price_df['specialpricecat'].astype(np.float32)
# #print("npdf cols2=",pd.unique(new_price_df['specialpricecat']))

# #print("npdf1=\n",new_price_df)


# new_price_df = new_price_df.set_index(['specialpricecat','product'],drop=False)

# #print("npdf2=\n",new_price_df)
# #print(sales_df.columns.dtype)
# sales_df['specialpricecat']=sales_df['specialpricecat'].astype(np.float32)
# sales_df['product']=sales_df['product'].astype(np.str)

# #print("nsdf1=\n",sales_df)


# sales_df.loc[:,'price']=np.around(sales_df.loc[:,'salesval']/sales_df.loc[:,'qty'],2)
# #print("nsdf`=\n",sales_df)



# new_sales_df = sales_df.set_index(['specialpricecat','product'],drop=False)

# #print("nsdf2=\n",new_sales_df)

# test_df=new_sales_df.join(new_price_df,how='inner',lsuffix="l",rsuffix='r')   #,sort=True)
# #test_df=pd.concat((new_sales_df,new_price_df),axis=1,join='outer')   #keys=('specialpricecat','product'))   #,on=['specialpricecat','product'])

# #print("tdf1=\n",test_df)

# test_df.drop(["productr",'specialpricecatr'],axis=1,inplace=True)
# test_df=test_df.rename(columns={'productl':'product','specialpricecatl':'specialpricecat'})
# test_df.set_index('date',drop=True,inplace=True)
# test_df.sort_index(ascending=False,inplace=True)



# test_df['discrep']=np.round(test_df['price_sb']-test_df['price'],2)

# # print("tdf2=\n",test_df)

# # test_df=test_df[(test_df['discrep']!=0)] # & (test_df['productgroup']=='10')]
# # #print("tdf3=\n",test_df)

# # test_df.dropna(axis=0,subset=['price','price_sb'],inplace=True)
# # test_df.sort_values(inplace=True,ascending=False,by='discrep')
# # #print("tdf4=\n",test_df)

# # summ_df=pd.pivot_table(test_df, values='discrep',index='code',columns='productgroup',aggfunc=np.sum, margins=True,dropna=True,observed=True)
# # summ_df.fillna(0,inplace=True)
# # print(summ_df)

