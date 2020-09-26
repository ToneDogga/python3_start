#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 07:27:32 2020

@author: tonedogga
"""

import pandas as pd
import numpy as np

sales_df=pd.read_pickle('sales_trans_df.pkl')    #.head(40000)
print(sales_df,sales_df.shape)



#print(pd.unique(sales_df['product']))

price_df=pd.read_pickle('price_df.pkl')
#print(price_df,price_df.shape)

#price_df=price_df.T
price_df.reset_index(inplace=True)
#print(price_df,price_df.shape)

new_price_df= pd.melt(price_df, 
            id_vars='product', 
            value_vars=list(price_df.columns[1:]), # list of days of the week
            var_name='specialpricecat', 
            value_name='price_sb')
#print("npdf cols=",pd.unique(new_price_df['specialpricecat']))

#print(new_price_df)
new_price_df['specialpricecat']=new_price_df['specialpricecat'].astype(np.float32)
#print("npdf cols2=",pd.unique(new_price_df['specialpricecat']))

#print("npdf1=\n",new_price_df)


new_price_df = new_price_df.set_index(['specialpricecat','product'],drop=False)

#print("npdf2=\n",new_price_df)
#print(sales_df.columns.dtype)
sales_df['specialpricecat']=sales_df['specialpricecat'].astype(np.float32)
sales_df['product']=sales_df['product'].astype(np.str)

#print("nsdf1=\n",sales_df)


sales_df.loc[:,'price']=np.around(sales_df.loc[:,'salesval']/sales_df.loc[:,'qty'],2)
#print("nsdf`=\n",sales_df)



new_sales_df = sales_df.set_index(['specialpricecat','product'],drop=False)

#print("nsdf2=\n",new_sales_df)

test_df=new_sales_df.join(new_price_df,how='inner',lsuffix="l",rsuffix='r')   #,sort=True)
#test_df=pd.concat((new_sales_df,new_price_df),axis=1,join='outer')   #keys=('specialpricecat','product'))   #,on=['specialpricecat','product'])

#print("tdf1=\n",test_df)

test_df.drop(["productr",'specialpricecatr'],axis=1,inplace=True)
test_df=test_df.rename(columns={'productl':'product','specialpricecatl':'specialpricecat'})
test_df.set_index('date',drop=True,inplace=True)
test_df.sort_index(ascending=False,inplace=True)



test_df['discrep']=np.round(test_df['price_sb']-test_df['price'],2)

return test_df

# print("tdf2=\n",test_df)

# test_df=test_df[(test_df['discrep']!=0)] # & (test_df['productgroup']=='10')]
# #print("tdf3=\n",test_df)

# test_df.dropna(axis=0,subset=['price','price_sb'],inplace=True)
# test_df.sort_values(inplace=True,ascending=False,by='discrep')
# #print("tdf4=\n",test_df)

# summ_df=pd.pivot_table(test_df, values='discrep',index='code',columns='productgroup',aggfunc=np.sum, margins=True,dropna=True,observed=True)
# summ_df.fillna(0,inplace=True)
# print(summ_df)

