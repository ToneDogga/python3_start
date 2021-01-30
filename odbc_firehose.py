#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:52:07 2020

@author: tonedogga
"""
import os
#import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime,timedelta




import bbtv1_dict as dd2
import pyodbc 
# check /etc/odbcinst.ini and /etc/odbc.ini for OBDC setup on linux



class odbc_firehose_class():
    def __init__(self):
        pass
    
    def load_sales_df(self,load_dir,load_file):
        status_report=""
        connection_string="DSN=salestran;SERVER=bbapp01.beerenberg.local;PORT=5432;UID=postgres;;SSLmode=disable;ReadOnly=0;Protocol=7.4;FakeOidIndex=0;ShowOidColumn=0;RowVersioning=0;ShowSystemTables=0;Fetch=100;UnknownSizes=0;MaxVarcharSize=255;MaxLongVarcharSize=8190;Debug=0;CommLog=0;UseDeclareFetch=0;TextAsLongVarchar=1;UnknownsAsLongVarchar=0;BoolsAsChar=1;Parse=0;ExtraSysTablePrefixes=;LFConversion=1;UpdatableCursors=1;TrueIsMinus1=0;BI=0;ByteaAsLongVarBinary=1;UseServerSidePrepare=1;LowerCaseIdentifier=0;XaOpt=1"
        try:
            connection= pyodbc.connect(connection_string)
        except pyodbc.OperationalError:    # ConnectionRefusedError: 
            print("ODBC connection failed.",connection_string[:80],".... Loading saved dfs.")
            status_report+="\nODBC connection failed."+str(connection_string[:80])+".... Loading saved dfs.\n"
            sales_df=pd.read_pickle(load_dir+load_file)   #'./sales_df.pkl')
            stock_levels_df=pd.read_pickle('./stock_levels_df.pkl')
       
            print("ODBC firehose: Ready sales_df",sales_df.shape,"stock_levels_df",stock_levels_df.shape)
            status_report+="\nODBC firehose: Ready sales_df:"+str(sales_df.shape)+" stock_levels_df:"+str(stock_levels_df.shape)+"\n"
         #   connection.close()
            return sales_df,stock_levels_df
        
       
        
  #     connection= pyodbc.connect(connection_string)
        if connection:
            #cursor = connection.cursor()
            #cursor = connection.cursor()
            #cursor.execute("SELECT customer_master.cat, sales_trans.code, sales_trans.costval, sales_trans.doctype, sales_trans.docentrynum, sales_trans.linenumber, sales_trans.location, sales_trans.product, sales_trans.productgroup, sales_trans.qty, sales_trans.refer, sales_trans.salesrep, sales_trans.salesval, sales_trans.territory, sales_trans.date, customer_master.glset, customer_master.specialpricecat FROM public.customer_master customer_master, public.sales_trans sales_trans WHERE customer_master.code = sales_trans.code AND ((sales_trans.date>={d '2019-09-01'}))")
            #for row in cursor.fetchall():
            #    print (row)
            sql = "SELECT customer_master.cat, sales_trans.code, sales_trans.costval, sales_trans.doctype, sales_trans.docentrynum, sales_trans.linenumber, sales_trans.location, sales_trans.product, sales_trans.productgroup, sales_trans.qty, sales_trans.refer, sales_trans.salesrep, sales_trans.salesval, sales_trans.territory, sales_trans.date, customer_master.glset, customer_master.specialpricecat, customer_master.sort FROM public.customer_master customer_master, public.sales_trans sales_trans WHERE customer_master.code = sales_trans.code AND ((sales_trans.date>={d '2018-01-01'}))"
            print("Beerenberg ODBC firehose DSN connection successful:",connection_string[:80],"....")
            status_report+="\nBeerenberg ODBC firehose DSN connection successful:"+str(connection_string[:80])+"....\n"
            print("load sales_df - SQL query=",sql[:80],"....")
            status_report+="\nload sales_df - SQL query="+str(sql[:80])+"....\n"
            sales_df = pd.read_sql(sql,connection)
     
           # print("ODBC complete.  Indexing\n",sales_df.shape,"rows.")
            print("Query complete.  Cleanup:\n",sales_df.shape,"rows.")
            status_report+="\nQuery complete.  Cleanup:\n"+str(sales_df.shape)+" rows.\n"
            sales_df.fillna(0,inplace=True)
            sales_df=sales_df[(sales_df.date.isnull()==False)]
            
  
           # sales_df.drop_duplicates(keep='first', inplace=True)
            sales_df['date']=pd.to_datetime(sales_df['date'])
            sales_df.sort_values(by=['date'], inplace=True, ascending=False)
       
            sales_df["period"]=sales_df.date.dt.to_period('D')
            sales_df['product']=sales_df['product'].astype('string')
            sales_df['productgroup']=sales_df['productgroup'].astype('string')   
            sales_df['code']=sales_df['code'].astype('string')
            sales_df['refer']=sales_df['refer'].astype('string')
            sales_df['sort']=sales_df['sort'].astype('string')
            sales_df['salesrep']=sales_df['salesrep'].astype('string')
            sales_df['territory']=sales_df['territory'].astype('string')
            sales_df['location']=sales_df['location'].astype('string')
            sales_df['glset']=sales_df['glset'].astype('string')
         #   sales_df=sales_df[(sales_df['product']!="0")]
            sales_df=sales_df[(sales_df['product']!="")]
        #    print("1",sales_df.dtypes)
            sales_df['cat']=pd.to_numeric(sales_df.cat, errors='coerce')   #.sort_values())
          #  sales_df['cat']=sales_df.cat.astype('float64')   #,'specialpricecat':'float','period':'category'})  #,copy=True)
        #    sales_df['specialpricecat']=sales_df.specialpricecat.astype('float64')   #,'specialpricecat':'float','period':'category'})  #,copy=True)
            sales_df['specialpricecat']=pd.to_numeric(sales_df.specialpricecat, errors='coerce')
            #   sales_df['cat']=sales_df['cat'].astype(float)   
         #   sales_df['specialpricecat']=sales_df['specialpricecat'].astype(float)
       #     print("2",sales_df.dtypes)
            sales_df['period'] = sales_df['period'].astype('category')
    
            sales_df.set_index('date',inplace=True,drop=False) 
        #    print("3",sales_df.dtypes)
        #    sales_df.set_index('date',inplace=True)
        #    sales_df.sort_index(ascending=True,inplace=True)
         #   print("sales_df finished=\n",sales_df,sales_df.shape)  
        #    print("pickling to",dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['raw_savefile'])
        #    sales_df.to_pickle(dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['raw_savefile'],protocol=-1)  

    
            print("sales_df finished\n",sales_df.dtypes,sales_df.shape)
            status_report+="\nsales_df finished:\n"+str(sales_df.dtypes)+" , "+str(sales_df.shape)+"\n"

      #      sql="SELECT product_master.location, product_master.code, product_master.productgroup, product_master.description, product_master.qtyinstock FROM admin.product_master product_master WHERE (product_master.productgroup In ('01','02','03','05','08','10','11','12','13','14','15','16','17','18','20','21','22','23','31','33','36','041','042','043','I45','15')) AND (product_master.qtyinstock>0) AND (product_master.inactive=0) AND (product_master.location In ('SA','HFG','HRM')) OR (product_master.location='SHP') AND (product_master.code='GP6X30') ORDER BY product_master.productgroup, product_master.code"
            sql="SELECT product_master.location, product_master.code, product_master.productgroup, product_master.description, product_master.qtyinstock FROM admin.product_master product_master WHERE (product_master.productgroup In ('01','02','03','05','08','10','11','12','13','14','15','16','17','18','20','21','22','23','31','33','36','041','042','043','I45','15')) AND (product_master.qtyinstock>0) AND (product_master.location In ('SA','HFG','HRM')) OR (product_master.location='SHP') AND (product_master.code='GP6X30') ORDER BY product_master.productgroup, product_master.code"


      #      print("Beerenberg ODBC firehose DSN connection successful.",connection_string[:60],"....")
            print("stock_levels_df - SQL query=",sql[:80],"....")
            status_report+="\nstock_levels_df - SQL query="+str(sql[:80])+"....\n"
            stock_levels_df = pd.read_sql(sql,connection)
       
            print("Query complete.  Cleanup:\n",stock_levels_df.shape,"rows.")
            status_report+="\nQuery complete.  Cleanup:\n"+str(stock_levels_df.shape)+" rows.\n"
            stock_levels_df['location']=stock_levels_df['location'].astype('string')
            stock_levels_df['code']=stock_levels_df['code'].astype('string')
            stock_levels_df['productgroup']=stock_levels_df['productgroup'].astype('string')
            stock_levels_df['description']=stock_levels_df['description'].astype('string')
            print("stock_levels_df finished\n",stock_levels_df.dtypes,stock_levels_df.shape)
            status_report+="\nstock_levels_df finished\n"+str(stock_levels_df.dtypes)+" , "+str(stock_levels_df.shape)
         #   print(stock_levels_df.dtypes,stock_levels_df.shape)
            connection.close()
            print("OBDC connection closed.",connection_string[:80],"....")
            status_report+="\nOBDC connection closed."+str(connection_string[:80])+"....\n"
            return sales_df,stock_levels_df
        else:
            print("Beerenberg ODBC firehose connection error. ODBC on",connection_string[:340],"....")
            status_report+="\nBeerenberg ODBC firehose connection error. ODBC on "+str(connection_string[:340])+"....\n"
            return pd.DataFrame([]),pd.DataFrame([])



    def load_production_df(self,load_dir,load_file):
     #   os.chdir("/home/tonedogga/Documents/python_dev")
        status_report=""
        connection_string="DSN=production;DATABASE=befa;SERVER=bbapp01.beerenberg.local;PORT=5432;UID=guy;;SSLmode=disable;ReadOnly=0;Protocol=7.4;FakeOidIndex=0;ShowOidColumn=0;RowVersioning=0;ShowSystemTables=0;Fetch=100;UnknownSizes=0;MaxVarcharSize=255;MaxLongVarcharSize=8190;Debug=0;CommLog=0;UseDeclareFetch=0;TextAsLongVarchar=1;UnknownsAsLongVarchar=0;BoolsAsChar=1;Parse=0;ExtraSysTablePrefixes=;LFConversion=1;UpdatableCursors=1;TrueIsMinus1=0;BI=0;ByteaAsLongVarBinary=1;UseServerSidePrepare=1;LowerCaseIdentifier=0;XaOpt=1"
        try:
            connection= pyodbc.connect(connection_string)
        except pyodbc.OperationalError:  
            print("ODBC connection failed",connection_string[:80],".... Loading saved dfs.")
            status_report+="\nODBC connection failed."+str(connection_string[:80])+".... Loading saved dfs.\n"
            production_df=pd.read_pickle(load_dir+load_file)
            schedule_df=pd.read_pickle(load_dir+'schedule_df.pkl')
            recipe_df=pd.read_pickle(load_dir+'recipe_df.pkl')
            print("ODBC firehose: Ready production_df",production_df.shape,"schedule_df",schedule_df.shape,"recipe_df",recipe_df.shape)
            status_report+="\nODBC firehose: Ready production_df "+str(production_df.shape)+" schedule_df "+str(schedule_df.shape)+" recipe_df "+str(recipe_df.shape)+"\n"
        #    connection.close()
            return production_df,schedule_df,recipe_df
        
        if connection:
            #cursor = connection.cursor()
            #cursor = connection.cursor()
            #cursor.execute("SELECT customer_master.cat, sales_trans.code, sales_trans.costval, sales_trans.doctype, sales_trans.docentrynum, sales_trans.linenumber, sales_trans.location, sales_trans.product, sales_trans.productgroup, sales_trans.qty, sales_trans.refer, sales_trans.salesrep, sales_trans.salesval, sales_trans.territory, sales_trans.date, customer_master.glset, customer_master.specialpricecat FROM public.customer_master customer_master, public.sales_trans sales_trans WHERE customer_master.code = sales_trans.code AND ((sales_trans.date>={d '2019-09-01'}))")
            #for row in cursor.fetchall():
            #    print (row)
            sql = "SELECT joboutput.jobid, joboutput.qtybatches,productionschedule.reworkin, productionschedule.reworkout,joboutput.qtyunits,productionschedule.manufactdate,product.code,to_date(productionschedule.manufactdate,'YYYYMMDD') FROM public.joboutput joboutput, public.product product, public.productionschedule productionschedule WHERE joboutput.jobid = productionschedule.jobid AND product.productid = joboutput.productid AND ((productionschedule.attupdmaterials = 'true')) ORDER BY to_date(productionschedule.manufactdate,'YYYYMMDD')"
            print("Beerenberg ODBC firehose DNS connection successful.",connection_string[:80],"....")
            status_report+="\nBeerenberg ODBC firehose DSN connection successful:"+str(connection_string[:80])+"....\n"
            print("production_df - SQL query=",sql[:80],"....")
            status_report+="\nload production_df - SQL query="+str(sql[:80])+"....\n"
            production_df = pd.read_sql(sql,connection)
       
            print("Query complete.  Cleanup:\n",production_df.shape,"rows.")
            status_report+="\nQuery complete.  Cleanup:\n"+str(production_df.shape)+" rows.\n"
            production_df['code']=production_df['code'].astype('string')
            production_df['to_date']=pd.to_datetime(production_df['to_date'])
            production_df['manufactdate']=pd.to_datetime(production_df['manufactdate'],infer_datetime_format=True)
         #   print(production_df,production_df.dtypes,production_df.shape)
 

            print("production_df finished=\n",production_df.dtypes,production_df.shape)
            status_report+="\nstock_levels_df finished\n"+str(production_df.dtypes)+" , "+str(production_df.shape)

            sql="SELECT product.code, recipe.code, recipe.yldboiloff, recipe.yldlineloss, product.weight FROM public.product product, public.recipe recipe WHERE recipe.recipeid = product.recipeid AND ((product.template='true')) ORDER BY product.code"
            print("schedule_df - SQL query=",sql[:80],"....") 
            status_report+="\nload schedule_df - SQL query="+str(sql[:80])+"....\n"
            schedule_df = pd.read_sql(sql,connection)

            print("Query complete.  Cleanup:\n",schedule_df.shape,"rows.")
            status_report+="\nQuery complete.  Cleanup:\n"+str(schedule_df.shape)+" rows.\n"
          #  schedule_df = schedule_df.rename(columns = {"code":"productcode"})   #,"code":"recipecode"})
            schedule_df['productcode']=schedule_df.iloc[:,0].astype('string')
            schedule_df['recipecode']=schedule_df.iloc[:,1].astype('string')
            schedule_df.drop(columns=['code','code'],axis=1,inplace=True)
            cols = list(schedule_df.columns)
            cols = cols[-2:] + cols[:-2]
            schedule_df = schedule_df[cols]
          #  print(schedule_df,schedule_df.dtypes)
     
            print("schedule_df finished=\n",schedule_df.dtypes,schedule_df.shape)
            status_report+="\nschedule_df finished\n"+str(schedule_df.dtypes)+" , "+str(schedule_df.shape)
            sql="SELECT recipe.code, recipeingredient.qty FROM public.recipe recipe, public.recipeingredient recipeingredient WHERE recipeingredient.recipid = recipe.recipeid AND ((recipe.template='true')) ORDER BY recipe.code"
            print("recipe_df - SQL query=",sql[:80],"....")   
            status_report+="\nload recipe_df - SQL query="+str(sql[:80])+"....\n"
            recipe_df = pd.read_sql(sql,connection)

            print("Query complete.  Cleanup:",recipe_df.shape,"rows.")
            status_report+="\nQuery complete.  Cleanup:\n"+str(recipe_df.shape)+" rows.\n"
            recipe_df['code']=recipe_df['code'].astype('string')
            print("recipe_df finished=\n",recipe_df.dtypes,recipe_df.shape)
            status_report+="\nrecipe_df finished\n"+str(recipe_df.dtypes)+" , "+str(recipe_df.shape)




            print("OBDC connection closed.",connection_string[:80],"....")
            status_report+="\nOBDC connection closed."+str(connection_string[:80])+"....\n"
            connection.close()
            return production_df,schedule_df,recipe_df
        else:
            print("Beerenberg ODBC firehose connection error. ODBC on",connection_string[:340],"....")
            status_report+="\nBeerenberg ODBC firehose connection error. ODBC on "+str(connection_string[:340])+"....\n"
            return pd.DataFrame([]),pd.DataFrame([]),pd.DataFrame([])




#def main():
    def load_dfs(self):
          os.chdir("/home/tonedogga/Documents/python_dev/bbtv1")
          sales_df,stock_levels_df=pd.DataFrame([]),pd.DataFrame([])
         # fr=odbc_firehose_class()
          sales_df,stock_levels_df=self.load_sales_df()
           #  sales_df.to_pickle(dd2.dash2_dict['sales']['save_dir']+dd2.dash2_dict['sales']['raw_savefile'],protocol=-1)    
      
          if sales_df is not None: 
              sales_df.to_pickle("./sales_df.pkl",protocol=-1)  
        #      sales_df.to_pickle("..\..\..\..\..\Dropbox\code\sales_df.pkl",protocol=-1)  
      
          else:
              print("sales_df is empty. no pkl saved.")
       
       
        
          if stock_levels_df is not None:  
               stock_levels_df.to_pickle("./stock_levels_df.pkl",protocol=-1)    
           #    stock_levels_df.to_pickle("..\..\..\..\..\Dropbox\code\stock_levels_df.pkl",protocol=-1)  
          else:
               print("stock_levels_df is empty. no pkl saved.")
        
      
      
      
         #  production_df,schedule_df,recipe_df=self.load_production_df() 
         #  if production_df is not None: 
         #       production_df.to_pickle("./production_df.pkl",protocol=-1)    
         #  #     production_df.to_pickle("..\..\..\..\..\Dropbox\code\production_df.pkl",protocol=-1)  
         #  else:
         #       print("production_df is empty. no pkl saved.")
       
       
      
      
         #  if schedule_df is not None:     
         #       schedule_df.to_pickle("./schedule_df.pkl",protocol=-1)    
         #    #   schedule_df.to_pickle("..\..\..\..\..\Dropbox\code\schedule_df.pkl",protocol=-1)  
         #  else:
         #       print("schedule_df is empty. no pkl saved.")
       
      
         #  if recipe_df is not None:
         #       recipe_df.to_pickle("./recipe_df.pkl",protocol=-1)
         # #      recipe_df.to_pickle("..\..\..\..\..\Dropbox\code\recipe_df.pkl",protocol=-1)
         #  else:
         #       print("recipe_df is empty. no pkl saved.")
       
          return sales_df,production_df


#if __name__ == "__main__":
#    main()
