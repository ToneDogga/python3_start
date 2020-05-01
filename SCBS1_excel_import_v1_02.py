#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:12:51 2020

@author: tonedogga
"""
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
#import SCBS0 as c


#filenames=["NAT-raw240420_no_shop_WW_Coles.xlsx"]
#filenames=["allsalestrans2020.xlsx","allsalestrans2018.xlsx"]
 


def quotes(test2):
    test2=str(test2)
    quotescount=0
    k=0
    testlen=len(test2)
    while k< testlen: 
    #    print("k=",k,"testlen",testlen)
        if (test2[k].isupper() or test2[k].isnumeric()) and quotescount%2==0:  # even
            test2=test2[:k]+'"'+test2[k:]
            testlen+=1
            k+=1
            quotescount=+1
        
        # closing quotes
        if (test2[k]==" " and quotescount%2==1):
                test2=test2[:k]+'"'+test2[k:]
                testlen+=1
                k+=1
                quotescount+=1
        k+=1
        
    if quotescount%2==1:  # odd
        test2=test2+'\"'

    
    return(test2)       




def main(c):
    filenames=c.filenames
    queryfilename=c.queryfilename




    print("\n\n****  Sales Crystal Ball Stack (SCBS) Excel loader and query builder 30 April 2020 by Anthony Paech")
    print("===================================================================================================")
    print("\nloading excel '",filenames,"' ......")
    
    print("load:",filenames[0])
    df=pd.read_excel(filenames[0],-1)  # -1 means all rows   
    print("df size=",df.shape)
    for filename in filenames[1:]:
        print("load:",filename)
        new_df=pd.read_excel(filename,-1)  # -1 means all rows   
        print("new df size=",new_df.shape)
        df=df.append(new_df)
        print("df size=",df.shape)
    
    
    df.fillna(0,inplace=True)
    
    #print(df)
    print("drop duplicates")
    df.drop_duplicates(keep='first', inplace=True)
    print("after drop duplicates df size=",df.shape)
    
    
       
    df["period"]=df.date.dt.to_period('D')
    df['period'] = df['period'].astype('category')
    
    print("loading query list '",queryfilename,"'")
    query_dict = pd.read_excel(queryfilename, index_col=0, header=0,  skiprows=0).T.to_dict()  #  doublequote=False
    
    print("\nquery dict=\n",query_dict,"\n")
    
    table_list = defaultdict(list)
    q=0
    
    for query_name in query_dict.keys():
    #for qname in query_table.index:
        table_list[q].append(query_name)
        table_list[q].append(pd.pivot_table(df.query(quotes(query_dict[query_name]['condition'])), values='qty', index=query_dict[query_name]['index_code'].split(),columns=['period'], aggfunc=np.sum, margins=False,dropna=False,observed=False, fill_value=0).T) 
        q+=1
        
    table_dict = dict((k, tuple(v)) for k, v in table_list.items())  #.iteritems())
    
    
    #print("\n table dict=\n",table_dict)
    
    with open("tables_dict.pkl","wb") as f:
        pickle.dump(table_dict, f,protocol=-1)
        
    #querynames=[table_dict[k][0] for k in table_dict.keys()]    
    print("pickling",len(table_dict),"tables (",[table_dict[k][0] for k in table_dict.keys()],")")

    
##################################

# print("\n\ntest unpickling")  
# with open("tables_dict.pkl", "rb") as f:
#     testout1 = pickle.load(f)
#   #  testout2 = pickle.load(f)
# qnames=[testout1[k][0] for k in testout1.keys()]    
# print("unpickled",len(testout1),"tables (",qnames,")")

# #query_dict2=testout1['query_dict']
# #print("table dict two unpickled=",testout1)

# #df2=testout1.keys()
# #print(testout1.keys())
# #print(testout1.values())
# #print(testout1[1])  
# print(testout1[0][1]) 
# print(testout1[1][1])
# #print(testout1.items())
# #for query_count in testout1.keys():
# #    print(query_count,"out table=\n",testout1[query_count])
    
# ########################################



    return


if __name__ == '__main__':
    main()

