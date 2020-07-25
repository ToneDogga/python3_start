#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:38:28 2020

@author: tonedogga
"""
from collections import namedtuple
#from collections import defaultdict
import pandas as pd



chart_names=["chart1","cr2","ct3"]
column_names=["col1","col2","col3"]


#chart = namedtuple("chart", ["name", "kind","stacked","second_y"])
data_column = namedtuple("data_column", ["active","name", "kind","colour","stacked","second_y"])





# 

#  chart_dict is a dictionary of all the charts created plus the report_type_dict to decode it
# at the end it is picked so it can be loaded

chart_elements_dict={column_names[0]:data_column(True,column_names[0],"kind","c1",False,False),
            column_names[1]:data_column(False,column_names[1],"kind2","c2",False,False),
            column_names[2]:data_column(False,column_names[2],"kind3","c3",False,False)}

#print("\nce=",chart_elements_dict)

no_of_charts=5

#chart_dict={0:chart_elements_dict[0]}
# fill out the entire charts with True for all columns 
charts={chart_name:{(chart_name,chart_element):chart_elements_dict[chart_element] for chart_element in chart_elements_dict.keys()} for chart_name in chart_names}
#chart_dict = defaultdict(lambda x: chart_elements_dict[x])


#print("\ncd=",charts)
#####################################33
#  Set values
charts['ct3'][('ct3','col1')]=data_column(False,'testx1','line','blue',True,False)
charts['ct3'][('ct3','col2')]=data_column(False,'testx2','line','blue',True,False)
charts['ct3'][('ct3','col3')]=data_column(False,'testx3','line','blue',True,False)
charts['cr2'][('cr2','col1')]=data_column(False,'testx4','line','blue',True,False)
charts['cr2'][('cr2','col2')]=data_column(False,'testx5','line','blue',True,False)
charts['cr2'][('cr2','col3')]=data_column(False,'testx6','line','blue',True,False)
#charts[('ct3','col3')]=data_column(False,'testx','line','blue',True,False)

#print("\ncd2=",charts)

# #charts={j:chart_dict[j] for j in chart_dict.keys()}
# print("\n\n")

# print(charts['cr2'])
# print("\n",charts['cr2']['col2'])
# print("\n",charts['cr2']['col2'][2])
# #print(data_column('col1'))
# #print("ct=",charts)
# #report_dict = defaultdict(lambda: defaultdict(dict))
# chart=[]

# for c in charts.keys():
#  #   f=chart_elements_dict['col1']   #.keys()
#   #  g=data_column('name',...)
#  #   print("c,f",c,f._fields)
#     new_chart=pd.DataFrame(charts[c],index=chart_elements_dict['col1']._fields).T
#     print("new chart=\n",c,new_chart)
#     chart.append(new_chart 
h=[pd.DataFrame(charts[c],index=chart_elements_dict['col1']._fields) for c in charts.keys()]
#for c in charts.keys():
#    pd.DataFrame(charts[c],index=[c,chart_elements_dict['col1']._fields]).stack() 
#print("h=\n",h)

#chart['ct3']['col3']=('testx',True,'col1','line','blue',True,False)
#df=df.set_index('market_name', append=True)

#print("h=\n",h[1])
#h['new']="new"
#df=pd.MultiIndex.from_frame(pd.concat(h,axis=1),levels=['chart','col'])
df=pd.concat(h,axis=1)
#print("df=\n",df)
#df2=pd.MultiIndex.from_frame(df,names=['chart','2','3'])
#print("df2=\n",df2)

df.columns.set_names('chart_name', level=0,inplace=True)
df.columns.set_names('column_name', level=1,inplace=True)
print("df=\n",df)
print("charts=\n",charts)
#df.loc['active',('chart1','col2')]='test'

#print("df=\n",df)


#df=df.set_index('market_name', append=True)
#df=df.set_index('market_name', append=True)
#print("df2=\n",df2)
#df2=pd.MultiIndex.from_frame(pd.concat(h,axis=0),names=['chart','2','3'])
#print("df3=\n",df2)
#k={h[1]}
#print(k)
#print("\n",h[1].loc[:,h[1]['active'==True]])
 #   f=chart_elements_dict['col1']   #.keys()
  #  g=data_column('name',...)
 #   print("c,f",c,f._fields)
    
#print("\n\n\nchart=\n",chart)

#report_dict[0][1]="chart1"
# chart 0, column 1
#report_dict[0][1]=chart_dict[1]
#report_dict[1][2]=chart_dict[2]

#chart = namedtuple("chart", ["name",chart_elements_dict])

#chart['test1'][1]="yes"

#print("\n",report_dict)

# chart_dict['a']['b']['c'] = 'foo'
# chart_dict['a']['b']['d'] = 'bar'

# print(chart_dict)

#t['a']['b']['c'] = 'foo'
#t['a']['b']['d'] = 'bar'

#print(t)


# value is (brand,specialpricecat, productgroup, product,type,name)
# type is 0 off promo. 1 is on promo, 2 is total of both, 3 is invoiced total


#product_type = namedtuple("product_type", ["brandno","customercat", "productgroup","product","type","name"])
#report_type_dict={0:"dictionary",
#                        3:"dataframe",
#                        5:"spreadsheet",
#                        6:"pivottable",
#                        8:"chart_filename"}
