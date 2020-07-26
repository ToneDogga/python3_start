#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:38:28 2020

@author: tonedogga
"""
from collections import namedtuple
#from collections import defaultdict
import pandas as pd
import pickle
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



class stack:
    def __init__(self):   
        self.stack_df_savename="stack_df.pkl"
        self.stack_plotp_savename="stack_plotp.pkl"
        self.__version__="0.1.0"

        with open(self.stack_df_savename, 'wb') as f:        
            pickle.dump([], f,protocol=-1)   
        with open(self.stack_plotp_savename, 'wb') as f:        
            pickle.dump([], f,protocol=-1)   
           
        return
    
 
    def push_stack(self,stack_df_element,stack_plotp_element):
        with open(self.stack_df_savename,"rb") as f:
            stack_df=pickle.load(f)
        stack_df.append(stack_df_element)   
        with open(self.stack_plotp_savename,"rb") as f:
            stack_plotp=pickle.load(f)
        stack_plotp.append(stack_plotp_element)   

        with open(self.stack_df_savename, 'wb') as f:        
            pickle.dump(stack_df, f,protocol=-1)
        with open(self.stack_plotp_savename, 'wb') as f:        
            pickle.dump(stack_plotp, f,protocol=-1)
    
        return len(stack_df)    
    
    
    def stack_len(self):
    #     if os.path.isfile('./stack.pkl'):
        with open(self.stack_df_savename,"rb") as f:
            stack_df=pickle.load(f)
        return len(stack_df)   
    
    
    
    def pop_all_plotps_stack(self):
        with open(self.stack_plotp_savename,"rb") as f:
            stack_plotp=pickle.load(f)
        with open(self.stack_plotp_savename, 'wb') as f:        
            pickle.dump([], f,protocol=-1)
        return stack_plotp  
    
    
    def pop_all_dfs_stack(self):
        with open(self.stack_df_savename,"rb") as f:
            stack=pickle.load(f)
        with open(self.stack_df_savename, 'wb') as f:        
            pickle.dump([], f,protocol=-1)   
        return stack    



#################################################3


s=stack()

chart_names=["chart1","chart2","chart3_three"]
with open("measures_savename.pkl", 'rb') as f:
    column_names = pickle.load(f)

scan_dict_savename="scan_dict.pkl"
#column_names=["col1","col2","col3"]
with open(scan_dict_savename, 'rb') as g:
    scan_data_dict = pickle.load(g)

#print(b)
#print("\n\n")
# latest_date=scan_data_dict['final_df'].index[-1]
# print("\nBeerenberg scan data at:",latest_date)
# print("+++++++++++++++++++++==================\n")


df=scan_data_dict['final_df']
print(df)


#chart = namedtuple("chart", ["name", "kind","stacked","second_y"])
data_column = namedtuple("data_column", ["active","title","y_title", "kind","style","colour","stacked","secondary_y"])

chart_elements_dict={column_names[m]:data_column(True,column_names[m],column_names[m][:20],"bar","r-","blue",False,False) for m in range(len(column_names))}
charts={(chart_name,chart_element):chart_elements_dict[chart_element] for chart_name in chart_names for chart_element in chart_elements_dict.keys()}
#####################################33
#  Set values
charts[('chart1','Units (000)')]=data_column(True,'terstx1',"y_title",'line',"g:","",False,False)
charts[('chart1','Dollars (000)')]=data_column(True,'testxsdfgsd1',"yg_title",'bar',"red","red",True,True)
#charts[('chart1','col3')]=data_column(True,'tesgsd1',"yfg_title",'line',"b-","",True,False)
#charts[('chart1','col4')]=data_column(True,'tfffesgsd1',"yfffg_title",'line',"k-","",False,False)

# charts['ct3'][('ct3','col2')]=data_column(False,'testx2','line',"ls",'blue',True,False)
# charts['ct3'][('ct3','col3')]=data_column(False,'testx3','line',"ls",'blue',True,False)
# charts['cr2'][('cr2','col1')]=data_column(False,'testx4','line',"ls",'blue',True,False)
# charts['cr2'][('cr2','col2')]=data_column(False,'testx5','line',"ls",'blue',True,False)
# charts['cr2'][('cr2','col3')]=data_column(False,'testx6','line',"ls",'blue',True,False)
# #charts[('ct3','col3')]=data_column(False,'testx','line','blue',ls,True,False)

#print("\ncd2=",charts.keys())
fields=chart_elements_dict[column_names[0]]._fields
print("\ncharts=",chart_names)
print("\ncolumn names=",column_names)
print("\nfields=",fields)
#h=[pd.DataFrame(charts,index=fields) for c in charts.keys()]

#df=pd.concat(h,axis=1)
#h=[pd.DataFrame(charts,index=fields) for c in charts.keys()]


def convert_dict_to_df(charts,fields):
    df=pd.concat([pd.DataFrame(charts,index=fields) for c in charts.keys()],axis=1)
    df.columns.set_names('chart_name', level=0,inplace=True)
    df.columns.set_names('column_name', level=1,inplace=True)
    return df.T


def plot_df(df,plotp_list):
  #  fig = plt.figure() # Create matplotlib figure
   # d=0
    if isinstance(df, pd.Series):
        df=df.to_frame()
    df['date'] = pd.to_datetime(df.index,format="%d/%m/%y",exact=False)
    df = df.set_index('date', append=True)
    df.index = df.index.droplevel(0)
 #   print("plotp_list",plotp_list)
 #   print("df=\n",df)
  #  df_prop=pd.DataFrame(plotp)
  #  print("\ndfp=\n",df_prop)
  #  print(df_prop.loc[:,'kind'].tolist())
    ax = plt.gca()   #add_subplot(111) # Create matplotlib axes

    for p in plotp_list:
     #   print(d,"df=\n",df.iloc[:,d])
    #    print("p=\n",p)
        df.plot(ax=ax,kind=p.kind,stacked=p.stacked,use_index=False,legend=False,style=p.style,title=p.title,secondary_y=p.secondary_y)    
  #      d+=1
        ax.set_ylabel(p.y_title,fontsize=8)   


        ticklabels = ['']*len(df.index)
         # Every 4th ticklable shows the month and day
        ticklabels[::8] = [item.strftime('%b') for item in df.index[::8]]
         # Every 12th ticklabel includes the year
        ticklabels[10::26] = [item.strftime('\n\n%Y') for item in df.index[10::26]]
   

 
     # #   plt.ylabel("sales (000)",fontsize=7)
     #  #  plt.xlabel("",fontsize=7)

        plt.gca().xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
     # #   plt.xtick_params(pad=100)
        plt.gcf().autofmt_xdate()





#     ax.xaxis.set_minor_locator(mpl.dates.MonthLocator())
#     ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%b'))
# #   ax.xtick_params(pad=20)
#     ax.xaxis.set_major_locator(mpl.dates.YearLocator())
#     ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))

     #  ax2.set_ylabel('Price',fontsize=8)
    ax.set_xlabel("",fontsize=7)
    plt.gca().legend(fontsize=7)  #,loc='best')
    #   fig.legend(fontsize=7,loc='best')
    return


def plot_column(df,plotp):
    #  data_column(active=False, title='Units (000)', kind='line', linestyle='-', colour='blue', stacked=False, second_y=False)
    
  #  print("series=\n",df,plotp.active,"stack len",s.stack_len())

    if plotp.active:
        if plotp.stacked:
            print("add to stack stack length=",s.push_stack(df,plotp))
        else: 
            if s.stack_len()>0:
                plot_df(pd.concat(s.pop_all_dfs_stack(),axis=1),s.pop_all_plotps_stack())
            else:
                plot_df(df,[plotp])
            
    return    



def remove_levels(df,n):
    df=df.T
    levels_to_remove = df.index.nlevels-n
    for _ in range(levels_to_remove):
        df=df.droplevel(level=0)
   # print("remove level df=\n",df)

    return df.T



  

plotp_df=convert_dict_to_df(charts,fields)
#print("\nplotp_df=\n",plotp_df)
#print("\ndf.T=\n",df.T)
#print("charts=\n",charts)
#print(df.T.index)

#print("\nggg=",charts[('chart1','Units (000)')])
#print("\nmmm=",charts[('chart1','col1')].stacked)
#print("\nmm=",charts[('chart1','col1')].style)

#gf=(charts.active==True)
#pop_whole_stack()
#print("df=",df)

#print("stack len=",s.stack_len())      

i=0
for c in charts.keys():
    
 #   print("c",c[0],c[1])
  #  print(remove_levels(df,1))
    plot_column(remove_levels(df,1).iloc[:,i],charts[c])
    i+=1
    plt.show()
plt.close('all')    