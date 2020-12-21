#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 09:35:04 2020

@author: tonedogga
"""
import os
import pickle
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np

from matplotlib import pyplot, dates
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, ion, show

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter

import glob
import imageio
import cv2
import shutil
from p_tqdm import p_map,p_umap
import seaborn as sns
import calendar
 
import datetime as dt
from datetime import datetime,timedelta
#import dash2_root
import dash2_dict as dd2

#dash=dash2_root.dash2_class()






class animate_engine(object):  
    def _resize(self,filename):
        fname = filename   #os.path.basename(filename)
        input_image  = imageio.imread(fname, format='PNG-FI')
        output_image = cv2.resize(input_image, dd2.dash2_dict['sales']['plots']['animation_resize']) #(1440,800),  #(1888, 1600))   #(256, 256))    ((944,800))
        imageio.imwrite(fname, output_image, format='PNG-FI')   # 48 bit
        return

    
      
  
   
 

            
    def _clear_old_plots(self,clear_dir):   
        # clear old plots
        files = glob.glob(clear_dir+"*.png")  #dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+'*.png')
      #  print("1files to del:",files)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("file delete Error: %s : %s" % (f, e.strerror))
                
                
 
             
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
   
    
   #  def preprocess(self,df,mat):
   #      # rename qolumns
       
   #     # df.rename(columns=rename_dict, inplace=True)
    
   #   #   df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
   #    #  df['date']=df.index
   #    #  df['mat']=df['salesval'].rolling(mat,axis=0).sum()
   #  #    print("rename collumsn")
        
   #     # df.rename(columns=rename_dict, inplace=True)
    
   # #     df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
   #    #  df['date']=df.index
   #  #    df['mat']=df['salesval'].rolling(window=mat,axis=0).sum()
   #    #  df=df[(df['mat']>=0)]
   #  #    print("rename collumsn")
   #      return df
    
    
    
    def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
    
       
    
    #  def _take_out_zeros(self,df,cols):
    #      # cols of a list of column names
    # #     df[cols]=df[cols].clip(lower=100.0)   #,axis=1)
    #    #  print("toz=\n",df)
    #      df[cols]=df[cols].replace(0, np.nan)
    #      return df
 
    
    
   
    def _add_trues_and_falses(self,df,cols):
        df[cols]=df[cols].replace(1,True)
        df[cols]=df[cols].replace(0,False)
        return df
    
   
       
    def _plot_brand_index(self,slices):   #tdf,output_dir,y_col,col_and_hue,savename):    
        tdf=slices['df']
        output_dir=slices['plot_dump_dir']
        y_col=slices['y_col']
        col_and_hue=slices['col_and_hue']
        savename=slices['savename']
        
   #     atdf=tdf[~np.isnan(tdf)].copy() # Remove the NaNs
   #     print("atoz=\n",atdf)
        tdf=tdf.astype(np.float64)
     
        tdf=self._add_trues_and_falses(tdf,col_and_hue[0])
        tdf=self._add_trues_and_falses(tdf,col_and_hue[1])
    #    tdf=self._take_out_zeros(tdf,y_col)
    #    tdf=self._take_out_zeros(tdf,col_and_hue[1])
        
        tdf[y_col].replace(0, np.nan,inplace=True)
      #  print("y_col",y_col,"toz[y_col]=\n",tdf[y_col])
 
    
    
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
        plt.close()
        return
     
      
 



    def _p_pareto_customer(self,slices):
        df=slices["df"]    
        k=slices['name']
        key=slices['key']
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]

#        print("pareto customer plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
   #     i_dict=df_dict.copy()
#        for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
        new_df=df.groupby(['code'],sort=False).sum()
   #     print("pareto customer",k,new_df,new_df.shape)
    #    print("pareto customer",k,new_df)
        if new_df.shape[0]>0:
            new_df=new_df[(new_df['salesval']>1.0)]
            new_df=new_df[['salesval']].sort_values(by='salesval',ascending=False)   
        #    new_df=new_df.droplevel([0])
    
            new_df['ccount']=np.arange(1,new_df.shape[0]+1)
            df_len=new_df.shape[0]
            
            ptt=new_df['salesval']
            ptott=ptt.sum()
            new_df['cumulative']=np.cumsum(ptt)/ptott
            new_df=new_df.head(top)
            
            fig, ax = pyplot.subplots()
            fig.autofmt_xdate()
          #  ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places

#                ax.yaxis.set_major_formatter('${x:1.0f}')
          #  ax.yaxis.set_tick_params(which='major', labelcolor='green',
          #           labelleft=True, labelright=False)

         #   ax.ticklabel_format(style='plain') 
         #   ax.yaxis.set_major_formatter(ScalarFormatter())
      
            #ax.ticklabel_format(style='plain') 
      #      ax.axis([1, 10000, 1, 100000])
            
            ax=new_df.plot.bar(y='salesval',ylabel="",fontsize=7,grid=False)
        #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
       #     axis.set_major_formatter(ScalarFormatter())
         #   ax.ticklabel_format(style='plain')
            ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places

            ax.yaxis.set_tick_params(which='major', labelcolor='green',labelleft=True, labelright=False)
            ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" customer $ ranking total dollars "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
         
         
            ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=True,style=["r-"],secondary_y=True)
            ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
            ax3 = ax.twiny() 
            ax4=new_df[['ccount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
            if df_len<=1:
                df_len=2
     
            
            ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
    
            self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_customer_$_ranking",output_dir)
            plt.close()
        else:
            print("pareto customer nothing plotted. no records for ",k,new_df)
 
        return








    def _p_pareto_product_dollars(self,slices):
        df=slices["df"]    
        k=slices['name']
        key=slices['key']
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]

 #       print("pareto product plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
     #   i_dict=df_dict.copy()
      #  for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
        new_df=df.groupby(['product'],sort=False).sum()

 #      print("pareto product dollars",k,new_df,new_df.shape)
        if new_df.shape[0]>0:
           new_df=new_df[(new_df['salesval']>1.0)]
           new_df=new_df[['salesval']].sort_values(by='salesval',ascending=False)   
       #    new_df=new_df.droplevel([0])
   
           new_df['pcount']=np.arange(1,new_df.shape[0]+1)
           df_len=new_df.shape[0]
           
           ptt=new_df['salesval']
           ptott=ptt.sum()
           new_df['cumulative']=np.cumsum(ptt)/ptott
           new_df=new_df.head(top)
           
           fig, ax = pyplot.subplots()
           fig.autofmt_xdate()
#               ax.yaxis.set_major_formatter('${x:1.0f}')

        #   ax.ticklabel_format(style='plain') 
        #   ax.yaxis.set_major_formatter(ScalarFormatter())
     
           #ax.ticklabel_format(style='plain') 
     #      ax.axis([1, 10000, 1, 100000])
           
           ax=new_df.plot.bar(y='salesval',ylabel="",fontsize=7,grid=False)
       #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
      #     axis.set_major_formatter(ScalarFormatter())
        #   ax.ticklabel_format(style='plain')
           ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 0 decimal places

           ax.yaxis.set_tick_params(which='major', labelcolor='green',labelleft=True, labelright=False)
           ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" product $ ranking total dollars "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
 
        
        
        
           ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=True,style=["r-"],secondary_y=True)
           ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
           ax3 = ax.twiny() 
           ax4=new_df[['pcount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
           if df_len<=1:
               df_len=2
    
           
           ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
   
           self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_product_$_ranking",output_dir)
           plt.close()
        else:
           print("pareto product dollars nothing plotted. no records for ",k,new_df)
     
        return





    def _p_pareto_product_units(self,slices):
        df=slices["df"]    
        k=slices['name']
        key=slices['key']
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]

     #   print("pareto product units plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
   #     i_dict=df_dict.copy()
      #  print("pareto product i_dict=\n",i_dict,"\n i_dict.items()=\n",i_dict.items())
     #   for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
        new_df=df.groupby(['product'],sort=False).sum()

  #     print("\n++++++pareto product units",k,new_df)
        if new_df.shape[0]>0:
           new_df=new_df[(new_df['qty']>1.0)]
           new_df=new_df[['qty']].sort_values(by='qty',ascending=False)   
       #    new_df=new_df.droplevel([0])
   
           new_df['pcount']=np.arange(1,new_df.shape[0]+1)
           df_len=new_df.shape[0]
           
           ptt=new_df['qty']
           ptott=ptt.sum()
           new_df['cumulative']=np.cumsum(ptt)/ptott
           new_df=new_df.head(top)
           
           fig, ax = pyplot.subplots()
           fig.autofmt_xdate()
#               ax.yaxis.set_major_formatter('${x:1.0f}')

        #   ax.ticklabel_format(style='plain') 
        #   ax.yaxis.set_major_formatter(ScalarFormatter())
     
           #ax.ticklabel_format(style='plain') 
     #      ax.axis([1, 10000, 1, 100000])
           
           ax=new_df.plot.bar(y='qty',ylabel="units",fontsize=7,grid=False)
       #        ax=ptt['total'].plot(x='product',ylabel="$",style="b-",fontsize=5,title="Last 90 day $ product sales ranking (within product groups supplied)")
      #     axis.set_major_formatter(ScalarFormatter())
        #   ax.ticklabel_format(style='plain')
           ax.yaxis.set_major_formatter(StrMethodFormatter('{x:1.0f}')) # 0 decimal places

           ax.yaxis.set_tick_params(which='major', labelcolor='green',labelleft=True, labelright=False)
           ax.set_title("["+self._clean_up_name(str(k))+"] Top "+str(top)+" product unit ranking total units "+str(int(ptott))+" total("+str(df_len)+")",fontsize=9)
 
           ax2=new_df.plot(y='cumulative',xlabel="",rot=90,fontsize=7,ax=ax,grid=True,style=["r-"],secondary_y=True)
           ax2.yaxis.set_major_formatter(ticker.PercentFormatter(1.0,0,"%"))
           ax3 = ax.twiny() 
           ax4=new_df[['pcount']].plot(use_index=True,ax=ax3,grid=False,fontsize=7,xlabel="",style=['w:'],legend=False,secondary_y=False)
           if df_len<=1:
               df_len=2
    
           
           ax4.xaxis.set_major_formatter(ticker.PercentFormatter(df_len-1,0,"%"))
   
           self._save_fig(self._clean_up_name(str(k))+"pareto_top_"+str(top)+"_product_units_ranking",output_dir)
           plt.close()
        else:
           print("pareto product units nothing plotted. no records for ",k,new_df)
        return

 


    def _p_plot_salesval(self,slices):   
         plot_df=slices["sliced_df"]    
         k=slices['name']
         key=slices['key']
         start=slices['start']   #+pd.offsets.Day(365)
      #  start_2yrs=slices['start']
         end=slices['end']
         latest_date=slices['end']
         output_dir=slices["plot_dump_dir"]

         styles1 = ['b-']
        # styles1 = ['bs-','ro:','y^-']
         linewidths = 1  # [2, 1, 4]
                 
         fig, ax = pyplot.subplots()
     #    fig.autofmt_xdate()
         ax=plot_df.plot(y='salesval_mat',grid=True,fontsize=6,style=styles1, lw=linewidths)
          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       
         ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places
 
         ax.set_title("["+self._clean_up_name(str(k))+"] $ sales moving total 365 days @:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
         ax.legend(title="",fontsize=6,loc="upper left")
         ax.set_xlabel("",fontsize=6)
         ax.set_ylabel("",fontsize=6)
         # ax.yaxis.set_major_formatter('${x:1.0f}')
         ax.yaxis.set_tick_params(which='major', labelcolor='green',
                       labelleft=True, labelright=False)
          
         self._save_fig(self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
         plt.close()
         return




    def _p_plot_qty(self,slices):   
         plot_df=slices["sliced_df"]    
         k=slices['name']
         key=slices['key']
         start=slices['start']   #+pd.offsets.Day(365)
      #  start_2yrs=slices['start']
         end=slices['end']
         latest_date=slices['end']
         output_dir=slices["plot_dump_dir"]

         styles1 = ['b-']
        # styles1 = ['bs-','ro:','y^-']
         linewidths = 1  # [2, 1, 4]
                 
         fig, ax = pyplot.subplots()
     #    fig.autofmt_xdate()
         ax=plot_df.plot(y='qty_mat',grid=True,fontsize=6,style=styles1, lw=linewidths)
          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       
         ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places
 
         ax.set_title("["+self._clean_up_name(str(k))+"] units moving total 365 days @:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
         ax.legend(title="",fontsize=6,loc="upper left")
         ax.set_xlabel("",fontsize=6)
         ax.set_ylabel("",fontsize=6)
         # ax.yaxis.set_major_formatter('${x:1.0f}')
         ax.yaxis.set_tick_params(which='major', labelcolor='green',
                       labelleft=True, labelright=False)
          
         self._save_fig(self._clean_up_name(str(k))+"_units_moving_total",output_dir)
         plt.close()
         return


   
    def _animate_plots_mp4(self,*,mp4_input_dir,mp4_fps,mp4_output_dir,mp4_output_filename,plot_output_dir):

        #---------------------------------------------------
    # only need once
       # imageio.plugins.freeimage.download()
    #---------------------------------------------------------------------

        filenames=sorted(glob.glob(mp4_input_dir+'*.png'))   #, key=os.path.getctime)
        #  add the same frame for start and end 60? times
        if len(filenames)>0:
            start_freeze=[filenames[0]]*dd2.dash2_dict['sales']['plots']['animation_start_freeze_frames']
            end_freeze=[filenames[-1]]*dd2.dash2_dict['sales']['plots']['animation_end_freeze_frames']
     #       print(start_freeze)
            
            lf=len(filenames)
              
            p_map(self._resize,filenames)
            filenames=start_freeze+filenames+end_freeze
     
           # print("\nCreating test.mp4")
            writer = imageio.get_writer(mp4_input_dir+mp4_output_filename, fps=mp4_fps)
          #  writer2 = imageio.get_writer(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4'], fps=mp4_fps)
    
            i=1
          #  print("Creating gif of",i,"/",lf,"plots....",end="\r",flush=True)
         
            for filename in filenames:
                 print("Creating mp4 of",i,"/",lf,"plots....",end="\r",flush=True)
                 fname = os.path.basename(filename)
                 writer.append_data(imageio.imread(mp4_output_dir+fname))
           #      writer2.append_data(imageio.imread(dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+fname))
    
                 i+=1
            
          #  writer=p_map(_mp4_write,filenames)
            
            writer.close()
           # writer2.close()
            print("\n")
            print(mp4_output_dir+mp4_output_filename+" completed")
       # print(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4']+" completed")
            self._clear_old_plots(mp4_input_dir)
        print("\n")
        self._copy_mp4s_over(mp4_input_dir,plot_output_dir)   

        return
  
    
  


    def _copy_mp4s_over(self,plot_dump_dir,plot_output_dir):      
        files = glob.glob(plot_dump_dir+"*.mp4")  #dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+'*.png')
        for item in files:
             filename = os.path.basename(item)
             shutil.copy(item, os.path.join(plot_output_dir, filename))
           
     
    
    def generate_annual_dates(self,df,*,start_offset,size):
        first_date=df.index[-1]+pd.offsets.Day(size+start_offset)
        last_date=df.index[0]  #first_date+pd.offsets.Day(365)
    
       # start_date=pd.to_datetime(start_date)
       # end_date=pd.to_datetime(end_date)
        
        dr=[d for d in pd.date_range(first_date,last_date)]
        for date in dr:
            yield date+pd.offsets.Day(-size),date
 



    
    def _plot_bi(self,new_pdf,key,y_col,col_and_hue,savename,plot_dump_dir,plot_output_dir,mp4_fps):
        brand_index_slices=[]
        for start,end in self.generate_annual_dates(new_pdf,start_offset=365,size=365):  #query_dict['080 SA all']):  #"2020-01-01","2020-02-02"):
            print("brand index:",key,":",start.strftime("%d/%m/%Y"),"to",end.strftime("%d/%m/%Y"),end="\r",flush=True)
            plot_pdf=new_pdf[(new_pdf.index>start) & (new_pdf.index<=end)].copy()
            brand_index_slices.append({"start":start,"end":end,"y_col":y_col,"col_and_hue":col_and_hue,"plot_dump_dir":plot_dump_dir,"plot_output_dir":plot_output_dir,"key":key,"savename":savename+" ["+start.strftime("%Y-%m-%d")+"-"+end.strftime("%Y-%m-%d")+"]","df":plot_pdf})  
       # print("\nbrand index slices=",brand_index_slices,len(brand_index_slices)) 
 
        p_map(self._plot_brand_index,brand_index_slices)
        self._animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_brand_index.mp4",plot_output_dir=plot_output_dir)
        return
   


         
   
    def animate_brand_index(self,plot_dump_dir,plot_output_dir,*,mp4_fps):
        pdf=pd.read_pickle(dd2.dash2_dict['scan']['save_dir']+dd2.dash2_dict['scan']['brand_index_jam_save_input_file'])

        pdf.sort_index(ascending=False,axis=0,inplace=True)
        new_pdf=pdf.copy()
    #    print("new_pdf=\n",new_pdf)
    #    print("scan sales brand index jam animations")
        key="brand_index_coles_jam"
        print("\n",key)
        y_col='Coles Beerenberg all jams-Units Sold off Promotion >= 5 % 6 wks'
        col_and_hue=['Coles Bonne Maman all jams-Wks on Promotion >= 5 % 6 wks','Coles St Dalfour all jams-Wks on Promotion >= 5 % 6 wks']
        savename="Brand index jams coles1"
#
        self._plot_bi(new_pdf,key,y_col,col_and_hue,savename,plot_dump_dir,plot_output_dir,mp4_fps)   #,dd2.dash2_dict['plots']['animation_resize'])
        
        new_pdf=pdf.copy()
        key="brand_index_woolworths_jam"
        print("\n",key)
        y_col='Woolworths Beerenberg all jams-Units Sold off Promotion >= 5 % 6 wks'
        col_and_hue=['Woolworths Bonne Maman all jams-Wks on Promotion >= 5 % 6 wks','Woolworths St Dalfour all jams-Wks on Promotion >= 5 % 6 wks']
        savename="Brand index jams woolworths1"
        
        self._plot_bi(new_pdf,key,y_col,col_and_hue,savename,plot_dump_dir,plot_output_dir,mp4_fps)   #,dd2.dash2_dict['plots']['animation_resize'])
 
    
 #----------------------------------------------------------------------------
 
    
 
    
        pdf=pd.read_pickle(dd2.dash2_dict['scan']['save_dir']+dd2.dash2_dict['scan']['brand_index_chutney_save_input_file'])

        pdf.sort_index(ascending=False,axis=0,inplace=True)
        

        new_pdf=pdf.copy()
        key="brand_index_coles_chutney"
        print("\n",key)
        y_col='Coles Beerenberg Tomato chutney 260g-Units Sold off Promotion >= 5 % 6 wks'
        col_and_hue=['Coles Jills Tomato chutney 400g-Wks on Promotion >= 5 % 6 wks','Coles Baxters Tomato chutney 225g-Wks on Promotion >= 5 % 6 wks']
        savename="Brand index Tomato chutney coles1"
    
 
        self._plot_bi(new_pdf,key,y_col,col_and_hue,savename,plot_dump_dir,plot_output_dir,mp4_fps)   #)dd2.dash2_dict['plots']['animation_resize'])
  
 
        new_pdf=pdf.copy()
        key="brand_index_woolworths_chutney"
        print("\n",key)
        y_col='Woolworths Beerenberg Tomato chutney 260g-Units Sold off Promotion >= 5 % 6 wks'
        col_and_hue=['Woolworths Whitlock Tomato chutney 275g-Wks on Promotion >= 5 % 6 wks','Woolworths Baxters Tomato chutney 225g-Wks on Promotion >= 5 % 6 wks']
        savename="Brand index Tomato chutney woolworths1"
 
        self._plot_bi(new_pdf,key,y_col,col_and_hue,savename,plot_dump_dir,plot_output_dir,mp4_fps)   #,dd2.dash2_dict['plots']['animation_resize'])

 
    
           
 
        print("brand index finished")
        return


    def plot_and_animate_query_dict(self,query_dict,plot_dump_dir,plot_output_dir,mp4_fps):
       
     #   self.animate_brand_index(plot_dump_dir,plot_output_dir,mp4_fps)

        for key in query_dict.keys():

            pareto_df=query_dict[key]
            df=pareto_df.rename(mapper={"date":"xdate"},axis=1)
            mat_df=df.resample('D',label='left').sum().round(0).copy()

            mat_df['salesval_mat']=mat_df['salesval'].rolling("365D",closed='left').sum()   # 365D rolling window  apply(get_rolling_amount,'365D')   #.to_frame(name='mat')    #.rolling("365D", min_periods=1).sum()
            mat_df['qty_mat']=mat_df['qty'].rolling("365D",closed='left').sum()   #   apply(get_rolling_amount,'365D')   #.to_frame(name='mat')    #.rolling("365D", min_periods=1).sum()

            slices=[]
            pareto_slices=[]
       #     if mat_df.shape[0]>0:
            print("animate",key,"df ",df.shape,"mat_df",mat_df.shape,"pareto df ",pareto_df.shape)
            for start,end in self.generate_annual_dates(query_dict[key],start_offset=365,size=365):  #query_dict['080 SA all']):  #"2020-01-01","2020-02-02"):
                    print("mats",start.strftime("%d/%m/%Y"),"to",end.strftime("%d/%m/%Y"),end="\r",flush=True)
                    plot_df=mat_df[(mat_df.index>start) & (mat_df.index<=end)].copy()
                    pareto2_df=pareto_df[(pareto_df.index>start) & (pareto_df.index<=end)].copy()
       
                    slices.append({"start":start,"end":end,"plot_dump_dir":plot_dump_dir,"plot_output_dir":plot_output_dir,"key":key,"name":key+" ["+start.strftime("%Y-%m-%d")+"-"+end.strftime("%Y-%m-%d")+"]","sliced_df":plot_df})  
                    pareto_slices.append({"start":start,"end":end,"plot_dump_dir":plot_dump_dir,"plot_output_dir":plot_output_dir,"key":key,"name":key+" ["+start.strftime("%Y-%m-%d")+"-"+end.strftime("%Y-%m-%d")+"]","df":pareto2_df})  

          
            print("\n\nquery=",key)    
            print("plot salesval mats")        
            p_map(self._p_plot_salesval,slices)  
            self._animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_mat_salesval.mp4",plot_output_dir=plot_output_dir)
     
            print("plot qty mats")
            p_map(self._p_plot_qty,slices)   
            self._animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_mat_qty.mp4",plot_output_dir=plot_output_dir)
       
            print("pareto product units")    
            p_map(self._p_pareto_product_units,pareto_slices)
            self._animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_pareto_product_units.mp4",plot_output_dir=plot_output_dir)
         
            print("pareto product dollars")    
            p_map(self._p_pareto_product_dollars,pareto_slices)
            self._animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_pareto_product_dollars.mp4",plot_output_dir=plot_output_dir)
          
            print("pareto customer dollars")    
            p_map(self._p_pareto_customer,pareto_slices)
            self._animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_pareto_customer_dollars.mp4",plot_output_dir=plot_output_dir)
              
     
    
 
        
        
        
        
        
        return

    
 