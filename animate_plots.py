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

 
import datetime as dt
from datetime import datetime,timedelta
#import dash2_root
import dash2_dict as dd2

#dash=dash2_root.dash2_class()






class animate_engine(object):  
    def _resize(self,filename):
        fname = filename   #os.path.basename(filename)
        input_image  = imageio.imread(fname, format='PNG-FI')
        output_image = cv2.resize(input_image, (944,800))   #(1888, 1600))   #(256, 256))
        imageio.imwrite(fname, output_image, format='PNG-FI')   # 48 bit
        return

    
    
    
    # def log_dir(self,prefix=""):
    #     now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    #     root_logdir = "./dash2_outputs"
    #     if prefix:
    #         prefix += "-"
    #     name = prefix + "run-" + now
    #     return "{}/{}/".format(root_logdir, name)
    

    
    # def animate_plots_gif(self,*,gif_input_dir,gif_duration,gif_output_dir):
    #     # turn the plots into a animated gif, mp4
    #     # turn to recommended ones into a gif
     
    #     filenames=sorted(glob.glob(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+'*.png'))   #, key=os.path.getctime)
    #     #sorted(glob.glob('*.png'), key=os.path.getmtime)
    #     #print("f=",filenames)
    #     print("\n")
    #     print("Creating gif...")
    #     images = []
    #   #  i=1
    #     lf=len(filenames)
    #   #  for filename in filenames:
    #   #      print("Creating gif of",i,"/",lf,"plots....",end="\r",flush=True)
    #   #      images.append(imageio.imread(filename))        
    #   #      i+=1
    #     p_map(self._resize,filenames)
    #     images=p_map(imageio.imread,filenames)    
            
    #     imageio.mimsave(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'], images,duration=gif_duration)
    #     print(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'],"completed\n")
    #     imageio.mimsave(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'], images,duration=gif_duration)
    #     print(plot_output_dir+dd2.dash2_dict['scheduler']['schedule_savefile_plots_gif'],"completed\n")
    #     return
        
  
    
    def animate_plots_mp4(self,*,mp4_input_dir,mp4_fps,mp4_output_dir,mp4_output_filename):

        #---------------------------------------------------
    # only need once
       # imageio.plugins.freeimage.download()
    #---------------------------------------------------------------------

        filenames=sorted(glob.glob(mp4_input_dir+'*.png'))   #, key=os.path.getctime)
        lf=len(filenames)
     #   print(filenames)
       # print("Creating",dd2.dash2_dict['scheduler']['schedule_savedir_plots']+dd2.dash2_dict['scheduler']['schedule_savefile_plots_mp4'])
        # i=0
        # for filename in filenames:
        #     print("Resizing plots of",i,"/",lf,"plots....",end="\r",flush=True)
        #     fname = os.path.basename(filename)
        #     input_image  = imageio.imread(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+fname, format='PNG-FI')
        #     output_image = cv2.resize(input_image, (1888, 1600))   #(256, 256))
        #     imageio.imwrite(dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+fname, output_image, format='PNG-FI')   # 48 bit
        #     i+=1
            
        p_map(self._resize,filenames)
        
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
        return
    
 #+-----------------------------------------------------------------------------------------------------------
             
    def _clear_old_plots(self,clear_dir):   
        # clear old plots
        files = glob.glob(clear_dir+"*.png")  #dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+'*.png')
      #  print("1files to del:",files)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("file delete Error: %s : %s" % (f, e.strerror))
                
                
      #      # clear old plots
      #   files = glob.glob(dd2.dash2_dict['scheduler']['schedule_savedir_plots']+'*.png')
      # #  print("2files to del:",files)
      #   for f in files:
      #       try:
      #           os.remove(f)
      #       except OSError as e:
      #           print("file delete Error: %s : %s" % (f, e.strerror))
        
                
      #+-----------------------------------------------------------------------------------------------------------

             
    
    def _save_fig(self,fig_id, output_dir,tight_layout=True, fig_extension="png", resolution=300):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, fig_id + "." + fig_extension)
      #  print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution,bbox_inches='tight')
        return
    
    
    def preprocess(self,df,mat):
        # rename qolumns
       
       # df.rename(columns=rename_dict, inplace=True)
    
     #   df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
      #  df['date']=df.index
      #  df['mat']=df['salesval'].rolling(mat,axis=0).sum()
    #    print("rename collumsn")
        
       # df.rename(columns=rename_dict, inplace=True)
    
   #     df=df.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0)
      #  df['date']=df.index
        df['mat']=df['salesval'].rolling(window=mat,axis=0).sum()
      #  df=df[(df['mat']>=0)]
    #    print("rename collumsn")
        return df
    
    
    
    def _clean_up_name(self,name):
        name = name.replace('.', '_')
        name = name.replace('/', '_')
        name = name.replace(',', '_')
        name = name.replace(' ', '_')
        return name.replace("'", "")
    



    def pareto_customer(self,slices):
        df=slices["df"]    
        df_dict={slices['name']:slices["df"][(slices["df"].index>=slices['start']) & (slices["df"].index<slices['end'])].copy()}
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]

#        print("pareto customer plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
   #     i_dict=df_dict.copy()
        for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
            new_df=v.groupby(['code'],sort=False).sum()
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


    def pareto_product_dollars(self,slices):
        df=slices["df"]    
        df_dict={slices['name']:slices["df"][(slices["df"].index>=slices['start']) & (slices["df"].index<slices['end'])].copy()}
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]

 #       print("pareto product plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
     #   i_dict=df_dict.copy()
        for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
            new_df=v.groupby(['product'],sort=False).sum()
 
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




    def pareto_product_units(self,slices):
        df=slices["df"]    
        df_dict={slices['name']:slices["df"][(slices["df"].index>=slices['start']) & (slices["df"].index<slices['end'])].copy()}
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]

     #   print("pareto product units plot type=",df_dict.keys(),latest_date.strftime('%d/%m/%Y'),output_dir)
        top=60
   #     i_dict=df_dict.copy()
      #  print("pareto product i_dict=\n",i_dict,"\n i_dict.items()=\n",i_dict.items())
        for k,v in df_dict.items():
        #    cust_df=self.preprocess(df_dict[k],mat).copy()
  #          new_df=df_dict[k].groupby(['code','product'],sort=False).sum()
            new_df=v.groupby(['product'],sort=False).sum()
 
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

 


    def mat(self,slices):
        mat_df=slices["df"]    
        df_dict={slices['name']:slices["df"][(slices["df"].index>=slices['start']) & (slices["df"].index<slices['end'])].copy()}
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]
        mat= 28 #dd2.dash2_dict['sales']['plots']['mat']
        
        
    #    print("plotting mat plot type=",df_dict.keys(),mat,latest_date.strftime('%d/%m/%Y'),output_dir)
        for k,v in df_dict.items():
           # mat_df=v.copy()
            mat_df=v.resample('W-WED', label='left', loffset=pd.DateOffset(days=-3)).sum().round(0).copy()
            
 #           loffset = '7D'
 #           weekly_sdf=sdf.resample('W-TUE', label='left').sum().round(0)   
 #           weekly_sdf.index = weekly_sdf.index + to_offset(loffset) 
 
            
            if True:   # mat_df.shape[0]>mat:
             #   mat_df=self.preprocess(mat_df,mat)
                mat_df['mat']=mat_df['salesval'].rolling(mat,axis=0).sum()
          #      df=df[(df['mat']>=0)]
     
           #     print("end mat preprocess=\n",df)
               # styles1 = ['b-','g:','r:']
                styles1 = ['b-']
              # styles1 = ['bs-','ro:','y^-']
                linewidths = 2  # [2, 1, 4]
                       
                fig, ax = pyplot.subplots()
                ax=mat_df.iloc[mat:][['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
                #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
             
                ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places
    
                ax.set_title("["+self._clean_up_name(str(k))+"] $ sales moving total "+str(mat)+" weeks @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
                ax.legend(title="",fontsize=6,loc="upper left")
                ax.set_xlabel("",fontsize=6)
                ax.set_ylabel("",fontsize=6)
               # ax.yaxis.set_major_formatter('${x:1.0f}')
                ax.yaxis.set_tick_params(which='major', labelcolor='green',
                             labelleft=True, labelright=False)
                
                self._save_fig(self._clean_up_name(str(k))+"_dollars_moving_total",output_dir)
                plt.close()+pd.offsets.Day(-365)
        return   



    def mat_salesval(self,slices):
        t_df=slices["df"]    
        k=slices['name']
        start=slices['start']   #+pd.offsets.Day(365)
      #  start_2yrs=slices['start']
        end=slices['end']
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]
     #   print("slices=",slices)
     #   print("se",start,end)
        plot_vals=[]
        for d in pd.date_range(start,end):
            v=t_df[(t_df.index>=(d+pd.offsets.Day(-365))) & (t_df.index<d)]
    
          #  print("v=\n",v) 
         #   mat= 28 #dd2.dash2_dict['sales']['plots']['mat']
            
            
                   #   mat_df=self.preprocess(mat_df,mat)
            mat_df=v.resample('D',label='left').sum().round(0).copy()
            
         #   print("s",s)   #,s.iloc[-1])
            plot_vals.append((d,mat_df['salesval'].sum()))
      #  print("plotvals=",plot_vals)
            
    
        plot_df = pd.DataFrame(plot_vals, columns =['date', 'salesval'])
        plot_df.set_index('date',inplace=True)
      #  print("plotdf=\n",plot_df)
       # mat_df=t_df[(u_df.index>=start) & (u_df.index<end)].copy()
        #if mat_df.shape[0]>367:
         #   mat_df['mat']=mat_df['salesval'].rolling(365,axis=0).sum()
         #   print("resampled mat_df=\n",mat_df)
 
              #      df=df[(df['mat']>=0)]
         
        #     print("end mat preprocess=\n",df)
            # styles1 = ['b-','g:','r:']
        styles1 = ['b-']
        # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
                 
        fig, ax = pyplot.subplots()
        ax=plot_df.plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       
        ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}')) # 2 decimal places
 
        ax.set_title("["+self._clean_up_name(str(k))+"] $ sales moving total2 365 days @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
        ax.legend(title="",fontsize=6,loc="upper left")
        ax.set_xlabel("",fontsize=6)
        ax.set_ylabel("",fontsize=6)
         # ax.yaxis.set_major_formatter('${x:1.0f}')
        ax.yaxis.set_tick_params(which='major', labelcolor='green',
                       labelleft=True, labelright=False)
          
        self._save_fig(self._clean_up_name(str(k))+"_dollars_moving_total2",output_dir)
        plt.close()
        return   





    def mat_qty(self,slices):
        t_df=slices["df"]    
        k=slices['name']
        start=slices['start']   #+pd.offsets.Day(365)
      #  start_2yrs=     
      
        end=slices['end']
        latest_date=slices['end']
        output_dir=slices["plot_dump_dir"]
     #   print("slices=",slices)
     #   print("se",start,end)
        plot_vals=[]
        for d in pd.date_range(start,end):
            v=t_df[(t_df.index>=(d+pd.offsets.Day(-365))) & (t_df.index<d)]
    
          #  print("v=\n",v) 
         #   mat= 28 #dd2.dash2_dict['sales']['plots']['mat']
            
            
                   #   mat_df=self.preprocess(mat_df,mat)
            mat_df=v.resample('D',label='left').sum().round(0).copy()
            
         #   print("s",s)   #,s.iloc[-1])
            plot_vals.append((d,mat_df['qty'].sum()))
      #  print("plotvals=",plot_vals)
            
    
        plot_df = pd.DataFrame(plot_vals, columns =['date', 'qty'])
        plot_df.set_index('date',inplace=True)
      #  print("plotdf=\n",plot_df)
       # mat_df=t_df[(u_df.index>=start) & (u_df.index<end)].copy()
        #if mat_df.shape[0]>367:
         #   mat_df['mat']=mat_df['salesval'].rolling(365,axis=0).sum()
         #   print("resampled mat_df=\n",mat_df)
 
              #      df=df[(df['mat']>=0)]
         
        #     print("end mat preprocess=\n",df)
            # styles1 = ['b-','g:','r:']
        styles1 = ['b-']
        # styles1 = ['bs-','ro:','y^-']
        linewidths = 1  # [2, 1, 4]
                 
        fig, ax = pyplot.subplots()
        ax=plot_df.plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
          #ax=mat_df[['mat']].plot(grid=True,fontsize=6,style=styles1, lw=linewidths)
       
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # 2 decimal places
 
        ax.set_title("["+self._clean_up_name(str(k))+"] units moving total 365 days @w/c:"+latest_date.strftime('%d/%m/%Y'),fontsize= 7)
        ax.legend(title="",fontsize=6,loc="upper left")
        ax.set_xlabel("",fontsize=6)
        ax.set_ylabel("",fontsize=6)
         # ax.yaxis.set_major_formatter('${x:1.0f}')
        ax.yaxis.set_tick_params(which='major', labelcolor='green',
                       labelleft=True, labelright=False)
          
        self._save_fig(self._clean_up_name(str(k))+"_units_moving_total",output_dir)
        plt.close()
        return   






#######################################################################################

       
    
    
    def generate_annual_dates(self,df,*,start_offset,size):
        first_date=df.index[-1]+pd.offsets.Day(size+start_offset)
        last_date=df.index[0]  #first_date+pd.offsets.Day(365)
    
       # start_date=pd.to_datetime(start_date)
       # end_date=pd.to_datetime(end_date)
        
        dr=[d for d in pd.date_range(first_date,last_date)]
        for date in dr:
            yield date+pd.offsets.Day(-size),date
            
            
  
            
    def animate_reports(self,query_dict,plot_output_dir,*,mp4_fps):    
       #
        
      #  plot_output_dir = log_dir("dash2")
      #  dd2.dash2_dict['sales']['plots']['plot_output_dir']=plot_output_dir
        plot_dump_dir=plot_output_dir+dd2.dash2_dict['sales']['plots']["animation_plot_dump_dir"]
        #print(plot_dump_dir)
        #os.mkdir(plot_dump_dir,)
        
      #  os.chdir("/home/tonedogga/Documents/python_dev")
     #   with open(dd2.dash2_dict['sales']["save_dir"]+dd2.dash2_dict['sales']["query_dict_savefile"], 'rb') as f:
     #       query_dict = pickle.load(f)
        
        #print(query_dict)
        for key in query_dict.keys():
            #key="080 SA all"
            df=query_dict[key]
        
        
            print("plotting mats",key)
            slices=[]
            #g=[d for d in pd.date_range("2019-01-01",latest_date)]
            if df.shape[0]>0:
                for start,end in self.generate_annual_dates(df,start_offset=365,size=365):  #query_dict['080 SA all']):  #"2020-01-01","2020-02-02"):
                #   print("mats",start,"to",end)
                   slices.append({"start":start,"end":end,"plot_dump_dir":plot_dump_dir,"name":key+" ["+start.strftime("%Y-%m-%d")+"-"+end.strftime("%Y-%m-%d")+"]","df":df})  
            
       
                print(key+"    mat salesval")    
                p_map(self.mat_salesval,slices)
                self.animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_mat_salesval.mp4")
  
       
                print(key+"    mat qty")    
                p_map(self.mat_qty,slices)
                self.animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_mat_qty.mp4")
    
  
            
                print("plotting paretos",key)
                slices=[]
                #g=[d for d in pd.date_range("2019-01-01",latest_date)]
                for start,end in self.generate_annual_dates(df,start_offset=0,size=365):  #query_dict['080 SA all']):  #"2020-01-01","2020-02-02"):
                 #  print(start,"to",end)
                   slices.append({"start":start,"end":end,"plot_dump_dir":plot_dump_dir,"name":key+" ["+start.strftime("%Y-%m-%d")+"-"+end.strftime("%Y-%m-%d")+"]","df":df})  
            
       
              #  print(key+"    mat2")    
              #  p_map(self.mat2,slices)
              #  self.animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_mat2.mp4")
         
                # print(key+"    mat")    
                # p_map(self.mat,slices)
                # self.animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_mat.mp4")
         
      
                print(key+"    product units")    
                p_map(self.pareto_product_units,slices)
                self.animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_pareto_product_units.mp4")
                
                print(key+"    product dollars")    
                p_map(self.pareto_product_dollars,slices)
                self.animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_pareto_product_dollars.mp4")
                
                print(key+"    customer dollars")    
                p_map(self.pareto_customer,slices)
                self.animate_plots_mp4(mp4_input_dir=plot_dump_dir,mp4_fps=mp4_fps,mp4_output_dir=plot_dump_dir,mp4_output_filename=key+"_pareto_customer_dollars.mp4")
            
        
        
        
        ###############################################################33
        # copy *.mp4 files across to main output dir
        
            files = glob.glob(plot_dump_dir+"*.mp4")  #dd2.dash2_dict['scheduler']['schedule_savedir_resized_plots']+'*.png')
            for item in files:
                 filename = os.path.basename(item)
                 shutil.copy(item, os.path.join(plot_output_dir, filename))
           
  
  
        return