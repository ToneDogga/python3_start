#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 14:04:16 2020

@author: tonedogga
"""
import os
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime,timedelta
   
os.chdir("/home/tonedogga/Documents/python_dev")
class scheduler(object):
    def stacker(self,*,schedule,min_day_size,max_day_size):
        
        
        
        
        schedule.set_index(['priority'],drop=False,inplace=True)
        start_date=schedule.iloc[0,1]
        schedule['original_stacked_date_pos']=schedule['stacked_date_pos']

       # schedule.set_index(['original_stacked_date_pos'],drop=False,inplace=True)
  
   #     print(schedule.to_string())
      #  schedule=schedule[['scheduled_date','product','batches','new_stock_will_be_approx',"stacked_date"]]
    # take a schedule of the schedule and fit it efficiency into each day using dd2.dash2_dict['scheduler']['stacker_productivity']  values
    
     #   schedule['new_dates']=schedule.index.get_level_values(0)
     #   print("stacker=\n",schedule)
        # for r1 in schedule.index:
        #     start_batch=schedule.iloc[r1,'batches']
        #     print(schedule)
        #     for r2 in schedule.index[1:]:
        #         tot_batches=schedule.loc[r2,'batches']+start_batch
        #         print("r2",r2,tot_batches)
        #schedule.to_pickle("fs.pkl",protocol=-1)   
        
  #      schedule.reset_index(drop=True,inplace=True)
   #     schedule.sort_values(["stacked_date_pos"],ascending=True,inplace=True)

        schedule['moved']=False
        schedule=schedule[~schedule['moved']]
        for passes in range(1,2):
            stacked_pos=0
            schedule['moved']=False
            inc_sp_flag=False
         #   schedule=schedule[~schedule['moved']]
            for row in schedule.index:  #range(0,schedule.shape[0]-1):
                if ~schedule.loc[row,'moved']:
                  #  try:
  
                    tot_batch=schedule.loc[row,'batches']
                  #  except:
                  #      break
                   # new_schedule=schedule[~schedule['moved']]
                   # ft=schedule.loc[row,'format_type']
                 #   schedule=schedule[schedule['format_type']==schedule.loc[row,'format_type']]
                    for row2 in schedule.index[row:]:
                        if (~(schedule.loc[row2,'moved']) | (schedule.loc[row,'moved'])) & (schedule.loc[row,'format_type']==schedule.loc[row2,'format_type']):
                         #   schedule=schedule[(schedule[row2,'format_type']==ft)]
                       #     if (schedule.loc[row,'format_type']==schedule.loc[row2,'format_type']):
                                 test_tot_batch=tot_batch+schedule.loc[row2,'batches']
                              #   print("pass=",passes,"r1=",row,tot_batch,"+r2=",row2,schedule.loc[row2,'batches'],"=",test_tot_batch)
                                 if (test_tot_batch<min_day_size):
                               #      print("too small <",min_day_size,test_tot_batch)
                          #           schedule.loc[row,'stacked_date_pos']=stacked_pos
                                    # try:
                                     tot_batch+=schedule.loc[row2,'batches']
                                    # except:
                                    #    break
                                   #  stacked_pos+=1
                                 elif (test_tot_batch>=max_day_size):
                              #       print("too big >=",max_day_size,test_tot_batch)
                                     inc_sp_flag=True
                                    # break
                                    # tot_batch=0
                                   #  stacked_pos+=1
                                     
                                 else:  
                                      #   tot_batch+=schedule.loc[row2,'batches']
                             #        print("just right test_tot_batch=",test_tot_batch)
                                     tot_batch=test_tot_batch
                                   #  print("r1=",schedule.loc[row,'stacked_date_pos'])  #=stacked_pos
                                   #  print("r2=",schedule.loc[row2,'stacked_date_pos']) 
                                     #print("sp=",stacked_pos)
                                   #  if (schedule.loc[row,'format_type']==schedule.loc[row2,'format_type']):
#                                         schedule.loc[row2,'stacked_date_pos']=schedule.loc[row,'stacked_date_pos']  #=stacked_pos
                                   #      if ~schedule.loc[row2,'moved']:    
                                     schedule.loc[row2,'stacked_date_pos']=stacked_pos
                                     inc_sp_flag=True
                                   #  schedule.loc[row,'stacked_date_pos']=stacked_pos

                                     schedule.loc[row2,'moved']=True
                               #      schedule.loc[row,'moved']=True
                                       
                                         #schedule=schedule[~schedule['moved']]
                                    # stacked_pos+=1
                                       #  print("format type match")
                            #         print(schedule.to_string())
                                     
                        else:
                            pass
                       #    print("skipped r1=",row,"r2=",row2,"sp",stacked_pos)
                            #schedule.loc[row2,'stacked_date_pos']=stacked_pos
                            

                    
                    if inc_sp_flag: 
                        stacked_pos+=1
                        inc_sp_flag=False
                                    # else:
                                    #     print("format types dont match")
    
    
        
       
         
    
    #    print("final schedule=\n",schedule.to_string())                             
            
    #    final_schedule=schedule[['priority','scheduled_date','product','batches','new_stock_will_be_approx']]
     #   final_schedule.drop(["scheduled_date","priority"],axis=1,inplace=True)
     #   final_schedule['new_date']=final_schedule['scheduled_date'].iloc[0]
        schedule.reset_index(drop=True,inplace=True)
  
     #   schedule.sort_values(["stacked_date_pos"],ascending=True,inplace=True)
        for p in range(1,2):
            schedule.sort_values(["stacked_date_pos"],ascending=True,inplace=True)
          #  schedule=schedule[(~schedule['moved']) & (schedule['stacked_date_pos']>=stacked_pos)]
         #   print("schedule slice=",schedule.to_string())
            try:
                topval=schedule[(~schedule['moved']) & (schedule['stacked_date_pos']>=stacked_pos)].iloc[:,6].to_numpy()
            except:
                pass
            else:
            #    print("p,tv",topval)
 
                change=(-(np.arange(stacked_pos,stacked_pos+topval.shape[0])-topval))
             #   print("stacked pos",stacked_pos,topval,change)
              #  print(schedule[((~schedule['moved']) & (schedule['stacked_date_pos']>=stacked_pos))])  #.iloc[:,6]-=change
                schedule.loc[((~schedule['moved']) & (schedule['stacked_date_pos']>=stacked_pos)),'stacked_date_pos']-=change
  
             #   schedule.loc[(~schedule['moved']) & (schedule['stacked_date_pos']>=stacked_pos),'stacked_date_pos']-=change
             #   schedule.loc[(~schedule['moved']) & (schedule['stacked_date_pos']>=stacked_pos),'moved']=True
        schedule.drop(['original_stacked_date_pos','moved','format_type'],inplace=True,axis=1)   
        
    #    start_date = dt.date.today()   #time.now()  #.normalize()   #dt.date.today()
     #   other_start_date=start_date.dt.datetime
        date_choices=pd.bdate_range(start_date,start_date+timedelta(schedule.shape[0]),normalize=True)
        date_choices_dict={}
        
        for j,d in enumerate(date_choices):
            i=int((d-start_date).days)
            date_choices_dict[j]=d

      #  print(date_choices_dict)
        schedule.drop(['scheduled_date'],axis=1,inplace=True)
        schedule['scheduled_date']=[date_choices_dict[i] for i in schedule['stacked_date_pos'].to_list()]
        schedule.drop(['stacked_date_pos'],axis=1,inplace=True)
        schedule.set_index(['scheduled_date'],drop=True,inplace=True)
      #  schedule.sort_index(ascending=True,axis=1,inplace=True)
        return schedule

s=scheduler()  
schedule=pd.read_pickle("fs.pkl")
#print("start schedule=\n",schedule.to_string())

min_day_size=50
max_day_size=70
  

schedule=s.stacker(schedule=schedule,min_day_size=min_day_size,max_day_size=max_day_size)
print("finish schedule=\n",schedule.to_string())