#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:25:54 2020

@author: tonedogga
"""

# import pandas as pd
# from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# festivities = ['2019-02-28','2019-05-31'] #both on Friday
# my_bank_holidays = pd.tseries.offsets.CustomBusinessMonthEnd(holidays=festivities)
# s = pd.date_range('2019-01-01', periods=12, freq=my_bank_holidays)
# df = pd.DataFrame(s, columns=['Date'])
# df['n_days'] = df['Date'].diff().dt.days.fillna(0)
# print(df)

# dr = pd.date_range(start='2015-07-01', end='2015-07-31')
# df = pd.DataFrame()
# df['Date'] = dr

# cal = calendar()
# holidays = cal.holidays(start=dr.min(), end=dr.max())

# df['Holiday'] = df['Date'].isin(holidays)
# print(df)

import pandas as pd
from pandas.tseries.holiday import *
from pandas.tseries.offsets import CustomBusinessDay


class SABusinessCalendar(AbstractHolidayCalendar):
   rules = [
     Holiday('New Year', month=1, day=1), #observance=sunday_to_monday),
     Holiday('Australia Day', month=1, day=26, observance=sunday_to_monday),
   #  Holiday('St. Patricks Day', month=3, day=17, observance=sunday_to_monday),
     Holiday('Anzac Day', month=4, day=25),
     Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
     Holiday('Easter Monday', month=1, day=1, offset=[Easter(),Day(+1)]),    
   #  Holiday('Canada Day', month=7, day=1, observance=sunday_to_monday),
   #  Holiday('July 4th', month=7, day=4, observance=nearest_workday),
     Holiday('October long weekend', month=10, day=1, observance=sunday_to_monday),
     Holiday('Christmas', month=12, day=25, observance=nearest_workday),
     Holiday('Boxing day', month=12, day=26, observance=nearest_workday),
     Holiday('Proclamation day', month=12, day=28, observance=nearest_workday)
   ]

SA_BD = CustomBusinessDay(calendar=SABusinessCalendar())
s = pd.bdate_range('2020-07-01', end='2021-01-07', freq=SA_BD)
df = pd.DataFrame(s, columns=['Date'])
print(df)



# class GothamBusinessCalendar(AbstractHolidayCalendar):
#    rules = [
#      Holiday('New Year', month=1, day=1, observance=sunday_to_monday),
#      Holiday('Groundhog Day', month=1, day=6, observance=sunday_to_monday),
#      Holiday('St. Patricks Day', month=3, day=17, observance=sunday_to_monday),
#      Holiday('April Fools Day', month=4, day=1),
#      Holiday('Good Friday', month=1, day=1, offset=[Easter(), Day(-2)]),
#      Holiday('Labor Day', month=5, day=1, observance=sunday_to_monday),
#      Holiday('Canada Day', month=7, day=1, observance=sunday_to_monday),
#      Holiday('July 4th', month=7, day=4, observance=nearest_workday),
#      Holiday('All Saints Day', month=11, day=1, observance=sunday_to_monday),
#      Holiday('Christmas', month=12, day=25, observance=nearest_workday)
#    ]

# Gotham_BD = CustomBusinessDay(calendar=GothamBusinessCalendar())
# s = pd.date_range('2018-07-01', end='2018-07-31', freq=Gotham_BD)
# df = pd.DataFrame(s, columns=['Date'])
# print(df)


# from datetime import date 
# import holidays 
  
# # Select country 
# uk_holidays = holidays.Australia()   #UnitedKingdom() 
  
# # Print all the holidays in UnitedKingdom in year 2018 
# for ptr in holidays.UnitedKingdom(years = 2018).items(): 
#     print(ptr) 
