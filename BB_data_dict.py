#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:42:38 2020

@author: tonedogga
"""

from collections import namedtuple
import numpy as np



#  tf model hyper parameters for order prediction based on scan sales data
batch_length=4
no_of_batches=1000
no_of_repeats=4
epochs=6


# product appears on low stock report if units stock is below this number
low_stock_limit=8000



# filenames in same directory as this programme
stock_level_query='stock_level_query.xlsx'
production_made_query='Production Schedule.xlsx'
production_made_sheet="Schedule"
production_planned_query='B Stock & Schedule Forecast.xlsx'
production_planned_sheet="Schedule"
report_savename="sales_trans_report_dict.pkl"
colesscan="Coles_scan_data_300620.xlsx"
woolscan="Ww_scan_data_300620.xlsx"
scandatalist=["Coles_scan_data_300620.xlsx","Ww_scan_data_300620.xlsx"]
  #  report_savename="sales_trans_report_dict.pkl"
savepkl="coles_and_ww_invoiced_and_scanned_sales.pkl"


sales_df_savename="sales_trans_df.pkl"
filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx","salestrans.xlsx"]
   

product_groups_only=["10","11","12","13","14","15","16","17","18"]
spc_only=["088","028"]   #,"038","048","028","080","020","030","040"]


# moving average total days used in prediction
mats=7

scan_data_files=["jam_scan_data_2020.xlsx","cond_scan_data_2020.xlsx","sauce_scan_data_2020.xlsx"]
scan_dict_savename="scan_dict.pkl"



productgroup_dict={"01":"30g glass",
                   "02":"40g glass",
                   "041":"15g Sterling",
                   "042":"30g Sterling",
                   "043":"50g Sterling",
                   "05":"14g SIS",
                   "10":"Jams 250ml glass jar",
                   "11":"Sauce 300ml glass bottle",
                   "12":"Dressings 300ml glass bottle",
                   "13":"Condiments 250ml glass jar",
                   "14":"Meal bases 250ml glass jar",
                   "15":"Condiments for cheese 150ml glass jar",
                   "16":"Traditional condiments 150ml glass jar",
                   "17":"Mustards 150ml glass jar",
                   "18":"Olive Oil 500ml",
                   "31":"2 Litre"}

productgroups_dict={1:"30g glass",
                   2:"40g glass",
                   41:"15g Sterling",
                   42:"30g Sterling",
                   43:"50g Sterling",
                   5:"14g SIS",
                   10:"Jams 250ml glass jar",
                   11:"Sauce 300ml glass bottle",
                   12:"Dressings 300ml glass bottle",
                   13:"Condiments 250ml glass jar",
                   14:"Meal bases 250ml glass jar",
                   15:"Condiments for cheese 150ml glass jar",
                   16:"Traditional condiments 150ml glass jar",
                   17:"Mustards 150ml glass jar",
                   18:"Olive Oil 500ml",
                   31:"2 Litre"}



salesrep_dict={"14":"Brendan Selby",
               "8":"Noel Lotorto",
               "33":"Howard Lee",
               "34":"Russell Heyzer",
               "35":"Stephen Wood",
               "36":"Annette Paech",
               "37":"Miles Rafferty",
               "39":"Sophia Simos",
               "11":"Coles and WW"
               }


spc_dict={122:"Harris farms",
          107:"Lite n easy",
          10:"woolworths",
          12:"coles",
          88:"SA stores",
          80:"SA distributors",
          20:"NSW distributors",
          40:"QLD distributors",
          30:"VIC distributors",
          38:"VIC stores",
          28:"NSW stores",
          48:"QLD stores",
          50:"WA distributors",
          70:"TAS distributors",
          92:"Shop",
          95:"Online shop"}


report = namedtuple("report", ["name", "report_type","cust","prod"])

report_type_dict={0:"dictionary",
                       3:"dataframe",
                       5:"spreadsheet",
                       6:"pivottable",
                       8:"chart_filename"}



   #report = namedtuple("report", ["name", "report_type","cust","prod"])
 
#scandata = namedtuple("scandata", ["market", "product","measure"])

market_rename_dict={"AU Woolworths scan":10,
                    "AU Coles Group scan":12}

market_rename_dict2={"AU Woolworths scan":"woolworths",
                    "AU Coles Group scan":"coles"}




# measure_dict={1:"units_sold_total",
#               2:"units_sold_off_promotion",
#               3:"units_sold_on_promotion",
#               4:"dollars_sold_total",
#               5:"dollars_sold_off_promotion",
#               6:"dollars_sold_on_promotion",
#               7:"full_price",
#               8:"promo_price",
#               9:"weighted_depth_of_dist"}


category_dict={0:"unknown",
                1:"jams",
                2:"condiments",
                3:"sauces",
                4:"dressings"}


plot_type_dict={0:"Units report",
                    1:"Dollars report",
                    2:"Dist vs ranking report",
                    3:"Dollars vs price report",
                    4:"Units/store/Week report",
                    5:"Share report",
                    }


#Used to convert the meaure column names to a measure type to match the dict above
# the first digit is the plot report to group them in, the second is whether to stack the fields when plotting
measure_conversion_dict={0:2,
                          1:5,
                          2:3,
                          3:3,
                          4:4,
                          5:4,
                          6:2,
                          7:2,
                          8:2,
                          9:5,
                          10:5,
                          11:5,
                          12:0,
                          13:0,
                          14:1,
                          15:1}




#measure list=
# ['Depth Of Distribution Wtd',   0
#'Dollars (000)',                        1
#'Dollars (000) Sold off Promotion >= 5 % 6 wks',             2
#'Dollars (000) Sold on Promotion >= 5 % 6 wks',              3
#'Price ($/Unit)',                4
#'Units (000)',                      5
#'Units (000) Ranked Position in Total Chutney Pickles Relish',         6 
# 'Units (000) Ranked Position in Total Jam/Curd/Marmalade',           7
# 'Units (000) Ranked Position in Total Sauces',                 8
#'Units (000) Share of Total Chutney Pickles Relish',            9
#'Units (000) Share of Total Jam/Curd/Marmalade',            10
# 'Units (000) Share of Total Sauces',                        11
#'Units (000) Sold off Promotion >= 5 % 6 wks',          12
#'Units (000) Sold on Promotion >= 5 % 6 wks',           13
#'Units/Store/Week - Baseline >= 5 6 wks',               14
#'Units/Store/Week - Incremental']                        15



#Stack the value when plotting?
stacked_conversion_dict={0:False,
                          1:False,
                          2:True,
                          3:True,
                          4:True,
                          5:False,
                          6:False,
                          7:False,
                          8:False,
                          9:False,
                          10:False,
                          11:False,
                          12:True,
                          13:True,
                          14:True,
                          15:True}


#Stack the value when plotting?
second_y_axis_conversion_dict={0:False,
                          1:True,
                          2:False,
                          3:False,
                          4:True,
                          5:False,
                          6:False,
                          7:False,
                          8:False,
                          9:False,
                          10:False,
                          11:False,
                          12:False,
                          13:False,
                          14:False,
                          15:False}


# y axis reversed for ranking reports
reverse_conversion_dict={0:False,
                          1:False,
                          2:False,
                          3:False,
                          4:False,
                          5:False,
                          6:True,
                          7:True,
                          8:True,
                          9:False,
                          10:False,
                          11:False,
                          12:False,
                          13:False,
                          14:False,
                          15:False}
 


variety_type_dict={0:"unknown",
                    1:"total",
                    2:"apricot",
                    3:"blackberry",
                    4:"blueberry",
                    5:"orange marm",
                    6:"blood orange marm",
                    7:"plum",
                    8:"strawberry",
                    9:"raspberry",
                    10:"other",
                    11:"fig almond",
                    12:"caramelis",
                    13:"worcestershire",
                    14:"sce",
                    15:"sce hot",
                    16:"fruit of forest",
                    17:"roadhouse",
                    18:"lime lemon",
                    19:"peri peri",
                    20:"pear",
                    21:"quince",
                    22:"mustard pickles",
                    23:"mango",
                    24:"red currant",
                    25:"rose petal",
                    26:"ajvar",
                    27:"4 fru",
                    28:"peach",
                    29:"cherry",
                    30:"fruit",
                    31:"blackcurrant",
                    32:"fig royal",
                    33:"mato&red",
                    34:"beetroot",
                    35:"chilli",
                    36:"burger",
                    37:"omoto hot",
                    38:"tomat",
                    39:"apple",
                    40:"takatala"
                    }

brand_dict={0:"total",
            1:"beerenberg",
            2:"st_dalfour",
            3:"bonne_maman",
            4:"cottees",
            5:"anathoth",
            6:"roses",
            10:"baxters",
            11:"whitlocks",
            12:"barker",
            13:"three_threes",
            14:"spring_gully",
            15:"masterfoods",
            16:"yackandandah",
            17:"goan",
            18:"podravka",
            20:"heinz",
            21:"leggos",
            22:"mrs_hs_ball",
            23:"branston",
            24:"maggie_beer",
            25:"country_cuisine",
            26:"fletchers",
            27:"jamie_oliver",
            28:"regimental",
            29:"maleny",
            7:"ixl",
            30:"red_kellys",
            100:"other"}




# value is (brand,specialpricecat, productgroup, product,type,name)
# type is 0 off promo. 1 is on promo, 2 is total of both, 3 is invoiced total

product_type = namedtuple("product_type", ["brandno","customercat", "productgroup","product","type","name"])

coles_and_ww_pkl_dict={"coles_beerenberg_jams_invoiced.pkl":(1,12,10,"jams",3,"coles_beerenberg_jams_invoiced"),   # special price cat, productgroup,productcode,product, on_promo, name)
          "coles_beerenberg_SJ300_invoiced.pkl":(1,12,10,"SJ300",3,"coles_beerenberg_SJ300_invoiced"),
          "coles_beerenberg_AJ300_invoiced.pkl":(1,12,10,"AJ300",3,"coles_beerenberg_AJ300_invoiced"),
          "coles_beerenberg_OM300_invoiced.pkl":(1,12,10,"OM300",3,"coles_beerenberg_OM300_invoiced"),
          "coles_beerenberg_RJ300_invoiced.pkl":(1,12,10,"RJ300",3,"coles_beerenberg_RJ300_invoiced"),
          "coles_beerenberg_TS300_invoiced.pkl":(1,12,11,"TS300",3,"coles_beerenberg_TS300_invoiced"),
          "coles_beerenberg_CAR280_invoiced.pkl":(1,12,13,"CAR280",3,"coles_beerenberg_CAR280_invoiced"),
          "coles_beerenberg_BBR280_invoiced.pkl":(1,12,13,"BBR280",3,"coles_beerenberg_BBR280_invoiced"),
          "coles_beerenberg_TC260_invoiced.pkl":(1,12,13,"TC260",3,"coles_beerenberg_TC260_invoiced"),
          "coles_beerenberg_HTC260_invoiced.pkl":(1,12,13,"HTC260",3,"coles_beerenberg_HTC260_invoiced"),
          "coles_beerenberg_PCD300_invoiced.pkl":(1,12,14,"PCD300",3,"coles_beerenberg_PCD300_invoiced"),
          "coles_beerenberg_BLU300_invoiced.pkl":(1,12,14,"BLU300",3,"coles_beerenberg_BLU300_invoiced"),
          "coles_beerenberg_RAN300_invoiced.pkl":(1,12,14,"RAN300",3,"coles_beerenberg_RAN300_invoiced"),
          "woolworths_beerenberg_jams_invoiced.pkl":(1,10,10,"jams",3,"woolworths_beerenberg_jams_invoiced"),   # special price cat, productgroup,productcode,product, on_promo, name)
          "woolworths_beerenberg_SJ300_invoiced.pkl":(1,10,10,"SJ300",3,"woolworths_beerenberg_SJ300_invoiced"),
          "woolworths_beerenberg_RJ300_invoiced.pkl":(1,10,10,"RJ300",3,"woolworths_beerenberg_RJ300_invoiced"),
          "woolworths_beerenberg_BOM300_invoiced.pkl":(1,10,10,"BOM300",3,"woolworths_beerenberg_BOM300_invoiced"),
          "woolworths_beerenberg_AJ300_invoiced.pkl":(1,10,10,"AJ300",3,"woolworths_beerenberg_AJ300_invoiced"),
          "woolworths_beerenberg_BB300_invoiced.pkl":(1,10,10,"BB300",3,"woolworths_beerenberg_BB300_invoiced"),
          "woolworths_beerenberg_BLJ300_invoiced.pkl":(1,10,10,"BLJ300",3,"woolworths_beerenberg_BLJ300_invoiced"),
          "woolworths_beerenberg_TS300_invoiced.pkl":(1,10,11,"TS300",3,"woolworths_beerenberg_TS300_invoiced"),
          "woolworths_beerenberg_BUR280_invoiced.pkl":(1,10,13,"BUR280",3,"woolworths_beerenberg_BUR280_invoiced"),
          "woolworths_beerenberg_TC260_invoiced.pkl":(1,10,13,"TC260",3,"woolworths_beerenberg_TC260_invoiced"),
          "woolworths_beerenberg_HTC260_invoiced.pkl":(1,10,13,"HTC260",3,"woolworths_beerenberg_HTC260_invoiced"),
          "woolworths_beerenberg_BBR280_invoiced.pkl":(1,10,13,"BBR280",3,"woolworths_beerenberg_BBR280_invoiced"),
          "woolworths_beerenberg_CAR280_invoiced.pkl":(1,10,13,"CAR280",3,"woolworths_beerenberg_CAR280_invoiced"),
          "woolworths_beerenberg_TCP260_invoiced.pkl":(1,10,13,"TCP260",3,"woolworths_beerenberg_TCP260_invoiced")}
    
    




     

   
# value is (brand,specialpricecat, productgroup, product,name)
product_type = namedtuple("product_type", ["brandno","customercat", "productgroup","product","on_promo","name"])

coles_and_ww_col_dict= {  "scan_week":(0,12,0,'_*',0,'scan_week'),
            1:(0,12,10,"jams",0,"coles_total_jam_curd_marm_off_promo_scanned"),
            2:(0,12,10,"jams",1,"coles_total_jam_curd_marm_on_promo_scanned"),
            3:(1,12,10,"jams",0,"coles_beerenberg_jams_off_promo_scanned"),
            4:(1,12,10,"jams",1,"coles_beerenberg_jams_on_promo_scanned"),
            5:(2,12,10,"jams",0,"coles_st_dalfour_jams_off_promo_scanned"),
            6:(2,12,10,"jams",1,"coles_st_dalfour_jams_on_promo_scanned"),
            7:(3,12,10,"jams",0,"coles_bonne_maman_jams_off_promo_scanned"),
            8:(3,12,10,"jams",1,"coles_bonne_maman_jams_on_promo_scanned"),
            9:(1,12,10,"SJ300",0,"coles_beerenberg_SJ300_off_promo_scanned"),
            10:(1,12,10,"SJ300",1,"coles_beerenberg_SJ300_on_promo_scanned"),
            11:(1,12,10,"RJ300",0,"coles_beerenberg_RJ300_off_promo_scanned"),
            12:(1,12,10,"RJ300",1,"coles_beerenberg_RJ300_on_promo_scanned"),
            13:(1,12,10,"OM300",0,"coles_beerenberg_OM300_off_promo_scanned"),
            14:(1,12,10,"OM300",1,"coles_beerenberg_OM300_on_promo_scanned"),
            15:(1,12,10,"AJ300",0,"coles_beerenberg_AJ300_off_promo_scanned"),
            16:(1,12,10,"AJ300",1,"coles_beerenberg_AJ300_on_promo_scanned"),
            17:(1,12,13,"TC260",0,"coles_beerenberg_TC260_off_promo_scanned"),
            18:(1,12,13,"TC260",1,"coles_beerenberg_TC260_on_promo_scanned"),
            19:(1,12,13,"HTC260",0,"coles_beerenberg_HTC260_off_promo_scanned"),
            20:(1,12,13,"HTC260",1,"coles_beerenberg_HTC260_on_promo_scanned"),
            21:(1,12,13,"CAR280",0,"coles_beerenberg_CAR280_off_promo_scanned"),
            22:(1,12,13,"CAR280",1,"coles_beerenberg_CAR280_on_promo_scanned"),
            23:(1,12,13,"BBR280",0,"coles_beerenberg_BBR280_off_promo_scanned"),
            24:(1,12,13,"BBR280",1,"coles_beerenberg_BBR280_on_promo_scanned"),
            25:(1,12,11,"TS300",0,"coles_beerenberg_TS300_off_promo_scanned"),
            26:(1,12,11,"TS300",1,"coles_beerenberg_TS300_on_promo_scanned"),
            27:(1,12,14,"PCD300",0,"coles_beerenberg_PCD300_off_promo_scanned"),
            28:(1,12,14,"PCD300",1,"coles_beerenberg_PCD300_on_promo_scanned"),
            29:(1,12,14,"BLU300",0,"coles_beerenberg_BLU300_off_promo_scanned"),
            30:(1,12,14,"BLU300",1,"coles_beerenberg_BLU300_on_promo_scanned"),
            31:(1,12,14,"RAN300",0,"coles_beerenberg_RAN300_off_promo_scanned"),
            32:(1,12,14,"RAN300",1,"coles_beerenberg_RAN300_on_promo_scanned"),
                33:(0,10,10,"jams",0,"woolworths_total_jam_curd_marm_off_promo_scanned"),
                34:(0,10,10,"jams",1,"woolworths_total_jam_curd_marm_on_promo_scanned"),
                35:(1,10,10,"jams",0,"woolworths_beerenberg_jams_off_promo_scanned"),
                36:(1,10,10,"jams",1,"woolworths_beerenberg_jams_on_promo_scanned"),
                37:(2,10,10,"jams",0,"woolworths_st_dalfour_jams_off_promo_scanned"),
                38:(2,10,10,"jams",1,"woolworths_st_dalfour_jams_on_promo_scanned"),
                39:(3,10,10,"jams",0,"woolworths_bonne_maman_jams_off_promo_scanned"),
                40:(3,10,10,"jams",1,"woolworths_bonne_maman_jams_on_promo_scanned"),
                41:(1,10,10,"SJ300",0,"woolworths_beerenberg_SJ300_off_promo_scanned"),
                42:(1,10,10,"SJ300",1,"woolworths_beerenberg_SJ300_on_promo_scanned"),
                43:(1,10,10,"RJ300",0,"woolworths_beerenberg_RJ300_off_promo_scanned"),
                44:(1,10,10,"RJ300",1,"woolworths_beerenberg_RJ300_on_promo_scanned"),
                45:(1,10,10,"BOM300",0,"woolworths_beerenberg_BOM300_off_promo_scanned"),
                46:(1,10,10,"BOM300",1,"woolworths_beerenberg_BOM300_on_promo_scanned"),
                47:(1,10,10,"AJ300",0,"woolworths_beerenberg_AJ300_off_promo_scanned"),
                48:(1,10,10,"AJ300",1,"woolworths_beerenberg_AJ300_on_promo_scanned"),
                49:(1,10,10,"BB300",0,"woolworths_beerenberg_BB300_off_promo_scanned"),
                50:(1,10,10,"BB300",1,"woolworths_beerenberg_BB300_on_promo_scanned"),
                51:(1,10,10,"BLJ300",0,"woolworths_beerenberg_BLJ300_off_promo_scanned"),
                52:(1,10,10,"BLJ300",1,"woolworths_beerenberg_BLJ300_on_promo_scanned"),
                53:(1,10,11,"TS300",0,"woolworths_beerenberg_TS300_off_promo_scanned"),
                54:(1,10,11,"TS300",1,"woolworths_beerenberg_TS300_on_promo_scanned"),
                55:(1,10,13,"BUR260",0,"woolworths_beerenberg_BUR260_off_promo_scanned"),
                56:(1,10,13,"BUR260",1,"woolworths_beerenberg_BUR260_on_promo_scanned"),
                57:(1,10,13,"TC260",0,"woolworths_beerenberg_TC260_off_promo_scanned"),
                58:(1,10,13,"TC260",1,"woolworths_beerenberg_TC260_on_promo_scanned"),
                59:(1,10,13,"HTC260",0,"woolworths_beerenberg_HTC260_off_promo_scanned"),
                60:(1,10,13,"HTC260",1,"woolworths_beerenberg_HTC260_on_promo_scanned"),
                61:(1,10,13,"BBR280",0,"woolworths_beerenberg_BBR280_off_promo_scanned"),
                62:(1,10,13,"BBR280",1,"woolworths_beerenberg_BBR280_on_promo_scanned"),
                63:(1,10,13,"CAR280",0,"woolworths_beerenberg_CAR280_off_promo_scanned"),
                64:(1,10,13,"CAR280",1,"woolworths_beerenberg_CAR280_on_promo_scanned"),
                65:(1,10,13,"TCP280",0,"woolworths_beerenberg_TCP280_off_promo_scanned"),
                66:(1,10,13,"TCP280",1,"woolworths_beerenberg_TCP280_on_promo_scanned")}

 




# df[0,12,10,"_T",0,'coles_jams_total_scanned']=df[0,12,10,"_*",0,'coles_total_jam_curd_marm_off_promo_scanned']+df[0,12,10,"_*",1,'coles_total_jam_curd_marm_on_promo_scanned']
#     df[1,12,10,"_t",0,'coles_beerenberg_jams_total_scanned']=df[1,12,10,"_*",0,'coles_beerenberg_jams_off_promo_scanned']+df[1,12,10,"_*",1,'coles_beerenberg_jams_on_promo_scanned']
#     df[2,12,10,"_t",0,'coles_st_dalfour_jams_total_scanned']=df[2,12,10,"_*",0,'coles_st_dalfour_jams_off_promo_scanned']+df[2,12,10,"_*",1,'coles_st_dalfour_jams_on_promo_scanned']
#     df[3,12,10,"_t",0,'coles_bonne_maman_jams_total_scanned']=df[3,12,10,"_*",0,'coles_bonne_maman_jams_off_promo_scanned']+df[3,12,10,"_*",1,'coles_bonne_maman_jams_on_promo_scanned']
#     #df=df*1000
    
#     df[1,12,10,"_t",0,'coles_beerenberg_jams_on_promo']=(df[1,12,10,"_*",1,'coles_beerenberg_jams_on_promo_scanned']>0)
#     df[2,12,10,"_t",0,'coles_st_dalfour_jams_on_promo']=(df[2,12,10,"_*",1,'coles_st_dalfour_jams_on_promo_scanned']>0)
#     df[3,12,10,"_t",0,'coles_bonne_maman_jams_on_promo']=(df[3,12,10,"_*",1,'coles_bonne_maman_jams_on_promo_scanned']>0)
    
#     df[0,10,10,"_T",0,'woolworths_jams_total_scanned']=df[0,10,10,"_*",0,'woolworths_total_jam_curd_marm_off_promo_scanned']+df[0,10,10,"_*",1,'woolworths_total_jam_curd_marm_on_promo_scanned']
    
#     df[1,10,10,"_t",0,'woolworths_beerenberg_jams_total_scanned']=df[1,10,10,"_*",0,'woolworths_beerenberg_jams_off_promo_scanned']+df[1,10,10,"_*",1,'woolworths_beerenberg_jams_on_promo_scanned']
#     df[2,10,10,"_t",0,'woolworths_st_dalfour_jams_total_scanned']=df[2,10,10,"_*",0,'woolworths_st_dalfour_jams_off_promo_scanned']+df[2,10,10,"_*",1,'woolworths_st_dalfour_jams_on_promo_scanned']
#     df[3,10,10,"_t",0,'woolworths_bonne_maman_jams_total_scanned']=df[3,10,10,"_*",0,'woolworths_bonne_maman_jams_off_promo_scanned']+df[3,10,10,"_*",1,'woolworths_bonne_maman_jams_on_promo_scanned']
     
#     df[1,10,10,"_t",0,'woolworths_beerenberg_jams_on_promo']=(df[1,10,10,"_*",1,'woolworths_beerenberg_jams_on_promo_scanned']>0)
#     df[2,10,10,"_t",0,'woolworths_st_dalfour_jams_on_promo']=(df[2,10,10,"_*",1,'woolworths_st_dalfour_jams_on_promo_scanned']>0)
#     df[3,10,10,"_t",0,'woolworths_bonne_maman_jams_on_promo']=(df[3,10,10,"_*",1,'woolworths_bonne_maman_jams_on_promo_scanned']>0)
    





coles_and_ww_convert_dict = {
          #      'scan_week': np.datetime64, 
                1: np.float64,
                2: np.float64,
                3: np.float64,
                4: np.float64,
                5: np.float64,
                6: np.float64,
                7: np.float64,
                8: np.float64,
                9: np.float64,
                10: np.float64,
                11: np.float64,
                12: np.float64,
                13: np.float64,
                14: np.float64,
                15: np.float64,
                16: np.float64,
                17: np.float64,
                18: np.float64,
                19: np.float64,
                20: np.float64,
                21: np.float64,
                22: np.float64,
                23: np.float64,
                24: np.float64,
                25: np.float64,
                26: np.float64,
                27: np.float64,
                28: np.float64,
                29: np.float64,
                30: np.float64,
                31: np.float64,
                32: np.float64,
                33: np.float64,
                34: np.float64,
                35: np.float64,
                36: np.float64,
                37: np.float64,
                38: np.float64,
                39: np.float64,
                40: np.float64,
                41: np.float64,
                42: np.float64,
                43: np.float64,
                44: np.float64,
                45: np.float64,
                46: np.float64,
                47: np.float64,
                48: np.float64,
                49: np.float64,
                50: np.float64,
                51: np.float64,
                52: np.float64,
                53: np.float64,
                54: np.float64,
                55: np.float64,
                56: np.float64,
                57: np.float64,
                58: np.float64,
                59: np.float64,
                60: np.float64,
                61: np.float64,
                62: np.float64,
                63: np.float64,
                64: np.float64,
                65: np.float64,
                66: np.float64} 
 



# scan_dict={"original_df":original_df,
#            "final_df":df,
#  #          "full_index_df":full_index_df,
#            "market_dict":market_dict,
#         #   "product_dict":product_dict,
#            "measure_conversion_dict":measure_conversion_dict,
#            "stacked_conversion_dict":stacked_conversion_dict,
#            'plot_type_dict':plot_type_dict,
#            'brand_dict':brand_dict,
#            'category_dict':category_dict,
#            'variety_type_dict':variety_type_dict,
#            'second_y_axis_conversion_dict':second_y_axis_conversion_dict,
#            'reverse_conversion_dict':reverse_conversion_dict}



report_dict={report("report_type_dict",0,"",""):report_type_dict,
             report("coles_and_ww_pkl_dict",0,"",""):coles_and_ww_pkl_dict}


