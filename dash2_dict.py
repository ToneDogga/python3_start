#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:27:40 2020

@author: tonedogga
"""


# =============================================================================
# 
# #  architecture                                inputs                         outputs
####  ------------------------------------------------------------------------------------------------
# #  dash2_class.sales.load                   list of filenames              a pandas df
# #  dash2_class.sales.preprocess                   df                            df
# #  dash2_class.sales.save                         df                       a df pickled file name
# 
# #  dash2_class.sales.query.load              a df pickled filename              df  
# #  dash2_class.sales.query.preprocess             df                           df
# #  dash2_class.sales.query.query          df, query dict                a directory of pickled df's           
# 
# #  dash2_class.sales.pivot.load          query dict, a dir of pkl df's         a dict of df's
# #  dash2_class.sales.pivot.preprocess           a dict of dfs                  a dict of df's
# #  dash2_class.sales.pivot.pivot           a dict of dfs, pivot_desc      a dict of df's   
# #  dash2_class.sales.pivot.save            a dict of df's                 a directory of excel spreadsheets
# 
# #  dash2_class.sales.plot.load           query dict, a dir of pkl dfs          a dict of df's
# #  dash2_class.sales.plot.preprocess            a dict of df's                 a dict of df's
# #  dash2_class.sales.plot.mat                  a dict of df's                a output dir of plots  
# #  dash2_class.sales.plot.compare_customer
# #  dash2_class.sales.plot.trend
# #  dash2_class.sales.plot.yoy_customer
# #  dash2_class.sales.plot.yoy_product
# #  dash2_class.sales.plot.pareto_customer
# #  dash2_class.sales.plot.pareto_product
# 
# #  dash2_class.sales.predict.load           query dict, a dir of pkl dfs          a dict of df's
# #  dash2_class.sales.predict.preprocess            a dict of df's                 a dict of df's
# #  dash2_class.sales.predict.next_week                X_train,y_train, validate, test     df
# #  dash2_class.sales.predict.actual_vs_expected
# 
# 
# =============================================================================
#  dash2_class.scan


#  dash2_class.price

###########################################################################################################################3
#  dash2_dict
#


from collections import namedtuple
import pandas as pd
from pandas.tseries.frequencies import to_offset
import datetime as dt



dash2_dict={
    "sales":{
        "in_dir":"./",
        "in_files":["allsalestrans2018.xlsx","allsalestrans2020.xlsx","salestrans.xlsx"],
        "save_dir":"./dash2_saves/",
        "output_dir":"./dash2_outputs/",
        "savefile":"salessavefile.pkl",
        "raw_savefile":"raw_savefile.pkl",
        "query_dict_savefile":"query_dict_savefile.pkl",
        "smoothing_mat":4,
        "annual_mat":52,
        "print_report":'AA_dash2_weekly_sales_snapshot.txt',
        "rename_columns_dict":{'specialpricecat':'spc','productgroup':'pg'},
      #  "glset_not_spc_mask_flag":False, #True,
      #  "glset_only":["SHP","NAT","DFS","EXS","ADM","ONL","CON"],    
        "pg_only":["10","11","12","13","14","15","16","17"],   #,"18"]
        "spcs_only":['012','010','080','088','020',"028",'030',"038",'040',"048",'050','060','070',"122","107"],   #,"028"]   #,"038","048","028","080","020","030","040']
        "spc_only":[12,10,80,88,20,28,30,38,40,48,50,60,70,122,107],   #,"028"]   #,"038","048","028","080","020","030","040']

        "queries":{
            "028 jan-nov 2019":[['AND',('spc',28)],["BD",("date",pd.to_datetime("2019-01-01"),pd.to_datetime("2019-11-30"))]],
            "088 jan-nov 2019":[['AND',('spc',88)],["BD",("date",pd.to_datetime("2019-01-01"),pd.to_datetime("2019-11-30"))]],
            "038 jan-nov 2019":[['AND',('spc',38)],["BD",("date",pd.to_datetime("2019-01-01"),pd.to_datetime("2019-11-30"))]],
            "048 jan-nov 2019":[['AND',('spc',48)],["BD",("date",pd.to_datetime("2019-01-01"),pd.to_datetime("2019-11-30"))]],
            "028 jan-nov 2020":[['AND',('spc',28)],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-11-30"))]],
            "088 jan-nov 2020":[['AND',('spc',88)],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-11-30"))]],
            "038 jan-nov 2020":[['AND',('spc',38)],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-11-30"))]],
            "048 jan-nov 2020":[['AND',('spc',48)],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-11-30"))]],

           # "devings_2019":[['AND',('code','DEVFIN')],["BD",("date",pd.to_datetime("2019-01-01"),pd.to_datetime("2019-12-31"))]],
           # "devings_all":[['AND',('code','DEVFIN')]],
         #   "2020 devings":[["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-12-31"))]],

            

              #  "g":[["OR",("pg","13"),("pg","15"),("pg","16"),("pg","17")]],   
         #   "Beerenberg 150ml-290g condiments 2020 pg (13 16 17)":[["OR",("pg","13"),("pg","16"),("pg","17")],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-12-31"))]],   
         #   "Beerenberg mealbases 2020 pg (14)":[["AND",("pg","14")],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-12-31"))]],   
         #   "Beerenberg SA 2020 spc (088)":[["AND",("spc",88)],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-12-31"))]],   
        #    "Beerenberg SJ300 not coles or WW":[["AND",("product","SJ300")],["NOT",("spc",10)],["NOT",("spc",12)]],   #,["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-12-31"))]],   
        #    "Beerenberg SJ300 2020 not coles or WW":[["AND",("product","SJ300")],['AND',('spc',88)],["NOT",("spc",10)],["NOT",("spc",12)],["BD",("date",pd.to_datetime("2020-01-01"),pd.to_datetime("2020-12-31"))]],   

        #    "Harris farm condiments 13 only":[["AND",("pg","13")]],   
        #    "Harris farm jams":[["AND",("pg","10")]],   
        #    "Harris farm sauces":[["AND",("pg","11")]],   
        #    "Harris farm mealbases":[["AND",("pg","14")]],   
        #    "Harris farm dressings":[["AND",("pg","12")]],   
 
       #     "FLNOR sauces":[["OR",("pg","11"),("code","FLNOR")]],
        #    'FLPAS SJ300':[["AND",("product","SJ300"),("code","FLPAS")]],
            
            
         # these queries below cannot be removed as they power the distribution and trend analysis   
            'last_today_to_365_days':[["BD",("date",pd.to_datetime("today")+pd.offsets.Day(-365),pd.to_datetime("today"))]],
            'last_365_to_731_days':[["BD",("date",pd.to_datetime("today")+pd.offsets.Day(-731),pd.to_datetime("today")+pd.offsets.Day(-365))]]  ,
            'all':[[]]
            #   'last_today_to_731_days':[["BD",("date",pd.to_datetime("today")+pd.offsets.Day(-731),pd.to_datetime("today"))]]
     #      'cnot_pasa_ts2':[['NOT',("product","TS300"),("code","FLPAS"),("salesrep","36")]],
      #      'ctns':[["B",("qty",8,16)]],
      #      'between_dates':[["BD",("date",pd.to_datetime("2018-01-01")+pd.offsets.Day(7),pd.to_datetime("today"))]],
     #      'cnot_pasa_ts3':[['OR',("product","TS300"),("code","FLPAS")]]
 # =============================================================================
#         
#         #   query of AND's - input a list of tuples.  ["AND",(field_name1,value1) and (field_name2,value2) and ...]
#             the first element is the type of query  -"&"-AND, "|"-OR, "!"-NOT, "B"-between
# #            return a slice of the df as a copy
# # 
# #        a query of OR 's  -  input a list of tuples.  ["OR",(field_name1,value1) or (field_name2,value2) or ...]
# #            return a slice of the df as a copy
# #
# #        a query_between is only a triple tuple  ["BD",(fieldname,startvalue,endvalue)]
#                "BD" for between dates, "B" for between numbers or strings
# # 
# #        a query_not is only a single triple tuple ["NOT",(fieldname,value)]   
#
#          lists of queries within lists are ANDed together as a final step
# 
#         
# =========================================================================

            },
        "pivots":{
            "distribution_report_dollars":"distribution_report_dollars.xlsx",
            "distribution_report_units":"distribution_report_units.xlsx",
            "distribution_report_dates":"distribution_report_dates.xlsx",
            "trend_heatmap_filename":"trend_heatmap.xlsx"
         #   "pivot1":{"values":"salesval","index":['glset','specialpricecat','code'],"columns":['year','month'],"savename":"pivot1.xlsx"}
            },  
        "plots":{
            "customers_to_plot_together":['FLNOR',"FLFRE","FLPAS","FLMIT"],   #"FLDAW","GLENORC","IGAATH"]  #"FLPAS","FLMIT","FLBRI"]
            "products_to_plot_together":['TS300',"CS300","BBQ300"],   
            "output_dir":"./dash2_outputs/",
            "scaling_point_week_no":51,
            "max_slope":0.16,
            "min_slope":-0.1,
            "min_size_for_trend_plot":6
            },    
        "predictions":{
            "save_dir":"./dash2_saves/",
            "RFR_order_predict_model_savefile":"RFR_order_predict_model.pkl",
         #   "GRU_order_predict_model_savefile":"_GRU_order_predict_model.h5",
            "invoiced_sales_weeks_offset":3,
            "rfr_mat":3,
            "batch_length":365,
            "no_of_batches":1000,            
            "no_of_repeats":2,
            "epochs":6,
            "dropout_rate":0.2,
            "start_point":0,
            "end_point":732,
            "predict_ahead_length":436,
            "minimum_length":800,
            "batch_length":365,
            "predict_length":365,
    #       "one_year":366,
    #     batch_jump=self.batch_length,  # predict ahead steps in days. This is different to the batch length  #int(round(self.batch_length/2,0))
            "neurons":1000,   #self.batch_length
            "data_start_date":"02/02/18",
    #    date_len=1300,      
    # #   dates = pd.period_range(self.data_start_date, periods=self.date_len),   # 2000 days
            "dollars":False,
            "pred_error_sample_size":12,
            "no_of_stddevs_on_error_bars":1,
            "patience":5,
            "mat":28,   #moving average in days #tf.constant([28],dtype=tf.int32)
            "train_percent":0.7,
            "valid_percent":0.2,
            "test_percent":0.1,
            "avepredfile":"ave_pred_file.xlsx",
            "units_invoiced_X_y":"units_invoiced_X_y.xlsx",
            "units_invoiced_X_y_savefile":"units_invoiced_X_y.pkl"
 
             }
        },
    
    "scan":{
        "scan_data_list":["coles_scan_data_enhanced_sept2020.xlsx","ww_scan_data_enhanced_sept2020.xlsx"], 
        "transposed_scan_data_list":["coles_scan_dataT.xlsx","ww_scan_dataT.xlsx"], 
        "scan_monthly_data_list":["chutneys_UPSPW.xlsx","sauces_UPSPW.xlsx","jams_UPSPW.xlsx","chutneys_units.xlsx","sauces_units.xlsx","jams_units.xlsx"],
        "e_scandata_number_of_weeks":53, 
        # "variety_mask_flag":False,
        # "variety_mask":[1,2,3],
        "queries":{
          #  "all":[[]],
            "Rosella UPSPW":[["AND",("brand_number",18),('measure_number',1)]],
          #  "brand three threes and heinz":[["AND",("brand_number",3)],['NOT',('measure_number',2)]],
          #  "coles only units (000)":[['AND',("retailer_number",12),('measure_number',2)]],
            "Coles st dalfour":[['AND',("retailer_number",12),('brand_number',2)]]
        #    "retailer choices":[["AND",("pg","10"),("spc",88)]],
            
        },    
      #  "scan_monthly_brand_include_mask_flag":False,
      #  "scan_monthly_brand_include_mask":[1,2,3],
        "in_dir":"./",
        "savefile":"scansavefile.pkl",
        "monthlysavefile":"scanmonthlysavefile.pkl",
        "save_dir":"./dash2_saves/",
        "output_dir":"./dash2_outputs/"
        },
    
    "price":{
        "in_file":"salestrans.xlsx",
        "in_dir":"./",
        "price_width":60,  # columns to import from salestrans.xlsx on price sheet
        "savefile":"pricesavefile.pkl",
        "save_dir":"./dash2_saves/",
        "output_dir":"./dash2_outputs/",
        "price_discrepencies_summary":"price_30_day_underpriced_summary.xlsx"
 

        }, 
    
    "production":{
        "save_dir":"./dash2_saves/",
        "SOH_savefile":"SOH_savefile.pkl",
        "PP_savefile":"PP_savefile.pkl",
        "PM_savefile":"PM_savefile.pkl",
        "stock_level_query":'stock_level_query.xlsx',
        "in_dir":"./",
        "low_stock_limit":800000,
        "production_made_query":'Production Schedule.xlsx',
        "production_made_sheet":"Schedule",
        "production_planned_query":'B Stock & Schedule Forecast.xlsx',
        "production_planned_sheet":"Schedule"     
         }, 
    
    "scheduler":{
        "schedule_savefile":"schedule_savefile.pkl",
        "schedule_savedir":"./dash2_saves/",
        "schedule_save_excel":"schedule.xlsx",
        "schedule_savedir_plots":"./dash2_saves/plots/",
        "schedule_savedir_resized_plots":"./dash2_saves/resized_plots/",
        "schedule_savefile_plots_gif":"recommended_schedule.gif",
        "schedule_savefile_plots_mp4":"schedule_options2.mp4",
    #    "schedule_plots_savedir":"./dash2_saves/plots/",
 
        "productgroup_mask":["10","11","12","13","14","15","16","17"],
        "productgroup_dict":{                   
             "Jams 250ml glass jar":"10",
             "Sauce 300ml glass bottle":"11",
             "Dressings 300ml glass bottle":"12",
             "Condiments 250ml glass jar":"13",
             "Meal bases 250ml glass jar":"14",
             "Condiments for cheese 150ml glass jar":"15",
             "Traditional condiments 150ml glass jar":"16",
             "Mustards 150ml glass jar":"17"
            },


        "pg_type":{
                    
                   # 0 for winter product group
                   # 1 for summer product group 
                   # that means they use different demand curves
                   
            "10":0,
            "11":1,
            "12":1,
            "13":0,
            "14":0,
            "15":1,
            "16":1,
            "17":0
            },
 

        # "productgroup_type":{
                    
        #            # 0 for winter product group
        #            # 1 for summer product group 
        #            # that means they use different demand curves
                   
        #     "Jams 250ml glass jar":0,
        #     "Sauce 300ml glass bottle":1,
        #     "Dressings 300ml glass bottle":1,
        #     "Condiments 250ml glass jar":0,
        #     "Meal bases 250ml glass jar":0,
        #     "Condiments for cheese 150ml glass jar":1,
        #     "Traditional condiments 150ml glass jar":1,
        #     "Mustards 150ml glass jar":0
        #     },
 
    
        "format_type":{
            "Jams 250ml glass jar":0,
            "Sauce 300ml glass bottle":1,
            "Dressings 300ml glass bottle":1,
            "Condiments 250ml glass jar":0,
            "Meal bases 250ml glass jar":0,
            "Condiments for cheese 150ml glass jar":2,
            "Traditional condiments 150ml glass jar":2,
            "Mustards 150ml glass jar":2
             },
 
        "pg_yield":{
            "10":600,
          #  "Honey 250ml glass jar":800,
            "11":550,
            "12":550,
            "13":700,
            "14":700,
            "15":1000,
            "16":1000,
            "17":1000
             },    

        
        # "format_yield":{
        #     "Jams 250ml glass jar":600,
        #   #  "Honey 250ml glass jar":800,
        #     "Sauce 300ml glass bottle":550,
        #     "Dressings 300ml glass bottle":550,
        #     "Condiments 250ml glass jar":700,
        #     "Meal bases 250ml glass jar":700,
        #     "Condiments for cheese 150ml glass jar":1000,
        #     "Traditional condiments 150ml glass jar":1000,
        #     "Mustards 150ml glass jar":1000
        #      },    

        # maximum number of batches that can be done in a single shift depends on the key which is the number of varietial changeovers 
        "stacker_productivity":{
            1:100,
            2:80,
            3:60,
            4:40,
            5:30,
            6:20
            }
        }
    }



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Ideas for future expansion and improvement
#
#    Measure sales kg / week vs production kg/ week  calculate a MSE andmake a graph of
# scheduling accuracy.
#
#     use a clustering algorithm like K-means to cluster 088 customers into groups
#
#
#     Overlay a 2021 expected graph trained on 2/2/19 - 2/2/21 actuals in a different colour to
#     actual vs expected
#
#     recheck the logic and precision of scan data ahead predict





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
                   "31":"2 Litre",
                   "999":"999"}


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
                   31:"2 Litre",
                   999:""}





salesrep_dict={"14":"Brendan Selby",
               "8":"Noel Lotorto",
               "33":"Howard Lee",
               "34":"Russell Heyzer",
               "35":"Stephen Wood",
               "36":"Annette Paech",
               "37":"Miles Rafferty",
               "39":"Sophia Simos",
               "11":"Coles and WW",
               "999":""}
               


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
          95:"Online shop",
          999:""}


spcs_dict={"122":"Harris farms",
          "107":"Lite n easy",
          "010":"woolworths",
          "012":"coles",
          "088":"SA stores",
          "080":"SA distributors",
          "020":"NSW distributors",
          "040":"QLD distributors",
          "030":"VIC distributors",
          "038":"VIC stores",
          "028":"NSW stores",
          "048":"QLD stores",
          "050":"WA distributors",
          "070":"TAS distributors",
          "092":"Shop",
          "095":"Online shop",
          "999":""}



retailers_name_dict={'AU Coles Group scan':'Coles', 'AU Woolworths scan':'Woolworths', 'AU Grocery scan':'Coles'}
retailers_number_dict={'Coles':12, 'Woolworths':10}

brand_number_dict={'beeren':1, 
                   'three ':5, 
                   'heinz ':15, 
                   'maleny':14, 
                   'st dal':2, 
                   'anatho':6,
                   'jills ':8, 
                   'master':9, 
                   'fehlbe':77, 
                   'rosell':18,
                   'sapphi':11,
                   'stubbs':12,
                   'baxter':7, 
                   'cottee':4,
                   'bonne ':3, 
                   'f. whi':16,
                   'always':17, 
                   'spring':10, 
                   'palms ':19,
                   "hank's":20,
                   'dick s':21,
                   'yackan':22,
                   'coles ':44,
                   'countr':24,
                   'riveri':25,
                   'anatho':26,
                   'homebr':27,
                   'buderi':28,
                   'ozespr':29,
                   'nandos':30,
                   'h.p.':31,
                   'woolwo':32,
                   'eta':33, 
                   'outbac':34,
                   'spirit':35,
                   'fletch':36,
                   "d.l. j":37,
                   'ixl':38,
                   "barker":39,
                   "bull's":40,
                   'branst':41,
                   'roses ':42,
                   'yarra ':43,
                   'burges':66,
                   'rockma':45,
                   "mrs h.":46,
                   'founta':47,
                   'port m':48,
                   'the ol':49,
                   'rhu br':50,
                   'maggie':51,
                   'macro':52,
                   'mon':53,
                   'office':54,
                   'natvia':55,
                   'hill &':56,
                   'white':57,
                   'leggos':58,
                   'pops':59,
                   'sandhu':60
                    
                   
                   }


measure_number_dict={
                   'Units/Store/Week':1,
                   'Units (000)':2}

# variety_name_dict={
#                     'BBQ  ':1,
#                     'Sweet Mustard Pickles 250g', 'St Dalfour Red Raspberry Spread 284g', 'Beerenberg Blueberry 300g', 'MasterFoods Classic Relish Corn 250g', 'IXL Lite 25% Less Sugar Strawberry 240g', "Bull's-eye BBQ Original 300ml", 'Yackandandah Lemon Curd 350g', "Barker's Chilli & Red Onion Filling 1.25kg", 'IXL Plum 250g', 'Fletchers Sweet Mustard Pickles 395g', 'Heinz Tom Ketchup Less Sugar 500ml', 'F. Whitlock & Sons Chutney Tomato 275g', 'MasterFoods BBQ Rch Tst Sqz 250ml', "D.L. Jardine's BBQ 5 Star 510ml", 'Riverina Grove Relish Tomato 250g', 'Anathoth Farm Lemon Curd 450g', 'Woolworths Select BBQ Squeeze 500ml', 'Roses Sweet Orange Marmalade 500g', 'Branston Pickles 310g', 'Woolworths Select Apricot 500g', 'Cottees Strawberry 250g', 'Rhu Bru Rhubarb Ginger 290g', 'Coles 45% Fruit Breakfast Marmalade 500g', 'Natvia Raspberry Fruit Spread 240g', 'St Dalfour Kumquat 284g', 'MasterFoods Cafe Relish Tomato Green 250g', 'Stubbs BBQ Original 510ml', 'Dick Smith Spring Gully Mgnif Orange Marmalade Spreadble Fruit 285g', 'Rhu Bru Rhubarb Raspberry 290g', 'Maggie Beer Chutney Sultan 270g', 'Outback Spirit Chutney Tom 285g', 'Rosella Sweet Mustard Pickles 500g', 'Dick Smith Spring Gully Mgnif Raspberry Spreadble Fruit 285g', 'Woolworths Gold Strawberry 310g', 'Coles Chutney Fruit 270g', 'Fehlbergs Jalapenos 470g', 'Rhu Bru Rhubarb Lavender 290g', 'Beerenberg Roadhouse Steak Sauce 300ml', 'Coles Simply Less Raspberry 255g', "Hank's Jam Raspberry 285g", 'IXL Strawberry 480g', 'Buderim Ginger Original Ginger Marmalade 365g', 'Fountain Tomato Sce Sqz 500ml', 'MasterFoods BBQ Spicy Lmtd Edit 500ml', 'Fountain BBQ Sqz 500ml', 'Coles Raspberry 500g', 'Cottees Apricot 250g', 'Beerenberg Relish Hot Dog Ny Style 260g', 'Beerenberg BBQ 300ml', 'Beerenberg Blood Orange Marmalade 300g', 'Burgess Relish Caramelised Onion 210g', 'Cottees Raspberry 250g', 'Nandos Pepper Steak Table Sauce 250ml', 'Leggos Sweet Mustard Pickles 500g', 'F. Whitlock & Sons Chutney Peach Mango Apricot 275g', 'Beerenberg Garlic Sce 300ml', "Hank's Jam Strawberry Marmalade 285g", 'Macro Orgnc Strawberry 250g', 'IXL Lite 25% Less Sugar Raspberry 240g', 'MasterFoods Classic Relish Gherkin 260g', 'Beerenberg Relish Tomato & Cracked Pepper 265g', 'Nandos Peri BBQ Squeeze 290ml', 'Natvia Strawberry Fruit Spread 240g', 'Spring Gully Pickled Sr Mustard 500g', 'Bonne Maman Raspberry Conserve 370g', 'Bonne Maman Four Fruits 370g', 'Macro Orgnc Apricot 250g', 'Beerenberg Raspberry 300g', 'H.P. Original Top Down 390ml', 'Heinz Tom Ketchup Top Down 220ml', 'Heinz Burger Sce Original 295ml', 'Maleny Cuisine Chilli Jam 300g', 'Macro Orgnc Raspberry 250g', 'Cottees Breakfast Marmalade 500g', 'St Dalfour Black Cherry Spread 284g', 'Baxters Chutney Mango 250g', 'Bonne Maman Blackberry Conserve 370g', 'Bonne Maman Blueberry Conserve 370g', "Hank's Jam Pear Vanilla 285g", 'MasterFoods Cafe Chutney Mango 250g', 'Homebrand Chutney Fruit 520g', 'Fountain Smart Tom NA/Sgr 500ml', 'Roses Raspberry Conserve 500g', 'Heinz BBQ Chilli Sauce 400ml', 'Sandhurst Relish Jalapeno 210g', 'Pops Tomato Sce 600ml', 'Ozespreads Mango & Lime 285g', 'Rosella Chutney Mango Habenero Coconut 250g', 'MasterFoods Sauce BBQ 2L', 'Macro Chutney Orgnc Tomato 275g', 'Beerenberg Double Shot Pepper Sce 300ml', "Bull's-eye BBQ Smokey Bacon 300ml", 'Fountain Tom Sce Sqz R/Slt/Sgr 500ml', 'MasterFoods Tomato Sce Sqz 920ml', 'Anathoth Farm Lemon Curd 420g', 'Woolworths Australian Tom Sce Squeeze 500ml', "Hank's Jam Pawpaw Lime Passion Fruit 285g", 'Heinz Ketchup Organic Upside Down 500ml', 'Coles 45% Fruit Apricot 500g', 'Hill & River Strawberry Rhubarb Vanilla 300g', "Barker's Passion Fruit Curd 400g", 'Cottees Raspberry 500g', 'MasterFoods BBQ Sqz 920ml', 'IXL Lite 50% Less Sugar Raspberry 220g', 'Leggos Sweet Mustard Pickles Spreadable 250g', 'Woolworths Australian BBQ Sauce Squeeze 500ml', 'Sapphire Beetroot & Horseradish Old Fashioned 227g', 'Beerenberg Chutney Takatala 280g', 'Baxters Chutney Fig Date & Balsamic 235g', 'Always Fresh Olive Kalamata Tapenade 230g', "D.L. Jardine's Chik'n Lik'n Mustard BBQ Sce 510g", 'Cottees Breakfast Marmalade 250g', 'Fountain BBQ 2L', 'Beerenberg Coopers BBQ Ale 300ml', 'IXL Raspberry 480g', 'Hill & River Fig & Ginger 300g', 'Port Macquarie Food Co Hastings Citrus Marmalade 300g', 'MasterFoods Lemon Butter 280g', "Officer's Mess Seville Orange Marmalade 310g", 'IXL Ginger Marmalade 480g', 'Always Fresh Red Cabbage Pickled 450g', 'Maleny Cuisine Onion Marm Sliced 300g', 'Beerenberg Sweet Mustard Pickles 265g', 'Heinz Tom Ketchup Pet 500ml', 'Coles Apricot 500g', 'Three Threes Mustard Pickles Spreadable 390g', 'Homebrand Tomato Sce 600ml', 'Beerenberg Blackberry 300g', 'Jills Cuisine Relish Trad 400g', 'Cottees Strawberry 500g', 'IXL Blackberry 480g', 'Beerenberg Sticky Rib 300ml', 'St Dalfour Blueberry Spread 284g', 'Woolworths Select Tomato Sce Sqz 500ml', 'Cottees Blackberry Conserve 500g', 'Rosella Tomato Sce 600ml', 'Dick Smith Spring Gully Mgnif Swt Fig Spreadble Fruit 285g', 'MasterFoods BBQ Sqz R/Slt 500ml', 'Coles Chutney Tom 250g', 'Beerenberg Strawberry 300g', "Mrs H.S.Ball's Chutney Mild 470g", "Mrs H.S.Ball's Chutney Peach 470g", 'MasterFoods Tom Sce Aus Grown Sqz 500ml', 'Baxters Chilli Jam 225g', 'Buderim Ginger Ginger Lemon Lime Marmalade 365g', 'Dick Smith Spring Gully Mgnif Strawberry Spreadble Fruit 285g', 'Woolworths Select Spreadable Raspberry Spreadble Fruit 285g', 'Coles Sweet Mustard Pickles 500g', 'Coles Breakfast Marmalade 500g', 'Coles Chutney Fruit 250g', 'Woolworths Secret Burger 330ml', "D.L. Jardine's Sweet Texas BBQ Sce 510ml", 'MasterFoods Cafe Relish Tomato 250g', 'Homebrand BBQ 2L', 'Beerenberg Chutney Tomato 260g', 'Homebrand Breakfast Marmalade 500g', 'Stubbs Hickory BBQ 510g', 'Woolworths Select Breakfast Marmalade 500g', 'Rosella Organic Tom Sce G/F 250g', 'Fountain Tom Sce V/Pack 2L', 'Heinz Tom Ketchup Pet 1L', 'Country Cuisine Plum Cinnamon 350g', 'Rosella Orgnc BBQ 250g', 'Cottees Plum 500g', 'F. Whitlock & Sons Relish Tomato & Smoky Chipotle 275g', 'Stubbs BBQ Spicy 510ml', 'Maleny Cuisine Chutney Eggplant 280g', 'Three Threes BBQ Rib 275ml', 'Always Fresh Bruschetta Relish 230g', 'Beerenberg Relish Balsamic Beetroot 280g', 'The Old Factory Chutney Tomato 400g', 'Yarra Valley Preserves Jumbleberry 230g', 'Beerenberg Chutney Mango 280g', 'St Dalfour Strawberry Spread 284g', 'Beerenberg Fig Almond 300g', 'IXL Apricot 250g', 'Beerenberg Chutney Tomato Hot 260g', 'Burgess Chutney Tom 210g', "Hank's Jam Chutney Tomato 240g", 'Roses Strawberry Conserve 500g', 'IXL Lite 50% Less Sugar Plum 220g', 'Coles Plum 500g', 'MasterFoods Tomato Sce Sqz 500ml', 'MasterFoods Tomato Sce 2L', 'St Dalfour Strawberry & Rhubarb 284g', 'Roses English Breakfast Marmalade 500g', 'H.P. Original 220ml', 'Roses Lime Marmalade 500g', 'Beerenberg Mustard Pickles 260g', 'ETA BBQ 375ml', 'Beerenberg Red Currant Jelly 300g', 'Coles 45% Frt Strawberry 500g', 'Heinz Burger Sce Bacon 295ml', "Bull's-eye Tangy Tomato BBQ 300ml", 'Rosella Chutney Fruit 525g', "Hank's Jam Chilli Jam 285g", 'Beerenberg Sticky Fig & Onion 280g', 'Palms Chutney Paw Paw Mango Sweet 325g', 'Rosella Relish Australian Corn 250g', 'Fountain BBQ Smart NA/Sgr 500ml', "Bull's-eye Dark Beer BBQ Sce 300ml", 'F. Whitlock & Sons Relish Caramelised Onion 275g', 'Country Cuisine West Whisky Marmalade 350g', 'MasterFoods Tomato Sce & Hidden Veg Sqz 500ml', 'White Crow Tomato Sce 2L', 'Fountain Tom Sce Spicy Red 250ml', 'Anathoth Farm Apricot 455g', 'Beerenberg Relish Burger Relish 260g', 'IXL Raspberry 250g', "D.L. Jardine's Mesquite BBQ 510g", 'Bonne Maman Orange Marmalade 370g', 'Rosella Tomato Sce Sqz 500ml', 'Homebrand BBQ 600ml', 'IXL Strawberry 250g', 'Sapphire Beetroot & Horseradish Extra Strong 227g', 'Spring Gully Strawberry Homestyle 295g', 'Beerenberg Relish Spicy BBQ Caramelised Onion 280g', 'IXL Breakfast Marmalade 480g', 'St Dalfour Fig Royal 284g', 'MasterFoods Reg Chutney Mango 250g', 'Beerenberg Apricot 300g', 'Macro Organic BBQ 250ml', 'MasterFoods Tom Sce Redcd Salt 475ml', 'Heinz Ketchup Edchup 500ml', 'Bonne Maman Rhubarb Conserve 370g', 'St Dalfour Seasonal Spread 284g', 'Maleny Cuisine Relish Spcy Tom 275g', 'Heinz Imp Pickled Ploughmans 320g', 'Spring Gully Apricot Homestyle 295g', 'IXL Apricot 480g', 'Three Threes Sweet Mustard Pickles 520g', 'Baxters Relish Caramelised Onion 240g', 'Ozespreads Cherry Berry 285g', 'MasterFoods Burger Sce 250ml', 'Always Fresh Paste Quince 100g', 'Rosella Chutney Fruit 250g', 'IXL Stevia Breakfast Marmalade 210g', 'Coles Premium Lemon Curd 330g', 'Three Threes Tomato Sce 275ml', 'Three Threes Relish Beetroot 250g', 'Heinz Tomato Ketchup 500ml', 'Bonne Maman Fruits Of The Forest Conserve 370g', 'Heinz Big Red Tom Sce Sqz 500ml', 'Rockman Brand Pickled Mustard 340g', 'IXL Apricot 600g', 'St Dalfour Apricot 284g', "Hank's Jam Chutney Ploughmans 285g", 'Woolworths Gold Relish Beetroot Horseradish 300g', 'Beerenberg Chilli Jam 300g', 'Cottees Mandarin Orange 500g', 'IXL Stevia Strawberry 210g', 'Coles Chutney Tomato 260g', 'MasterFoods Sauce BBQ Reduced Salt & Sugar 475ml', 'Yarra Valley Preserves Rhubarb Rasbrry Van Bean 220g', 'IXL Forest Fruits 480g', 'Beerenberg Chutney Cheeseboard 300g', 'Anathoth Farm Raspberry 455g', 'Macro Orgnc Orange Marmalade 250g', 'Anathoth Farm Boysenberry 455g', 'Anathoth Farm Three Berry 455g', 'Bonne Maman Apricot Conserve 370g', 'Heinz Burger Sce Chipotle 295ml', 'Homebrand Apricot 500g', 'Woolworths Select Blackcurrant 500g', 'Yackandandah Blackberry Apple 370g', 'Woolworths Gold Relish Cherry Tom Chilli 300g', 'St Dalfour 4 Fruit Nas 284g', 'Bonne Maman Fig Conserve 370g', 'MasterFoods BBQ Smokey Sqz 500ml', 'IXL Lite 25% Less Sugar Apricot 240g', 'MasterFoods BBQ Sauce American Steak 500ml', 'Yarra Valley Preserves Mango Vanilla Bean 230g', 'Beerenberg Chutney Fruit 290g', 'Baxters Chutney Classic Tomato 225g', 'Spring Gully Pickled Tomato Green 415g', 'Beerenberg Tomato Sce Hot 300ml', "Officer's Mess Strawberry Conserve 310g", 'Fletchers Chutney Fruit 420g', 'Rosella Chutney Apple Cider Vinegar 250g', 'Country Cuisine Summer Garden 220g', 'Beerenberg Relish Caramelised Onion 280g', 'Spring Gully Sweet Mustard Pickles 400g', 'Heinz BBQ Classic Sauce 400ml', 'Beerenberg Tomato Sce 300ml', 'Woolworths Select Plum 500g', 'Woolworths Select Raspberry 500g', 'Maleny Cuisine Chutney App/Fig/Gngr 280g', 'Woolworths Select Strawberry 500g', 'Jills Cuisine Relish Chilli Tomato 400g', 'Coles Strawberry 500g', 'Heinz Smokey BBQ Sauce 500ml', "Hank's Jam Triple Berry 285g", 'Country Cuisine Blackberry 220g', 'Hill & River Lemon & Lime Butter 270g', 'Maleny Cuisine Chutney Mango Apple Pumpkin 280g', 'Beerenberg Relish Sweet Chilli 280g', 'Sapphire Beetroot & Horseradish Sugar Free 227g', "Bull's-eye Sweet Whiskey Glaze 300ml", 'White Crow Tomato Sce 600ml', 'Rosella Chutney Tomato Peri Peri 250g', 'Fountain Steak 250ml', 'Ozespreads Apple Sultana & Cinnamon 285g', 'IXL Lite 50% Less Sugar Strawberry 220g', "Officer's Mess Raspberry Blackberry Conserve 300g", 'Ozespreads Spiced Plum 285g', 'Homebrand Strawberry 500g', 'Bonne Maman Black Cherry Conserve 370g', 'Always Fresh Relish Caramelised Onion 200g', 'Stubbs Sticky Sweet BBQ 510g', 'Anathoth Farm Strawberry 455g', 'St Dalfour Orange Marmalade 284g', 'MasterFoods BBQ Sqz 500ml', 'Rosella Tom Sce Glass Bottle 580ml', 'Port Macquarie Food Co Strawberry 300g', 'Mon Tomato Sce 580ml', 'Dick Smith Tom Sce Aus Grown G/F 500ml', 'Cottees Apricot Conserve 500g', 'Woolworths Tom Ketchup Sqz 487ml', 'Beerenberg Plum Satsuma 300g', 'Beerenberg Orange Marmalade 300g', 'Woolworths Tomato Sce 2L', 'Bonne Maman Strawberry Conserve 370g', 'IXL Lite 50% Less Sugar Apricot 220g'
    
    
    
#     }


# glset_dict={
#     NAT
# ONL
# SHP
# DFS
# EXS
# CON
# ADM


    
    
#     }
