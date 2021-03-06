import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd


infilename="NAT-raw310120all.xlsx"
#infilename="FLFRE_FLFUL_FLPAS-raw310120.xlsx"

#infilename="FLNOR_FLBRI-raw.xlsx"
#infilename="FLNOR-raw.xlsx"
#infilename="NAT-raw.xlsx"


importrows=-1  # all

code_encode_save="customer_encode_save.pkl"
product_encode_save="product_encode_save.pkl"
#   LY_product_encode_save="LY_product_encode_save.p"
#    glset_encode_save="glset_encode_save.p"

scaler_save="scaler.pkl"
SGDR_save="SGDRegressor.pkl"
LSVR_save="LinearSVRegressor.pkl"
SVR_save="SVRegressor.pkl"
RFR_save="RFRegressor.pkl"
SGDC_save="SGDClassifier.pkl"
RFC_save="RFClassifier.pkl"

outfile="RegressionResults.txt"
outfile_predict="PredictResults.txt"

outxlsfile="SalesPredictResults.xlsx"

datasetpluspredict="datasetpluspredict.csv"
datasetworking="datasetworking.csv"
#scalerdump1="prescaler1.xlsx"
scalerdump="scalerarraybyproduct.xlsx"
scalerdump2="scalerarraybyproduct2.xlsx"

##########################################################

#customer_codes=["FLNOR","FLBRI"]

#predict_start_date=dt.datetime.now
#one_year_ago = (dt.datetime.now()+relativedelta(years=-1)).strftime('%Y/%m/%d')

########################################
# reporting date format
#dateformat="year/month/day"
dateformat="year/week"
#dateformat="year/month"

#####################################################
# binning.  no columns or rows are excluded
#ycolno=1
#ybins=[0,8,16,24,32,40,56,80,2000]
#Xbins=[[],
#       []]

# convert dates into day count changes?
#day_deltas=True   
# date field to delta must be called 'date'

###########################
excludecols=["cat","glset","doctype","docentrynum","linenumber","location","refer","territory","specialpricecat"]  #2,3,4,5,6,7,9,10,12,13,14]  # this is for the coding and decoding (after binning and day delta conversion)
#excludecols=["cat","glset","doctype","docentrynum","linenumber","location","refer","salesrep","territory","specialpricecat"]  #2,3,4,5,6,7,9,10,12,13,14]  # this is for the coding and decoding (after binning and day delta conversion)

#excludecols=["cat","code","costval","glset","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"]  #2,3,4,5,6,7,9,10,12,13,14]  # this is for the coding and decoding (after binning and day delta conversion)
#featureorder=["prod_encode","qty","productgroup","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]
featureorder_r=["code_encode","product","qty","salesrep","productgroup","date","date_encode","day_delta","GMV"]    #,"day_order_delta"]   #,"day_of_year","week_of_year","month_of_year","year"]
featureorder_c=["code_encode","product","qty","salesrep","productgroup","date","date_encode","day_delta","GMV"]    #,"day_order_delta"]   #,"day_of_year","week_of_year","month_of_year","year"]
##############################
# forecasting purposes
# data starts 2/2/2018  with week 0 as a bin_no
startbin=51  # 1/2/2019
#backshift=80   #  weeks to look forward so we have to look back from the newest week of data

###################################################

#bins = [0,26,57,87,118,148,179,210,240,271,301,332,363,391,420,451,481,512,543,574,604,635,665,696,727,755,786,817,847,878,908,930,960,990,1020,1050,1080,1110,1140,1170,1200,1230]
bins = list(range(0,1400,7))
#noofbins=40
rescale_constant = 130   #24   # 0.69 is mean of all the scaler array by product .xlsx entries     productsame as 1/3   #0.2   # the factor that the scaling effects the calculated last_order_upspd
balance=0.88   # 80% on fixed element and 20% on sales distributions
###################################
#startdate=pd.to_datetime("20/07/01",format="%Y/%m/%d")
#finishdate=pd.to_datetime("21/06/30",format="%Y/%m/%d")
startdate=pd.to_datetime("2020/07/01",format="%Y/%m/%d")  #.dt.strftime("%Y/%m/%d")
finishdate=pd.to_datetime("2021/06/30",format="%Y/%m/%d") #.dt.strftime("%Y/%m/%d")
startdate = dt.datetime.strftime(startdate, '%Y/%m/%d')  # %H:%M:%S.%f')
finishdate = dt.datetime.strftime(finishdate, '%Y/%m/%d')  # %H:%M:%S.%f')


#################################3
#include_productgroup=[10,11,12,13,14,15,16]

#RF_estimators=4000  # regression
RF_estimators=1000  # regression
RFC_estimators=100   # classification
mintransactions=16
minqty=16
