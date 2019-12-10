import datetime as dt
from dateutil.relativedelta import relativedelta

#infilename="FLFUL_FLPAS_FLFRE-raw.xlsx"
#infilename="FLNOR_FLBRI-raw.xlsx"
#infilename="FLNOR-raw.xlsx"
infilename="NAT-raw.xlsx"


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


##########################################################

#customer_codes=["FLNOR","FLBRI"]

predict_start_date=dt.datetime.now
one_year_ago = (dt.datetime.now()+relativedelta(years=-1)).strftime('%Y/%m/%d')

########################################
# reporting date format
#dateformat="year/month/day"
dateformat="year/month"

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
excludecols=["cat","costval","glset","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"]  #2,3,4,5,6,7,9,10,12,13,14]  # this is for the coding and decoding (after binning and day delta conversion)
#excludecols=["cat","code","costval","glset","doctype","docentrynum","linenumber","location","refer","salesrep","salesval","territory","specialpricecat"]  #2,3,4,5,6,7,9,10,12,13,14]  # this is for the coding and decoding (after binning and day delta conversion)
#featureorder=["prod_encode","qty","productgroup","date","date_encode","day_delta","day_of_year","week_of_year","month_of_year","year"]
featureorder_r=["code_encode","prod_encode","qty","productgroup","date","date_encode","day_delta"]    #,"day_order_delta"]   #,"day_of_year","week_of_year","month_of_year","year"]
featureorder_c=["code_encode","prod_encode","qty","productgroup","date","date_encode","day_delta"]    #,"day_order_delta"]   #,"day_of_year","week_of_year","month_of_year","year"]
##############################

bins = [0,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,810,840,870,900,930,960,990]
noofbins=33
sensitivity_constant = 0.35   # same as 1/3   #0.2   # the factor that the scaling effects the calculated last_order_upspd



#################################3
#include_productgroup=[10,11,12,13,14,15,16]

#RF_estimators=4000  # regression
RF_estimators=1000  # regression
RFC_estimators=100   # classification
mintransactions=16
minqty=16
