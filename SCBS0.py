#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:58:04 2020

@author: tonedogga
"""


import numpy as np
predict_ahead_steps=440

 #   epochs_cnn=1
epochs_wavenet=20
no_of_batches=100000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
batch_length=16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
#    y_length=1
neurons=1600  #1000-2000
 
pred_error_sample_size=40

patience=5   #5

# dictionary mat type code :   aggsum field, name, color
   # mat_type_dict=dict({"u":["qty","units","b-"]
                  #  "d":["salesval","dollars","r-"],
                  #  "m":["margin","margin","m."]
#                   })
   
mats=[28]   #omving average window periods for each data column to add to series table
start_point=np.max(mats)+15  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
mat_types=["u"]  #,"d","m"]
   
units_per_ctn=8
   
# train validate test split 
train_percent=0.7
validate_percent=0.2
test_percent=0.1

filenames=["allsalestrans2020.xlsx","allsalestrans2018.xlsx"]

queryfilename="queryfile.xlsx"




   # filename="NAT-raw310120_no_shop_WW_Coles.xlsx"
#filename="NAT-raw240420_no_shop_WW_Coles.xlsx"
 
   #     filename="allsalestrans020218-190320.xlsx"   

#index_code=['code','productgroup'] 
 #   index_code=['code','productgroup'] 
   # index_code=['product','code'] 
  
#index_code=['code','product'] 
   # index_code=['code']
   # index_code=['product']  
   # index_code=['productgroup'] 
   # index_code=['code']
   # index_code=['code','productgroup','product']
###############################################33
  #  you also need to change the mask itself which is in the load data function
#################################################    


 #   mats=[30,90]   #omving average window periods for each data column to add to series table
 #   mat_types=["d"]  #,"d","m"]
  

#print("\nexcel input data filename='",filename,"'\n")
#print("excel query fields=",index_code)
#print("moving average days",mats)
#print("start point",start_point)
#print("predict ahead steps=",predict_ahead_steps,"\n")
required_starting_length=731+np.max(mats)+batch_length   # 2 years plus the MAT data lost at the start + batchlength


########################################################################    

#    load the sales_trans.xls file
# load the query.xls file
#    this contains all the products, product groups, customers, customer groups, glsets and special price categories you want to     

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

# pipeline



loaddata = __import__('SCBS_excel_import_v1-02')
batch = __import__('SCBS_batches_v1-00')
learn = __import__('SCBS_model_v1-00')
predict = __import__('SCBS_predict_v1-00')


from sklearn.pipeline import Pipeline

series_prediction_model = Pipeline([
        ("excel_load", loaddata.main()),
        ("batching", batch.main()),
        ("learn", learn.main()),
        ("predict", predict.main()),
    ])


sales_prediction_model.predict()

#exec(open("test2.py").read())



# series_prediction = Pipeline([
#         ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
#         ("lin_reg", LinearRegression()),
#     ])
        #     model = Pipeline([
        #             ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        #             ("std_scaler", StandardScaler()),
        #             ("regul_reg", model),
        #         ])
        # model.fit(X, y)
        # y_new_regul = model.predict(X_new)





