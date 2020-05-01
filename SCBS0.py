#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:58:04 2020

@author: tonedogga
"""


import numpy as np
from datetime import datetime
import os


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./SCBS_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)




class c(object):
    pass
  
c.predict_ahead_steps=44

 #   epochs_cnn=1
c.epochs_wavenet=5
c.no_of_batches=80000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
c.batch_length=16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
#    y_length=1
c.neurons=1000  #1000-2000
c.dropout_rate=0.4

c.pred_error_sample_size=100

c.patience=5   #5
   
c.mats=[28]   #omving average window periods for each data column to add to series table
c.start_point=np.max(c.mats)+c.batch_length  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
c.mat_types=["u"]  #,"d","m"]
   
c.units_per_ctn=8
   
# train validate test split 
c.train_percent=0.7
c.validate_percent=0.2
c.test_percent=0.1

c.filenames=["allsalestrans2020.xlsx","allsalestrans2018.xlsx"]

c.queryfilename="queryfile.xlsx"


c.required_starting_length=(365*2)+np.max(c.mats)+c.batch_length   # 2 years plus the MAT data lost at the start + batchlength




########################################################################    

c.output_dir = log_dir("SCBS")
os.makedirs(c.output_dir, exist_ok=True)


c.images_path = os.path.join(c.output_dir, "images/")
os.makedirs(c.images_path, exist_ok=True)
    



#file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#m, n = X_train.shape


#checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
#checkpoint_epoch_path = checkpoint_path + ".epoch"
#final_model_path = "./my_deep_mnist_model"

#########################################################



#    load the sales_trans.xls file
# load the query.xls file
#    this contains all the products, product groups, customers, customer groups, glsets and special price categories you want to     

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors

# pipeline


import SCBS1_excel_import_v1_02 as scbs1 
import SCBS2_batches_v1_00 as scbs2 
import SCBS3_model_v1_00 as scbs3 
import SCBS4_predict_v1_00 as scbs4 

scbs1.main(c)
scbs2.main(c)
scbs3.main(c)
scbs4.main(c)



# xloaddata = __import__('SCBS1_excel_import_v1-02')
# xbatch = __import__('SCBS2_batches_v1-00')
# xlearn = __import__('SCBS3_model_v1-00')
# xpredict = __import__('SCBS4_predict_v1-00')


# from sklearn.pipeline import Pipeline

# series_prediction_model = Pipeline([
#          ("excel_load", scbs1),
#          ("batching", scbs2),
#          ("learn", scbs3),
#          ("predict", scbs4),
#      ])


# sales_prediction_model.predict([])

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





