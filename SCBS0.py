#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:58:04 2020

@author: tonedogga
"""


# Common imports
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
#import random
import datetime as dt
import gc
import sys
from numba import cuda
""

from collections import defaultdict
from datetime import datetime
#import SCBS0 as c

import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

 

#filename="tables_dict.pkl"


import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow import keras
assert tf.__version__ >= "2.0"

print("\n\nUse pretrained models to make predictions - By Anthony Paech 30/4/20")
print("========================================================================\n")       

print("Python version:",sys.version)
print("\ntensorflow:",tf.__version__)
print("keras:",keras.__version__)
print("sklearn:",sklearn.__version__)


import os
import random
import csv
import joblib
import pickle
from natsort import natsorted
from pickle import dump,load
import datetime as dt
from datetime import date
from datetime import timedelta
import gc
from numba import cuda



#from sklearn.preprocessing import StandardScaler,MinMaxScaler

#import itertools
#from natsort import natsorted
#import import_constants as ic

print("numpy:",np.__version__)
print("pandas:",pd.__version__)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print("matplotlib:",mpl.__version__)

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import multiprocessing
print("\nnumber of cpus : ", multiprocessing.cpu_count())


visible_devices = tf.config.get_visible_devices('GPU') 

print("tf.config.get_visible_devices('GPU'):",visible_devices)
# answer=input("Use GPU?")
# if answer =="n":

#     try: 
#       # Disable all GPUS 
#       tf.config.set_visible_devices([], 'GPU') 
#       visible_devices = tf.config.get_visible_devices() 
#       for device in visible_devices: 
#         assert device.device_type != 'GPU' 
#     except: 
#       # Invalid device or cannot modify virtual devices once initialized. 
#       pass 
    
#     #tf.config.set_visible_devices([], 'GPU') 
    
#     print("GPUs disabled")
    
# else:
tf.config.set_visible_devices(visible_devices, 'GPU') 
print("GPUs enabled")
   
    

# if not tf.config.get_visible_devices('GPU'):
# #if not tf.test.is_gpu_available():
#     print("\nNo GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
#   #  if IS_COLAB:
#   #      print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
# else:
#     print("\nSales prediction - GPU detected.")


#print("tf.config.get_visible_devices('GPU'):",tf.config.get_visible_devices('GPU'))


# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)


# import numpy as np
# import pandas as pd
# from datetime import datetime
# import os
# import gc


# import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)




# from tensorflow import keras
# assert tf.__version__ >= "2.0"

# #if not tf.config.list_physical_devices('GPU'):
# #    print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
# #    if IS_COLAB:
# #        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")


# # Disable all GPUS 
# #tf.config.set_visible_devices([], 'GPU') 



#  #visible_devices = tf.config.get_visible_devices() 
# # for device in visible_devices: 
# #     print(device)
# #     assert device.device_type != 'GPU' 

# #tf.config.set_visible_devices([], 'GPU') 
# #tf.config.set_visible_devices(visible_devices, 'GPU') 


# # Common imports
# import numpy as np
# import os
# from pathlib import Path
# import pandas as pd
# import pickle
# #import random
# import datetime as dt
# from collections import defaultdict
# import gc


  
 
# # to make this notebook's output stable across runs
# np.random.seed(42)
# tf.random.set_seed(42)

# # To plot pretty figures
# #%matplotlib inline
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)


# print("Python version:",sys.version)
# print("\ntensorflow:",tf.__version__)
# print("keras:",keras.__version__)
# print("sklearn:",sklearn.__version__)
# #print("cuda:",numba.cuda.__version__)




visible_devices = tf.config.get_visible_devices('GPU') 
print("tf.config.get_visible_devices('GPU'):",visible_devices)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors



# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
    
  



def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "./SCBS_outputs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)



def main():
    print("\n\nstart module start\n\n")    
    
    
    class c(object):
        pass
      
    #c.predict_ahead_steps=440
    
     #   epochs_cnn=1
    c.epochs=12
    c.no_of_batches=15000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    #c.batch_length=16   #16 # 16  # one week=5 days   #4   #731   #731  #365  3 years of days  1096
    #    y_length=1
    c.neurons=1000  #1000-2000
    c.dropout_rate=0.2  
    # patience=10
    # epochs=10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    c.start_point=32
    # no_of_batches=8000
    c.end_point=800
    # # predict aherad length is inside batch_length
    c.predict_ahead_length=365
    c.batch_length=365    #20+c.predict_ahead_length
    # #batch_length=(end_point-start_point)+predict_ahead_length
   # c.X_window_length=c.batch_length-c.predict_ahead_length
    
        
    c.date_len=1300
       
    c.dates = pd.period_range("02/02/18", periods=c.date_len)   # 2000 days
    
    # #future_steps=400
    # #blank_future_days=365
    # # batch_total=100000
    # train_percent=0.7
    # validate_percent=0.2
    # test_percent=0.1
    
    #
    c.pred_error_sample_size=50
    
    c.patience=5   #5
       
    c.mats=[28]   #omving average window periods for each data column to add to series table
    #c.start_point=np.max(c.mats)+c.batch_length  # we need to have exactly a multiple of 365 days on the start point to get the zseasonality right  #batch_length+1   #np.max(mats) #+1
    c.mat_types=["u"]  #,"d","m"]
       
    c.units_per_ctn=8
       
    # train validate test split 
    c.train_percent=0.7
    c.validate_percent=0.2
    c.test_percent=0.1
    
  #  c.filenames=["allsalestrans2020.xlsx","allsalestrans2018.xlsx"]
    c.filenames=["allsalestrans190520.xlsx","allsalestrans2018.xlsx"]
    
 

    
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
    
    
    import SCBS1_excel_import_v2_00 as excelimport 
    import SCBS2_batches_v2_00 as batchup
    import SCBS3_model_v4_01 as trainmodel
    import SCBS4_predict_v4_01 as predict 
    
    excelimport.main(c)
    batchup.main(c)
    trainmodel.main(c)
    predict.main(c)
    
    print("\n\nstart module finished\n\n")
    tf.keras.backend.clear_session()
    #cuda.select_device(0)
    #cuda.close()
            
 
    gc.collect()
    return


if __name__ == '__main__':
    main()

  

    

    
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
    




