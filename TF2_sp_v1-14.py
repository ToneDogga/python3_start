# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:10:04 2020

@author: Anthony Paech 2016
"""


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# try:
#     # %tensorflow_version only exists in Colab.
#     %tensorflow_version 2.x
#     IS_COLAB = True
# except Exception:
#     IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

print("\n\nSales prediction tool using a neural network - By Anthony Paech 25/2/20")
print("=======================================================================")       

print("Python version:",sys.version)
print("\ntensorflow:",tf.__version__)
print("sklearn:",sklearn.__version__)

# Common imports
import numpy as np
import pandas as pd
import os
import random

print("numpy:",np.__version__)
print("pandas:",pd.__version__)


if not tf.test.is_gpu_available():
    print("\nNo GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
  #  if IS_COLAB:
  #      print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
else:
    print("\nSales prediction - GPU detected.")



# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

#n_steps = 50

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
    



   
def load_data(filename):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows


    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

    df["week"]=df.date.dt.to_period('W')
   
 #   mask = mask.replace('"','').strip()    
 #   print("mask=",mask)
    
 #   mask=((df['code']=='FLPAS') & (df['product']=='SJ300'))

  #  mask=(df['product']=='SJ300')
  #  mask=(df['code']=='FLPAS')
 #   mask=((df['code']=='FLPAS') & (df['productgroup']==10))
    mask=((df['code']=='FLPAS') & (df['productgroup']==10) & (df['product']=='SJ300'))

   # print("mask=",str(mask))
    print("pivot table being created.")
    table = pd.pivot_table(df[mask], values='qty', index=['code','productgroup','product'],columns=['week'], aggfunc=np.sum, margins=False,observed=False)   #, observed=True. fill_value=0)
    print("\ntable=\n",table)   #.head(5))#  #   f.write("\n\n"+table.to_string())
    print("table created.")
    
    product_names=list(table.index)   #"t"  #list(table[table["product"]])
    #print("product names=",product_names)
    sales=table.to_numpy()
 #   sales=sales[...,:n_steps]
    print("sales=\n",sales,sales.shape)        
    return sales[..., np.newaxis].astype(np.float32),product_names





def build_mini_batch_input(series,no_of_batches,no_of_steps):
    print("build mini batch series shape",series.shape)
    np.random.seed(42) 
    series_steps_size=series.shape[1]
    print("series steps size=",series_steps_size)

#  input series array is structured as a 2d array [[step values]]  shape eg (1,82)
# the numnber of mini batches should be the same length as the input layer number of neurons
# the output array is structured as a 3D array [no of mini-batches,steps,step values]
# each mini batch is a training series of say 20 steps long, with a random starting point somewhere on the batch size.
# if the starting point cannot go too close to the end of the series as it would
# overflow over the end.
# also how does it predict the first 20 steps?
          

    random_offsets=np.random.randint(0,series_steps_size,size=(no_of_batches)).tolist()
  #  print("random_offsets=",random_offsets)   #,random_offset.shape)
 #   single_batch=series[:,:no_of_steps]
    new_mini_batch=np.roll(series[:,:no_of_steps+1],random_offsets[0],axis=1).astype(int) 
    prediction=new_mini_batch[:,-1]
    for i in range(1,no_of_batches):
        temp=new_mini_batch[:,:no_of_steps]
        ptemp=prediction
        new_mini_batch=np.roll(series[:,:no_of_steps+1],random_offsets[i],axis=1)    #random.randint(0,no_of_steps,size=(no_of_batches,1,1)),axis=1)
        prediction=np.concatenate((ptemp,new_mini_batch[:,-1]))
        new_mini_batch=np.vstack((temp,new_mini_batch[:,:no_of_steps]))
        
        if i%100==0:
            print("\rBatch:",i,"new_mini_batch.shape:",new_mini_batch.shape,flush=True,end="\r")

        
 #   print("new_mini_batch=",new_mini_batch,new_mini_batch.shape)
 #   print("prediction=",prediction)
    print("\n")        
    return new_mini_batch[:,:no_of_steps],prediction 




def plot_learning_curves(title,epochs,loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, epochs, 0, 2])
    plt.legend(fontsize=11)
    plt.title(title,fontsize=11)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    return




#  How the batching works
#  The input array is three dimentional
# batch_size, n_steps, n_inputs
#
# batch size is the number of batches of each instance of the series
# in other words, it is the number of unique training series (1)
#
# n_steps is the number of time steps in each batch (104)
#
# n_inputs is the actual size of the data at each time step  (1)   

#n_steps = 50



def main():
#    print("\n\nSales prediction tool using a neural network - By Anthony Paech 25/2/20")
#    print("=======================================================================")       
 
 #   mask=((df["code"]=="FLPAS") & (df["product"]=="SJ300"))

   # n_steps = 100
    predict_ahead_steps=52
    epochs=70
    no_of_batches=10000   #1       # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point
    batch_length=20   #20
 #   no_of_batch_copies=4  # duplicate the batches
 #   max_batch_size=1
    
   # train validate test split
    train_percent=0.7
    validate_percent=0.2
    test_percent=0.1

########################################################################    

#    mask="(df['product']=='SJ300')"
    filename="NAT-raw310120all.xlsx"
    print("loading series....",filename) 
    series2,product_names=load_data(filename)    #,n_steps+1)  #,batch_size,n_steps,n_inputs)
    print("\nProduct names length=",len(product_names))
    
    
    
   #  series2=np.array([48,  16,  24,  56,   8,  16,  48,  64,  40,  16,  48,  24,  32,  24,  24,  24,  16,  96,
   #  8,  48,  32,  64,  16,  24,  64,  32,  56,  64,   8,  48,  16,  56,  48,  24,  32, 136,
   #  40,   8,  80,  40,  16,  24,  40,  16,  24,  40,  24,  24,  32,  40,  48,  32,  24,  40,
   #  32,  32,  56,  24,  32,  24,  24,  48,  16,  56,  48,  32,  32,  64,  88,  32,  24,   8,
   #  40,  48,  24,  32, 104,  24,  56,  40,  32, 56]).astype(np.float32).reshape(1,-1,1)
   #  product_names=["test"]
   # # print("series2=\n",series2.shape)
    
    n_steps=series2.shape[1]-1
    n_inputs=series2.shape[2]
    
    x_axis=np.arange(0,n_steps+1+predict_ahead_steps,1)

    print("file loaded shape=",series2.shape)  
   
    print("n_steps=",n_steps)
    print("n_inputs=",n_inputs)
    print("predict_ahead_steps=",predict_ahead_steps)
    print("epochs=",epochs)
    
    max_y=np.max(series2)

    print("max y=",max_y)
    
    
    np.random.seed(42)
    
    ##################################################
    
    #  Plot starting series
        
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.title("A Unit sales series: "+str(product_names),fontsize=14)

   #plt.title("A unit sales series", fontsize=14)
    plt.plot(x_axis[:n_steps+1], series2[0,:], "b-",label=r"$unit sales$")
  #  plt.plot(x_axis, series2[0,:], "b-",label=r"$unit sales$")

#   plt.plot(x_axis[:-1], sales[product_row_no,x_axis[:-1]], "b-", linewidth=3, label="A training instance")
    plt.legend(loc="best", fontsize=14)
    plt.axis([0, n_steps+predict_ahead_steps+1, 0, max_y+10])
    plt.xlabel("Week")
    plt.ylabel("UnitSales")
   
    plt.show()
   
    
    

   
   ###############################################
#   build model and train using random mini batches of 20 long
    print("Build model and train")
   
#   build_mini_batch_input(series,no_of_batches,no_of_steps)
    X,y=build_mini_batch_input(series2,no_of_batches,batch_length)
#   series=create_extra_batches2(series2,extra_batches,no_of_batch_copies)     # rotate the weeks forward in the batch by one week each time to maintain the integrity of the series, just change its starting point.  and then duplicate them 
    print("mini_batches X shape=",X.shape)  
    print("mini_batches y shape=",y.shape)  
   
    batch_size=X.shape[0]
    print("Batch size=",batch_size)

   
   
    train_size=int(round(batch_size*train_percent,0))
    validate_size=int(round(batch_size*validate_percent,0))
    test_size=int(round(batch_size*test_percent,0))
  
 #  print("train_size=",train_size)
 #  print("validate_size=",validate_size)
 #  print("test_size=",test_size)
   
    X_train, y_train = X[:train_size, :], y[:train_size]
    X_valid, y_valid = X[train_size:train_size+validate_size, :], y[train_size:train_size+validate_size]
    X_test, y_test = X[train_size+validate_size:, :], y[train_size+validate_size:]
    X_all, y_all = series2,series2[-1]
   
  # print("\npredict series shape",series.shape)
    print("X_train shape, y_train",X_train.shape, y_train.shape)
    print("X_valid shape, y_valid",X_valid.shape, y_valid.shape)
    print("X_test shape, y_test",X_test.shape, y_test.shape)
   
    
    answer=input("Recalculate a new model?")
    if answer=="y":

        
        
        #print("Make the second SimpleRNN layer return only the last output:")
    
        print("\nUsing a SimpleRNN layer model.")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        model = keras.models.Sequential([
            keras.layers.SimpleRNN(batch_length, return_sequences=True, input_shape=[None,1]),   #[none,1]
            keras.layers.SimpleRNN(20),
            keras.layers.Dense(1)
        ])
        
        model.compile(loss="mse", optimizer="adam")
        history = model.fit(X_train, y_train, epochs=epochs,
                            validation_data=(X_valid, y_valid))
        
     
        ##############3
        
        
        print("\nsave model\n")
        model.save("simple_sales_predict_model.h5")
        
        
        ################3
        
        
        model.summary()
        
   
##################
     
     #   model.evaluate(X_valid, y_valid)
        
     #   plot_learning_curves(history.history["loss"], history.history["val_loss"])
     #   plt.show()
        
        
    
    
    
    
        
    ################################################3                    
    #  test the model    
    
        print("testing the model")
    
    
            
     #   for product_row_no in range(0,n_rows):
            #plt.title("Testing the model", fontsize=14)
        plt.title("Testing the model: "+str(product_names),fontsize=14)
        
        #    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
        #    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
    #    plt.plot(x_axis[:-predict_ahead_steps], series2[0,:], "b*", markersize=10, label="instance")
      #  plt.plot(x_axis[1:-predict_ahead_steps], series2[0,1:], "w*", markersize=10, label="target")
        plt.plot(x_axis[:-predict_ahead_steps], series2[0,:], label=r"$unit sales$")
        
        y_pred = model.predict(X_valid)[0]     #series2[0,:].reshape(1,-1,1))
        print("X_valid y_pred=",y_pred,y_pred.shape)   #[:,-n_steps:,:]
    
        plt.plot(x_axis[-predict_ahead_steps], y_pred, "r.", markersize=10, label="prediction")
      #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
        
     
        
        plt.legend(loc="best")
        plt.xlabel("Week")
        
        plt.show()
         
        
        print("testing the model finished")


    else:    
#################################################################################    
    
        print("\nload model")
        model = keras.models.load_model("simple_sales_predict_model.h5")
 
    
#############################################    
    
    print("forecasting",predict_ahead_steps,"steps ahead.")
    
 #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    
    print("series2 shape=",series2.shape)
    n_steps=series2.shape[1]
    current_series=series2  #[np.newaxis,:]
 #   print("current series=\n",current_series[:10],current_series.shape)
    
 #   X = X_new
 #   n_steps=X.shape[1]
 #   print("X_shape=",X.shape,"n_steps=",n_steps)
 #   ys= model.predict(X[:,1:])[:, np.newaxis,:]
    ys= model.predict(current_series)    #[:, np.newaxis,:]
 #   print("ys shape=",ys.shape)
    ys=ys[:, np.newaxis,:]
 #   print("afetr ys shape=",ys.shape)

    Y_new=np.concatenate((series2,ys),axis=1)
 #   print("X_new",Y_new,Y_new.shape)
    for step_ahead in range(1,predict_ahead_steps):
 # #         print("step ahead=",step_ahead,"X[:,step_ahead:].shape=",X[:,step_ahead:].shape)
                    
         y_pred_one = model.predict(Y_new[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
    #     print("y_pred_one shape=",y_pred_one,y_pred_one.shape)
         y_pred_one=y_pred_one[:,np.newaxis,:]    
    #     print("after ypred one shape=",y_pred_one.shape)

 #      #   print("y_pred_one=",y_pred_one,y_pred_one.shape,"ypred=",y_pred_one.reshape(-1))
         Y_new = np.concatenate((Y_new, y_pred_one),axis=1)    #[:, np.newaxis,:]), axis=1)
  #       print("\rPredict step:",step_ahead,Y_new,"Y_new.shape:",Y_new.shape,flush=True,end="\r")

    
  #  print("\n\n")    
    print("Y new finished.shape=",Y_new, Y_new.shape)
   
   
################################################3                    
            
 #   for product_row_no in range(0,n_rows):
        #plt.title("Testing the model", fontsize=14)
    plt.title("Actual + Prediction: "+str(product_names),fontsize=14)
    
    #    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
    #    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
    plt.plot(x_axis[:n_steps], Y_new[0,:n_steps,0], "b-", markersize=5, label="actual")
    plt.plot(x_axis[n_steps-1:], Y_new[0,n_steps-1:,0], "r-", markersize=5, label="prediction")
    
       
  #  plt.plot(x_axis, Y_pred, "g.", markersize=10, label="prediction")
  #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
    
 
    
    plt.legend(loc="best")
    plt.xlabel("Week")
    
    plt.show()
     
    
    
       
    
       
#######################################################################################    
    
    print("\nload model")
    model = keras.models.load_model("simple_sales_predict_model.h5")
   
    
    print("validating existing sales using trained model",n_steps,".")
    
 #   X_new,Y_new=build_mini_batch_input(series2,no_of_batches,n_steps+1)
    
 #   print("Y_new shape=",Y_new.shape)
  #  n_steps=Y_new.shape[1]
 #   current_series=Y_new[:,batch_length:,:].reshape(1,-1,1)  #[np.newaxis,:]
  #  series2=Y_new.reshape(1,-1,1)  #[np.newaxis,:]

    print("series2=\n",series2.shape)
    
 #   X = X_new
 #   n_steps=X.shape[1]
 #   print("X_shape=",X.shape,"n_steps=",n_steps)
 #   ys= model.predict(X[:,1:])[:, np.newaxis,:]
    ys= model.predict(series2)    #[:, np.newaxis,:]
 #   print("ys shape=",ys.shape)
    ys=ys[:, np.newaxis,:]
 #   print("afetr ys shape=",ys.shape)

    X_new=np.concatenate((series2,ys),axis=1)
 #   print("X_new",X_new,X_new.shape)
    for step_ahead in range(1,n_steps):
 # #         print("step ahead=",step_ahead,"X[:,step_ahead:].shape=",X[:,step_ahead:].shape)
                    
         y_pred_one = model.predict(X_new[:,step_ahead:,:])   #X[:, new_step:])    #[:, np.newaxis,:]
    #     print("y_pred_one shape=",y_pred_one,y_pred_one.shape)
         y_pred_one=y_pred_one[:,np.newaxis,:]    
    #     print("after ypred one shape=",y_pred_one.shape)

 #      #   print("y_pred_one=",y_pred_one,y_pred_one.shape,"ypred=",y_pred_one.reshape(-1))
         X_new = np.concatenate((X_new, y_pred_one),axis=1)    #[:, np.newaxis,:]), axis=1)
  #       print("\rPredict step:",step_ahead,Y_new,"Y_new.shape:",Y_new.shape,flush=True,end="\r")

    
  #  print("\n\n")    
    print("X new finished.shape=",X_new, X_new.shape)
   
                   
            
 #   for product_row_no in range(0,n_rows):
        #plt.title("Testing the model", fontsize=14)
    plt.title("Validating the model: "+str(product_names),fontsize=14)
    plt.axis([0, n_steps+1, 0, max_y+10])

    #    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
    #    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
    plt.plot(x_axis[:n_steps], X_new[0,:n_steps,0], "b*", markersize=7, label="actual")
    plt.plot(x_axis[:n_steps], Y_new[0,:n_steps,0], "r.", markersize=5, label="prediction")
    
       
  #  plt.plot(x_axis, Y_pred, "g.", markersize=10, label="prediction")
  #  plt.plot(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
    
 
    
    plt.legend(loc="best")
    plt.xlabel("Week")
    
    plt.show()
   
#######################################    
    
   
    
    
    print("Validate model & learning curve")
    
    X,Y=build_mini_batch_input(Y_new,no_of_batches,Y_new.shape[1])
    print("X shape,Y_shape",X.shape,Y.shape)
    print("n_steps=",n_steps)
    new_steps=X.shape[1]

    
    X_train, Y_train = X[:train_size, -new_steps:], Y[:train_size,-new_steps:]
    X_valid, Y_valid = X[train_size:train_size+validate_size, -new_steps:], Y[train_size:train_size+validate_size,-new_steps:]
    X_test, Y_test = X[train_size+validate_size:, -new_steps:], Y[train_size+validate_size:,-new_steps:]
    

   # print("\npredict series shape",series.shape)
    print("X_train shape, Y_train",X_train.shape, Y_train.shape)
    print("X_valid shape, Y_valid",X_valid.shape, Y_valid.shape)
    print("X_test shape, Y_test",X_test.shape, Y_test.shape)

    
    
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[new_steps, 1]),   # 50
        keras.layers.Dense(1)     # 10
    ])
    
    model.compile(loss="mse", optimizer="adam")
    history = model.fit(X_train, Y_train, epochs=epochs,
                        validation_data=(X_valid, Y_valid))  
    
    
    model.evaluate(X_valid, Y_valid)
    
    plot_learning_curves("Learning curve:"+str(product_names),epochs,history.history["loss"], history.history["val_loss"])
    plt.show()
    
    print("\n\nFinished.")
    
    
    return





if __name__ == '__main__':
    main()


    
          
          
          
