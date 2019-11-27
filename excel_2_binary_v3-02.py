#!/usr/bin/python3
from __future__ import print_function
from __future__ import division

from sklearn import preprocessing
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
import joblib
import pickle
import queue as queue

import SGC_cfg   # import config settings for excel import
from datetime import datetime, timedelta


#lb = preprocessing.LabelBinarizer()




def save_model(model,filename):
##    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
##    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
##    dataframe = pandas.read_csv(url, names=names)
##    array = dataframe.values
##    X = array[:,0:8]
##    Y = array[:,8]
##    test_size = 0.33
##    seed = 7
##    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
##    # Fit the model on training set
##    model = LogisticRegression()
##    model.fit(X_train, Y_train)
    # save the model to disk
    #filename = 'finalized_model.sav'
    joblib.dump(model, filename)
    return 

def load_model(filename):
    # some time later...

    # load the model from disk
    loaded_model = joblib.load(filename)
 #   result = loaded_model.score(X_test, Y_test)
 #   print(result)
    return loaded_model

def discrete_ybins(array,bins):
    # divide the classes (or features) into a bucket number of equally sized ranges
    return pd.cut(array.astype(float),bins,labels=range(len(bins)-1),right=False,retbins=False)    #.astype(int)
   
def discrete_Xbins(array,bins,df,excludecols):
    # divide the 2D X features into bins described in the 2D array bins. number of equally sized ranges
    binlen=len(bins)
    colname=df.columns.tolist()
 #   print("colname=",colname)
   # Xbins=[]   #np.empty([1,1],dtype=str)
    for i in range(0,binlen):
  #      print("calc Xbins i",i,"colname=",colname[i])
        if len(bins[i])>0 and i not in excludecols:
            df.insert(i,colname[i]+"_bin",pd.cut(array[i].astype(float),bins[i],labels=range(len(bins[i])-1),right=False,retbins=False).astype(str))
      #      print("b=\n",df.head(10))
            df.drop(columns=colname[i],axis=1,inplace=True)
    #        print("after=\n",df.head(10))
        #else:
         #   Xbins.append(np.empty([0]))    
    return df   


def read_excel(filename,rows):
    xls = pd.ExcelFile(filename)    #'salestransslice1.xlsx')
    return xls.parse(xls.sheet_names[0]).head(rows)
    #xd=df.to_dict()
    print(df)
  #  return df


def write_excel(df,outfilename):
    df.to_excel(outfilename)    #'salestransslice1.xlsx')
    return
 

def get_classes(df,ycol):
    return df.iloc[:,ycol].values.astype(str)

def get_features(df):
    return df.values.T.astype(str)


def day_delta(df):

    df['day_delta'] = (df.date.max()-df.date).dt.days.astype(int)
#    df['day_delta'] = (df.date-df.date.min()).dt.days
    df.drop(columns='date',axis=1,inplace=True)
 #   df.drop(df['date'],inplace=True)
##    date_N_days_ago = datetime.now() - timedelta(days=2)
##
##    print(datetime.now())
##    print(date_N_days_ago)
    return df

def encode(df,excludecols,ycolno,ybins,q):
    #
    # copy the y col to the end of the df


    colnames=df.columns

    df["ycol"]=df.iloc[:,ycolno].values

  #  print(" starting df=",df)
    
    X_coldict=defaultdict(int)
    y_coldict=defaultdict(int)
    #f=open("savedata.dat","wb")
    #f.close()
   # f=open("savedata.dat","wb")
 #   print(colnames)
    i=0
    j=0
    s=[]

    # encode the lot, include the y column as part of X so that we don't lose consistency
    for col in iter(colnames):
        if i in excludecols:  # can't exclude the 0 column
            pass
        else:
            lb = preprocessing.LabelBinarizer()

            #prodb=df[col].values.astype(str)
            prodb=df[col].astype(str).tolist()    #.astype(str)

          #  print("col=",col,"prodb=",prodb)
            bbin=lb.fit_transform(prodb)

            q.put(pickle.dumps(lb))

            del lb
             
            if i==0:
                start=0
                ef=bbin
            else:
                start=finish
                ef=np.hstack((ef,bbin))

                
            finish=ef.shape[1]
            colwidth=finish-start

            X_coldict[i]={colnames[i]:colwidth}
        i+=1

########  add y column 
    lb = preprocessing.LabelBinarizer()

  #  print("columns found=",i,"processed=",X_coldict)
   # ybins=df["message"].values.astype(str)
    y_encode=lb.fit_transform(ybins)

    q.put(pickle.dumps(lb))
    y_coldict[0]={colnames[ycolno]:y_encode.shape[1]}
    
###################
    #   
    return ef,X_coldict,y_encode,y_coldict,j,q


def create_df(npbin,yclass):
    j=pd.DataFrame(columns=["condition","message"])
    j["condition"]=list(map(''.join, npbin.astype(str)))
    j["message"]=list(map(''.join, yclass.astype(str)))
    return j


def decode(gf,X_column_dict,excludecols,ycolno,ybins,q):
        
    condition_len=gf["condition"].str.len()[1]

    message=gf["message"]
    gf=gf.condition.str.extractall('(.)')[0].unstack().rename_axis(None)
 #   print(gf)
    key_list=[]
    values=0
    old=0
    
    colcount=0
    for x in X_column_dict:
          
        mylist = list(X_column_dict.values())
        keys = list(mylist[colcount].keys())[0]
        key_list.append(keys)
        old=old+values
        values = list(mylist[colcount].values())[0]
        
        temp=gf[old]
        for z in range(1,values):
    #        print("z=",z,"temp=",temp)
            temp=temp + gf[old+z]
        gf[keys]=temp
    #    print("keys=",keys,"values=",values,"gf[keys]=",gf[keys])
        colcount+=1

  #  gf["message"]=message
    gf = gf.drop(gf.columns[range(0,condition_len)], axis=1)
  #  print("final gf=\n",gf)
  #  print(gf.shape) 
  #  print("key_list=",key_list)
  #  print("excluded cols=",excludecols)
  #  print("ycolno=",ycolno)
  #  f=open("savefile.dat","rb")
    i=0
    for x in X_column_dict:
   #     if i==ycolno:
   #         pass
   #     else:  
        splitcol=gf.iloc[:,i].str.extractall('(.)')[0].unstack().rename_axis(None).to_numpy().astype(int)
     #   print("\n",splitcol)
        s=q.get()
        lb=pickle.loads(s)
        gf[key_list[i]]=lb.inverse_transform(splitcol)
   #     print("decoded=",i,key_list[i],splitcol)
        i+=1


######### decode y column
 #   lb = preprocessing.LabelBinarizer()

    splitcol=message.str.extractall('(.)')[0].unstack().rename_axis(None).to_numpy().astype(int)
    s=q.get()
    lb=pickle.loads(s)
    gf["y_bins"]=lb.inverse_transform(splitcol)    #df[col].values.astype(s
        
##    print(myInnerList1)
##    print(myInnerList11)
##
##    myInnerList2 = list(mylist[1].values())
##    myInnerList21 = list(mylist[1].keys())
##
##    print(myInnerList2)
##    print(myInnerList21)
##

##############################

    return gf



def main():
    
    #mlb = MultiLabelBinarizer()
    #lb = preprocessing.LabelBinarizer()


    q = queue.Queue()

##    for i in range(5):
##        q.put(i)
##
##    while not q.empty():
##        print(q.get())

    print("\n\n\n\n\n\nSimple Genetic Classifier encoder")
    print("y column no=",SGC_cfg.ycolno)

    ##loop=True
    ##while loop:
    ##    ycolno=int(input("Enter column number for classes output:"))
    ##    loop=ycolno in excludecols
    ##    if loop:
    ##        print("column number is currently excluded.  Excluded=",excludecols)
    #int(input("Enter number of different bins for the class column:"))

    pd.set_option('max_colwidth', 255)
    #pd.options.display.width = 0
    pd.options.display.max_rows = 999
    df=read_excel(SGC_cfg.filename,SGC_cfg.importrows)
    print(SGC_cfg.filename,"(",SGC_cfg.importrows,") rows -> column names:\n",df.columns)
   # write_excel(df,"testout2.xlsx")
 
 #   print(df)
  #  print("Excluded X column numbers:",SGC_cfg.excludecols)



    yarray=get_classes(df,SGC_cfg.ycolno)

    # drop the ycol from the feature set
    df.drop(columns=df.columns[SGC_cfg.ycolno],axis=1,inplace=True)

 #   print("y gone=\n",df.head(10))
    print("yarray=",yarray)
    ybins=discrete_ybins(yarray,SGC_cfg.ybins)
    ybins=np.nan_to_num(ybins)   #,nan=0.0,posinf=-1.0,neginf=-1.0)
    print("ybins=",ybins.tolist())


    if SGC_cfg.day_deltas:    # copnvert a date and a day change count
        df=day_delta(df)
      #  SGC_cfg.ycolno-=1  # removed the date column

   # print(df)
    Xarray=get_features(df)
  #  print("Xarray=",Xarray)
  #  print("Xarray shape",Xarray.shape)    #.tolist())
    df=discrete_Xbins(Xarray,SGC_cfg.Xbins,df,SGC_cfg.excludecols)
    df=df.fillna(0)   #,nan=0.0,posinf=-1.0,neginf=-1.0)

   # print("Xbin ranges",SGC_cfg.Xbins)
   # print("X bins=\n",Xbins)
    print(df.tail(100))
   #
   # add column y to the end of the feature array
   #
    X_encode,X_column_dict,y_encode,y_column_dict,column_count,q=encode(df,SGC_cfg.excludecols,SGC_cfg.ycolno,ybins,q)

  #  print("X encode shape=",X_encode.shape)
   # print("X encode=\n",X_encode)
    print("X column dict=\n",X_column_dict)
   # print("y encode=\n",y_encode)
  #  print("y column dict =\n",y_column_dict)
    gf=create_df(X_encode,y_encode)
    print("\nconditions dataframe head 10=\n",gf.head(10).to_string())
 #   print("\nconditions dataframe=\n",gf.to_string())

    print("final df shape=",gf.shape)
 #   print("y_encode=",y_encode)
    condition_len=gf["condition"].str.len()[1]

    print("condition len=",condition_len)
    print("message len=",gf["message"].str.len()[1])

    gf.to_csv("encoded.csv", index=False)   #sep='\t',
#######################


    hf=decode(gf,X_column_dict,SGC_cfg.excludecols,SGC_cfg.ycolno,ybins,q)
########################
 #   print(hf)
    write_excel(hf,"outfilename.xlsx")
    
       # print("xcol=",xcol,x2col,x3col)
    return





    #ff=decode(bin_encode,column_dict)
    #print("ff shape=",ff.shape)



#d=d.T
#print("d.T=\n",d)
#array = np.fromiter(d.items(), dtype=int, count=len(d))
#print("array=\n",array)
    #print("feature names:",mlb.classes_)
    ##prodbinstr=np.column_stack((sprod,np.repeat(['\n'], prodbin.shape[0])[:,None])).tostring()
    ##print("pbs=",prodbinstr)
    ##print("len=",len(prodbinstr))
     
  #  newproducts=lb.inverse_transform(prodbin)
    #newp=pd.factorize(newproducts)[0]
  #  print("new prodcts=\n",newproducts)
    #print("classes:",mlb.classes_)
    #print("params:",lb.get_params())


##mlb.fit_transform([(1, 2), (3,)])
##
##
##print(mlb.classes_)
##
##print(mlb.fit_transform([{'sci-fi', 'thriller'}, {'comedy'}]))
##
##print(mlb.classes_)
##
##print(list(mlb.classes_))
##



if __name__ == '__main__':
    main()


