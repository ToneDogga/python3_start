from sklearn import preprocessing
import numpy as np
import pandas as pd
#import xlrd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict

mlb = MultiLabelBinarizer()
lb = preprocessing.LabelBinarizer()

def read_excel(filename,rows):
    xls = pd.ExcelFile(filename)    #'salestransslice1.xlsx')
    df = xls.parse(xls.sheet_names[0]).head(rows)
    #xd=df.to_dict()
    print(df)
    return df


def encode(df):
    colnames=df.columns
    coldict=defaultdict(int)

 #   print(colnames)
    i=0
    for col in iter(colnames):
        prodb=df[col].values.astype(str)
        bbin=lb.fit_transform(prodb)
        if i==0:
            start=0
            ef=bbin
        else:
            start=finish
            ef=np.hstack((ef,bbin))

        finish=ef.shape[1]
        colwidth=finish-start
        coldict[i]={colnames[i]:colwidth}
       # print("colwidth=",colwidth,"i=",i,"col=",col,"ef=\n",ef)
        print(coldict[i],"=\n",prodb,"binary=",bbin)

        i+=1
    return ef,coldict


def create_conditions(npbin):
  #  length=npbin.shape[1]
  #  npbin=npbin.astype(str)
   # print("npbin=",npbin)
 #   k=np.array(list(map(''.join, npbin.astype(str))))
    j=pd.DataFrame(columns=["condition","message"])
    j["condition"]=list(map(''.join, npbin.astype(str)))
    return j



def decode(df,coldict):
   # colnames=coldict[1][0]
   # print("keys=",coldict.keys())
   # print("values=",coldict.values())
   # print("items=",coldict.items())

    values=list(coldict.values())
    keys=list(coldict.keys())
    

    #col_len=len(coldict.keys())
    #print("col len",col_len)
    #colnames=list(coldict.values())
  #  print(colnames)

    print("values",values,"keys",keys)
    #print("colnames:",colnames[2][1])

    i=0
    for col in iter(coldict):
        print(coldict.get[col])
        
##        prodb=df[col].values.astype(str)
##        print(colnames[i],"=\n",prodb)
##        if i==0:
##            start=0
##            ef=lb.fit_transform(prodb)
##        else:
##            start=finish
##            newproducts=lb.inverse_transform(prodbin)
##            #ef=np.hstack((ef,lb.fit_transform(prodb)))
##
##        finish=ef.shape[1]
##        colwidth=finish-start
##        coldict[i]={colnames[i]:colwidth}
##        print("colwidth=",colwidth,"i=",i,"col=",col,"ef=\n",ef)
##        i+=1
        #print("pb=\n",prodbin,"\nstr=\n",sprod)
       # print("prodbin.shape=",prodbin.shape)
       # print("pb=\n",prodbin)


    print("ef=\n",ef)
    print("coldict=",coldict)
    return ff

pd.set_option('max_colwidth', 100)
#pd.options.display.width = 0
pd.options.display.max_rows = 999
df=read_excel("salestransslice1.xlsx",5)
bin_encode,column_dict=encode(df)
print("bin encode shape=",bin_encode.shape)
print("bin encode=\n",bin_encode)
print("coldict=\n",column_dict)
gf=create_conditions(bin_encode)
print("conditions dataframe=\n",gf.to_string())
print("shape=",gf.shape)
ff=decode(bin_encode,column_dict)
print("ff shape=",ff.shape)



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

