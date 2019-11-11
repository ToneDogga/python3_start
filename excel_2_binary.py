from sklearn import preprocessing
import numpy as np
import pandas as pd
#import xlrd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict

#d = defaultdict(LabelEncoder)
#d = defaultdict(LabelEncoder)

mlb = MultiLabelBinarizer()

lb = preprocessing.LabelBinarizer()
#lb.fit([1, 2, 6, 4, 2])

##print("1",lb.classes_)
##
##print("2",lb.transform([1, 4, 2, 6]))
##
##print("2a",lb.fit_transform(['yes', 'no', 'no', 'yes']))
##
##
##lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
##
##print("3",lb.classes_)
##
##
##print("4",lb.transform([0, 1, 2, 1]))
##
##print("5",lb.fit_transform(['SJ300',"aj300","SPJ300"]))
##
##print("6",lb.transform(["rj300",'SJ300',"WS300"]))

#df=pd.read_excel("testdata.xlsx", "Sheet1",dtype={"cat":np.int32,"productgroup":np.int32,"qty":np.int32},convert_float=True).head(int(sys.argv[2]))
#df=pd.read_excel("testdata.xlsx",convert_float=True)


xls = pd.ExcelFile('salestransslice1.xlsx')
df = xls.parse(xls.sheet_names[0])[0:5]
#xd=df.to_dict()
print(df)
colnames=df.columns
ef=np.array(dtype=str)
#ef=pd.DataFrame(columns=colnames)
print("ef=\n",ef)

print(colnames)
#df.iloc[0]=colnames
i=0
for col in iter(colnames):
    #df2=df[col]
    #df=df["code"]
    #print("i=",i,"col=",col,"df=\n",df)
    #products=["SJ300","RJ300","10","5","6"]
    #print("start products=",products)
    prodb=df[col].values.astype(str)
    print(colnames[i],"=\n",prodb)
    #prodbin=mlb.fit_transform(prodb)   #products)
    #dtype=dict(names=[col,formats=str)
    ef=hstack(ef,lb.fit_transform(prodb))
    i+=1
    #print("pb=\n",prodbin,"\nstr=\n",sprod)
   # print("prodbin.shape=",prodbin.shape)
   # print("pb=\n",prodbin)


print("ef=\n",ef)
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

