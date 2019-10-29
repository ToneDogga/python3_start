import numpy as np
import pandas as pd
import sys

def read_csv_file_in(filename):
    environment=pd.read_csv(filename, names=["condition","message"], header=0, dtype={"condition":np.str,"message":np.str})
   # print(environment)
  #  input("?")
    return(environment)




#pop=pd.dataframe(condition:["0010#010"])
#pop = pd.DataFrame({'a':range(3),'condition': "0010#010", 'translated': ""})
#string=pd.Series.str.maketrans(condition,"01010101")
#pop2=pop.transform([np.sqrt, np.exp])
#print(pop)
import_file=sys.argv[1]
df=read_csv_file_in(import_file)
clen =10

#df = pd.DataFrame({'col1': {0: "101", 1: "1#0", 2: "##0"}, 'col2': {0: "000", 1: "1#1", 2: "000"}})
print(df)
#di = {"##0" : "110"}    #, 1: "0", 0:"0"}

#print(df2)
#df3=df['col1'].map(di)
#print(df3)
#df["col1"].replace(di, inplace=True)
#print(df)
#df4=df[df['col1'][0].str.contains("#")]
#print(df4)
#df5=df[0].replace({"col1": di})
#print(df5)

for c in range(clen-1,0,-1):
    df["new"] = df.condition.str[c]
#print(col1)
#colno=1
#col2 = df.condition.str[colno]
print(df)
