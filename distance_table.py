import scipy
import math
from scipy.spatial import distance
import numpy as np
import pandas as pd
import datetime
import random

def build_distance_lookup(ysize,xsize):
    x = np.arange(0.0, xsize, 1)
    y = np.arange(0.0, ysize, 1)
    a = np.array([(0.0,0.0)])
    b = np.array([(j,i) for j in y for i in x])
#    dist = scipy.spatial.distance.cdist(a,b,'euclidean')                              
    dist = distance.cdist(a,b,'euclidean')                              

    return(np.reshape(dist, (ysize, xsize),order='F'))
 

def dd(x,y):
    return(distance.cdist([(0,0)],[(x,y)],'euclidean')[0][0])                              
    
    

xsize=10
ysize=100
dist=build_distance_lookup(ysize,xsize)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print("\ndist=",dist)
#dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1]})
dataframe = pd.DataFrame.from_records(dist)
print("pd=",dataframe)

##starttime=str(datetime.datetime.now())
##
##for c in range(0,100000):
##   # findx=int(input("x?"))
##   # findy=int(input("y?"))
##    findx=random.randint(0,xsize-1)
##    findy=random.randint(0,ysize-1)
##    print("c=",c,dataframe.loc[findy,findx])
##
##print("\n\npandas algorithm. Started at: "+starttime+"\n\n\n")
##print("Finished at:"+str(datetime.datetime.now())+"\n\n")
##
##
##input("?")
##
##starttime=str(datetime.datetime.now())
##
##for c in range(0,100000):
##   # findx=int(input("x?"))
##   # findy=int(input("y?"))
##    findx=random.randint(0,xsize-1)
##    findy=random.randint(0,ysize-1)
##    print("c=",c,dist[findy][findx])
##
##print("\n\nnumpy algorithm. Started at: "+starttime+"\n\n\n")
##print("numpy finished at:"+str(datetime.datetime.now())+"\n\n")
##
##input("?")
##starttime=str(datetime.datetime.now())
##
##for c in range(0,100000):
##   # findx=int(input("x?"))
##   # findy=int(input("y?"))
##    findx=random.randint(0,xsize-1)
##    findy=random.randint(0,ysize-1)
##    print("c=",c,math.sqrt(findy*findy+findx*findx))
##
##print("\n\ncalc algorithm. Started at: "+starttime+"\n\n\n")
##print("calc finished at:"+str(datetime.datetime.now())+"\n\n")

##input("?")
##starttime=str(datetime.datetime.now())
##
##for c in range(0,100000):
##   # findx=int(input("x?"))
##   # findy=int(input("y?"))
##    findx=random.randint(0,xsize-1)
##    findy=random.randint(0,ysize-1)
##    print("c=",c,distance.cdist([(0,0)],[(findx,findy)],'euclidean')[0][0])
##   # print("c=",c,math.sqrt(findy*findy+findx*findx))
##
##print("\n\nscipy algorithm. Started at: "+starttime+"\n\n\n")
##print("scipy finished at:"+str(datetime.datetime.now())+"\n\n")

input("?")

find=[]
for c in range(0,100000):
   # findx=int(input("x?"))
   # findy=int(input("y?"))
    findx=random.randint(0,xsize-1)
    findy=random.randint(0,ysize-1)
  #  print("c=",c,findx,findy)
    find.append((findx,findy))
starttime=str(datetime.datetime.now())    
d=distance.cdist([(0,0)],find,'euclidean')[0]
for c in range(0,100000):
    print("c=",c,d[c])
   # findx=int(input("x?"))

print(len(d))
print("d=",d)
   # print("c=",c,math.sqrt(findy*findy+findx*findx))

print("\n\nvector scipy algorithm. Started at: "+starttime+"\n\n\n")
print("vector scipy finished at:"+str(datetime.datetime.now())+"\n\n")

