import scipy
import math
from scipy.spatial import distance
import numpy as np


def build_distance_lookup(xsize,ysize):
    x = np.arange(0.0, xsize, 1)
    y = np.arange(0.0, ysize, 1)
    a = np.array([(0.0,0.0)])
    b = np.array([(i,j) for j in y for i in x])
#    dist = scipy.spatial.distance.cdist(a,b,'euclidean')                              
    dist = distance.cdist(a,b,'euclidean')                              

    return(np.reshape(dist, (xsize, ysize),order='F'))
 
    

xsize=900
ysize=900
dist=build_distance_lookup(xsize,ysize)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
    #print("\ndist=",dist2)


while True:
    findx=int(input("x?"))
    findy=int(input("y?"))

    print(dist[findy][findx])




