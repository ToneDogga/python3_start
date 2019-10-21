import numpy as np

def vall_distance(stops,pdata):
    # vectorized version for speed
  # newpdata = pdata[..., np.newaxis]
   # newpdata = pdata[stops, np.newaxis]
    newpdata = pdata[stops]
   # print("\nnewpdata=\n",newpdata)
 
    stoplist=np.array(newpdata[...,0],dtype=int)
   # print("stoplist=",stoplist)   #,"stoplist2=",stoplist2,"stoplist3",stoplist3)
    destinations=np.roll(stoplist, -1, axis=1)
   # print("destinations=",destinations)
    stopdist=pdata[stoplist]
    distances=stopdist[...,2:]

  #  print("\ndistances=\n",distances)
    d=distances[[0],[stoplist],[destinations]]
  #  print("\nd=\n",d)
    nps=np.sum(d,axis=2)
  #  print("nps=",nps)
          
##    print("\nindexed distances=\n",distances[[stoplist],[destinations]])
    return(nps[0,...])





def vpath_distance(pdata):
    # vectorized version for speed   
    stoplist=np.array(pdata[...,0],dtype=int)
    nextlist=np.array(stoplist[1:],dtype=int)
    destinations=np.append(nextlist,[stoplist[0]])
  #  print("stoplist=",stoplist)
  #  print("destinations=",destinations)
  #  print("\npdata=\n",pdata)
    stopdist=pdata[stoplist]
    distances=stopdist[...,2:]
  #  print("\ndistances=\n",distances)
    d=distances[[stoplist],[destinations]]
   # print("\nindexed distances=\n",d)

  #  return(np.sum(distances[[stoplist],[destinations]]))
    return(np.sum(d))

    

stops=np.array([[0,1,2],[1,0,2],[2,1,0],[1,2,0],[0,2,1]])
#stops=np.array([[0,1,2],[1,0,2]])

data=np.array([[0,(10,20),0,3,2],[1,(11,30),4,0,7],[2,(22,4),5,6,0]])

test=vpath_distance(data)
print("distance=",test)
print("\n\n\n")
test2=vall_distance(stops,data)
print("all distance=",test2)
