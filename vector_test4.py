import numpy as np


#
# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the wheel m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.
#
# crossover - PMX partially matched crossover
# the new generation of strings is mated at random
# this means each pair of strings is crossed over at a uniform point in the string
# find two random split points in the string a swap the remaining information over between the mating pairs
#
# mutation
# this is the occasional and small chance SB (1/1000) that an element of a journey list is swapped randomly.
#
# go back to start with the new generation



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
    finalsums=nps[0,...]
    print("distances=",finalsums)
    xt = np.true_divide(finalsums, finalsums.sum())  #axis=1, keepdims=True))
    print("xt=",xt)
    mtmean=2*np.mean(xt)
    print("2 * mt mean=",mtmean)
    inverted=np.abs(mtmean-xt)
    print("inverted=",inverted)
    sm=np.min(inverted)
   # print("sm=",sm)
    if sm>0:
        scaling=10/sm
    else:
        scaling=10000
    print("scaling=",scaling)

    inverted*=scaling
    print("scaled inverted=",inverted)
    yt=np.around(inverted.astype(np.double))  #,decimals=3)
    yt=yt.astype(int, copy=False)
    #print("roundy",yt)
    return(yt)

def build_wheel(count,n):
    print("build wheel count,n",count,n)
    p=[n]*count
    print("p=",p)
    return(p)


def create_wheel(probs):
    count=np.arange(len(probs),dtype=int)
#    print("count=",count,"probs=",probs)
    wheel=np.repeat(count,probs)
    return(wheel)

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

    

stops=np.array([[0,1,2],[2,0,1],[2,1,0],[1,0,2],[0,2,1]])
#stops=np.array([[0,1,2],[1,0,2]])

data=np.array([[0,(10,20),0,3.2,2.1],[1,(11,30),4,0,7.7],[2,(22,4),5.2,6,0]])

test=vpath_distance(data)
print("distance=",test)
print("\n\n\n")
test2=vall_distance(stops,data)
print("wheel weights=",test2)

wheel=create_wheel(test2)
print("final wheel=",wheel)
