# Genetic algorithm solution to the travelling salesman problem written by Anthony Paech 20/8/19
#
#  v3 improvements 20/10/19
# use numpy vectorisation instead of for loops to speed up
# better report on best result so far
# allow the ability to stop mid way
from __future__ import print_function
from __future__ import division



#import scipy
import math
from scipy.spatial import distance
import numpy as np
#import pandas as pd
import datetime
import random
import sys
from timeit import Timer
import pygame
from itertools import chain

#
#  new logic optimised for speed
# use numpy and vectorization
# avoid for loops
#
#  numpy array
#  [ [index,(x,y),distancetoindex0,distancetoindex1,....,distancetoindexn],..]


def create_starting_genepool(stops,poolsize):
    #stops=j.no_of_stops   # less one as we need to start and finish in the same place
    gene=[]
    for g in range(poolsize):
        gene.append([])
        gene[g]=[]
        for alleles in range(0,stops-1):
            unique=False
            c=gene[g]
            while not unique:
                destination=random.randint(0,stops-2)
                # check if point already exists
                if len(gene[g])==0:
                    unique=True
                else:    
                    for d in c:
               #     print("h=",h,"poimnt=",point)
               #     input("?")
                        if (destination==d):
                   #     print("point found",point,h)
                            unique=False
                            break
                        else:
                            unique=True       
              #  print("gene=",gene)
              #  input("?")
             
            gene[g].append(destination)
        gene[g].append(gene[g][0])   # return to end
    return(np.asarray(gene))       




##def create_stops2(x,y,no_of_stops):
##    stops=np.empty((1,no_of_stops,2)).astype(int)   # starting point - distance
##    endpoint=np.empty((1,no_of_stops,2)).astype(int)
##    for s in range(1,no_of_stops):
##        points=np.empty((1,2)).astype(int)
##        
####        unique=False
####        while not unique:
##        point1=int(random.randint(0,x-1))    #,random.randint(0,y-1)]
##        point2=int(random.randint(0,y-1))
##        points=np.concatenate((point1,point2))
##       # points=np.append(points,point2)
##        
##        if s==1:
##            endpoint=np.concatenate((point1,point2))
##        #    endpoint=np.append(endpoint,point2)
##            
##        # check if point already exists
####            if stops.size==0:
####                unique=True
####            else:    
####                for h in stops:
####               #     print("h=",h,"poimnt=",point)
####               #     input("?")
####                    if (points==h): #and (point2==h):
####                   #     print("point found",point,h)
####                        unique=False
####                        break
####                    else:
####                        unique=True       
####             #   print("h loop finished")       
##
##
##        stops=np.append((stops,int(s-1),points))
##       # stops=np.append(stops,points)
##    #    stops=np.append(stops,(s-1,tuple(point)))
##    stops=np.append((stops,int(s),endpoint))
##   # stops=np.append(stops,endpoint)
##    #stops=np.append(stops,(s,tuple(endp)))
##
##    #print("create stops=",stops)
##    #input("?")
##    #stops2 = np.array(list(chain.from_iterable(stops)))
##    #stops3=np.array(stops)
##    return(stops)  

def create_stops(x,y,no_of_stops):
    stops=np.zeros((1,no_of_stops,2)).astype(int)   # starting point - distance
    endpoint=np.zeros((1,no_of_stops,2)).astype(int)
    for s in range(1,no_of_stops):
        point=np.zeros((1,2)).astype(int)
      #  unique=False
     #   while not unique:
        point[0][0]=random.randint(0,x-1)
        point[0][1]=random.randint(0,y-1)
        if s==1:
            endpoint=point
            # check if point already exists
       #     if stops.size==0:
        #        unique=True
        #    else:
         #       if np.any(stops==point):
             #   for h in stops:
              #      print("h=",h,"poimnt=",point)
               #     input("?")
               #     if (point[0][0]==h) and (point[0][1]==h):
          #          print("point found",point)
          #          unique=False
          #          break
          #      else:
          #          unique=True       
             #   print("h loop finished")       

        stops[0][s-1]=point
    stops[0][s]=endpoint
    #print("create stops=",stops)
    #input("?")
    return(stops)  


##def build_distance_lookup(ysize,xsize,no_of_stops):
##    stops=create_stops(xsize,ysize,no_of_stops)
##    s = np.arange(0,no_of_stops,1)
##    a = np.array([stops[i][1] for i in s])   
##    d = np.around(distance.cdist(a,a,'euclidean'),2)                           
##    return(np.concatenate((stops,d),axis=1))
##



##def path_distance(path):
##    # numpy array of tuples
## #   print("path=",path)
##    destinations=np.roll(path, -1, axis=0)
## #   print("destinations=",destinations)
####    s = np.arange(1,len(path),1)
####    print("s=",s)
####    a = np.array([path[i] for i in s])
####    print("a=",a)
## #   d = np.around(distance.cdist(path,destinations,'euclidean'),2)                           
##    d = np.around(distance.cdist(path,destinations,'euclidean'),2)                           
##    dist=d.diagonal()
##  #  print("dist=",dist)
##    return(np.sum(dist))    

def path_distance(path):
    # numpy array of tuples
    length=len(path)
    i=0
    result=np.array([])
    while i<length:
    #    print("i=",i)
        onepath=path[i]
    #    print("onepath=",onepath)
        destination=np.roll(onepath, -1, axis=0)
    #    print("destination=",destination)
        d = np.around(distance.cdist(onepath,destination,'euclidean'),2)
    #    print("d=",d)
        dist=d.diagonal()
    #    print("dist=",dist)
        total=np.around(np.sum(dist),2)
    #    print("total=",total)
        result=np.append(result,total)
     #   print("result=",result)
        i+=1
    return(result)




def fitness_calc(genepool,stops):
    # vectorized version for speed
   # print("fitness calc.  \nstops=\n",stops,"\npdata=\n",pdata)
    #newpdata = pdata[tuple(stops)]
  #  print("stops=\n",stops)
    ss=int(round(stops.size/2,2))
    count=np.arange(0,ss,1,int).reshape((1,ss,1))  #.reshape((ss,1))
  #  print("count=",count)
    stops2=stops.reshape((ss,2))
    count2=count.reshape((ss,1))
  #  print("count2=",count2)
   # stops2=stops[...,:2]
  #  print("after stops=\n",stops2)
    indexer=np.concatenate((count2,stops2),axis=1)
  #  print("indexer=",indexer)
  #  print("stops[...,0]=",stops[...,0])
  #  s=stops[...,0].astype(int)
  #  point=stops[...,1]
  #  print("point=",point)

    j=indexer[genepool]
 #   print("j=",j)
    journeys=j[...,1:]
  #  print("j2=",journeys)
   # jour2=jour
   # print("jour shape,size=",jour.shape,jour.size)
  #  jour2=np.expand_dims(jour,axis=2)   #(1,jour.size,2,1))
   # print("jour2=",jour2,jour2.shape)
  #  journeys=np.reshape(jour2,(2,jour2.size/2,2))
   # journeys=np.expand_dims(jour,axis=1)  
#    print("journeys=\n",journeys,journeys.shape)
   # jour=journeys.reshape((1,journeys.shape))
 #   path_distance=vectorise(journeys)
    distances=path_distance(journeys)
 #   print("distances=\n",distances)   #,distances.shape)
    
    #  return fitness,best_distance,best_list
##    newpdata = pdata[stops]
##
##    #print("\nnewpdata=\n",newpdata)
## 
##    stoplist=np.array(newpdata[...,0],dtype=int)
##  #  print("stoplist=",stoplist)   #,"stoplist2=",stoplist2,"stoplist3",stoplist3)
##    destinations=np.roll(stoplist, -1, axis=1)
##  #  print("destinations=",destinations)
##    stopdist=pdata[stoplist]
##  #  print("stopdist=",stopdist)
##    distances=stopdist[...,2:]
##
##  #  print("\ndistances=\n",distances)
##    d=distances[[0],[stoplist],[destinations]]
##  #  print("\nd=\n",d)
##    nps=np.sum(d,axis=2)
##  #  print("nps=",nps)
##          
####    print("\nindexed distances=\n",distances[[stoplist],[destinations]])
##    finalsums=nps[0,...]
###    print("final sums",finalsums)
##
##    finalsumsv=finalsums.reshape(-1, 1)
##
##    stopchoices=stopdist[...,1]
##   # print("stop choices=",stopchoices)
##   # print("final sums",finalsums)
##    journeys=np.concatenate((stopchoices,finalsumsv),axis=1)  # add distances to journeys
##   # print("journeys=",journeys)
  #  print("journeyscut :2",journeys)
  #  print("journeyscut-1",indexer[...,:1])

    minval=np.amin(distances)     #[...,-1])
    best_distance=np.around(minval,2)
  #  print("best distance=",best_distance)
    ind=np.argmin(distances)   #,axis=1)
  #  print("best dist index=",ind)
  #  print("best journey ind=",journeys[ind])
    best_stop_order=journeys[ind]    #np.argmin(distances)]    #ind,...]
  #  print("best stop order=",best_stop_order)
    
  #  print("distances=",finalsums,"best distance=",best_distance)
    xt = np.true_divide(distances, distances.sum())  #axis=1, keepdims=True))
   # print("xt=",xt)
    mtmean=2*np.mean(xt)
   # print("2 * mt mean=",mtmean)
    inverted=np.abs(mtmean-xt)
   # print("inverted=",inverted)
    sm=np.min(inverted)
   # print("sm=",sm)
    if sm>0:
        scaling=10/sm
    else:
        scaling=10000
   # print("scaling=",scaling)

    inverted*=scaling
  #  print("scaled inverted=",inverted)
    yt=np.around(inverted.astype(np.double))  #,decimals=3)
    yt=yt.astype(int, copy=False)
    #print("roundy",yt)
    return(yt,best_distance,best_stop_order.tolist())




def create_wheel(probs):
    count=np.arange(len(probs),dtype=int)
#    print("count=",count,"probs=",probs)
    wheel=np.repeat(count,probs)
    return(wheel)


##  
##def vpath_distance(pdata):
##    # vectorized version for speed   
##    stoplist=np.array(pdata[...,0],dtype=int)
##    nextlist=np.array(stoplist[1:],dtype=int)
##    destinations=np.append(nextlist,[stoplist[0]])
##  #  print("stoplist=",stoplist)
##  #  print("destinations=",destinations)
##  #  print("\npdata=\n",pdata)
##    stopdist=pdata[stoplist]
##    distances=stopdist[...,2:]
##  #  print("\ndistances=\n",distances)
##    d=distances[[stoplist],[destinations]]
##   # print("\nindexed distances=\n",d)
##
##  #  return(np.sum(distances[[stoplist],[destinations]]))
##    return(np.sum(d))



    

def draw_path(pygame,screen,font,bestlist,gen,distance,best_distance):
    
    screen.fill((0,0,0))
        
    for i in range(0,len(bestlist)-1):
        t1=bestlist[i]
        t2=bestlist[i+1]
        pygame.draw.line(screen, (255,255,0),t1,t2)
    #else:
    #    blitspace=screen.blit(font.render("                        ", True, (0,255,0)), (500, 5))  #, 2)
    #    screen.fill((0,0,0),rect=blitspace)

    screen.blit(font.render("Gen:"+str(gen)+" Distance="+str(distance)+" Best Dist="+str(best_distance), True, (0,255,255)), (5, 5))  #, 2)
          
    blitspace=screen.blit(font.render("                        ", True, (0,255,0)), (500, 5))  #, 2)
    screen.fill((0,0,0),rect=blitspace)
   # screen.blit(font.render("Distance="+str(distance), True, (0,255,0)), (300, 5))  #, 2)

    pygame.display.flip()
    return





def pick_mates2(genepool,wheel,poolsize):
    # spin the roulette wheel and choose a pair of genes unique with unique numbers (in the form of a 1D array of stops)
    #
    n = 2  # for 2 random indices.  Mate a pair
    
    index = np.random.choice(wheel.shape[0],(poolsize, n), replace=False)

    mates=np.array(genepool[wheel[index]])
    unique_rows, uniq_cnt = np.unique(mates, axis=0, return_counts=True)
    unique_array = unique_rows[uniq_cnt==1]
    #unique_rows = np.unique(mates, axis=1)
   # print("mates=",mates,mates.shape,"\nunique=",unique_array,unique_array.shape)
    print("\rGenepool array shape:",unique_array.shape,end="\r",flush=True)
    #input("?")
    return(unique_array)




def pick_mates(genepool,wheel,poolsize):
    # spin the roulette wheel and choose a pair of genes unique with unique numbers (in the form of a 1D array of stops)
    #
    n = 1  # for 2 random indices.  Mate a pair
    
    index1 = np.random.choice(wheel.shape[0],(poolsize, n), replace=False)
    index2 =np.random.choice(genepool.shape[0],(poolsize,n),replace=False)
##    print("index1 shape=",index1.shape)
##    print("index2 shape=",index2.shape)
##    print("index1 shape[1:]=",index1.shape[:1])
##    print("index2 shape[1:]=",index2.shape[:1])
##
##    index3=np.reshape(index1,index1.shape[:1])
##    index4=np.reshape(index2,index2.shape[:1]) 
##
##  #  print("index3=",index3,"index4=",index4)
##    print("gp1=",genepool[wheel[index3]])
##    print("gp2=",genepool[index4])

   # print("wheel=",wheel)
    mates=np.concatenate((np.array(genepool[wheel[index1]]),np.array(genepool[index2])),axis=1)
    unique_rows, uniq_cnt = np.unique(mates, axis=0, return_counts=True)
    unique_array = unique_rows[uniq_cnt==1]
    #unique_rows = np.unique(mates, axis=1)
   # print("mates=",mates,mates.shape,"\nunique=",unique_array,unique_array.shape)
    print("\rGenepool array shape:",unique_array.shape,end="\r",flush=True)
    #input("?")
    return(unique_array)



#def pmx_new(mates):
    #  the mates array is a the poolsize number of pairs of 1D arrays in a numpy array
    # each pair needs to have a portion of its DNA swapped.  But not the first stop or the last stop
    # this is used when the order of the bits matters sych as in a travelling salesman problem
    # select two random cutting points that are not the same on the length of the chromosome
    #  a =   984|567|1320
    #  b =   871|230|9546   
#   map string b to string a
  #  moves to:
  #   a' =   984|230|1320
  #   b' =   871|567|9546
  #

  # now we haveto swap back the "duplicates"
  # number 2,3,0 have been mapped string b to string a
  # now on b, 
#  then mapping string a onto b,
#   a" =  984|230|1657
#   b" =  801|567|9243   
#
#  the other 5,6 and 7 in b swap with the a's 2,3 and 0
  # so each string contains ordering information partially determined by each of the parents
   #
    




def pmx_swap(jlist1,pos1,jlist2,pos2):
    # swap the list element in jlist1[pos1] with jlist2[pos2]

   # print("1=",jlist1,"2=",jlist2,"pos1=",pos1,"pos2=",pos2)

  #  print("before swap A=",jlist1,"at",pos1)
  #  print("before swap B=",jlist2,"at",pos2)

    temp11=jlist1[:pos1]
    temp12=jlist1[pos1+1:]
    swap1=jlist1[pos1:pos1+1]

    temp21=jlist2[:pos2]
    temp22=jlist2[pos2+1:]
    swap2=jlist2[pos2:pos2+1]
     
    newjlist1=temp11+swap2+temp12
    newjlist2=temp21+swap1+temp22

  #  print("after swap A=",temp11,"|",swap2,"|",temp12)
  #  print("after swap B=",temp21,"|",swap1,"|",temp22)

    return(newjlist1,newjlist2)

#def pmx_np_loop(mates):
    


# mating and reproduction
def pmx(jlist1,jlist2):    # partially matched crossover
    
    # this is used when the order of the bits matters sych as in a travelling salesman problem
    # select two random cutting points that are not the same on the length of the chromosome
    #  a =   984|567|1320
    #  b =   871|230|9546   
#   map string b to string a
  #  moves to:
  #   a' =   984|230|1320
  #   b' =   871|567|9546
  #

  # now we haveto swap back the "duplicates"
  # number 2,3,0 have been mapped string b to string a
  # now on b, 
#  then mapping string a onto b,
#   a" =  984|230|1657
#   b" =  801|567|9243   
#
#  the other 5,6 and 7 in b swap with the a's 2,3 and 0
  # so each string contains ordering information partially determined by each of the parents
   #

 #   print("pmx crossover. alleles=",no_of_alleles)




    cut1_choice=random.randint(1,len(jlist1)-1)   # choose the first cut position in the chromosome
#  not right at the start
    cut2_choice=cut1_choice
    while cut2_choice==cut1_choice:
        cut2_choice=random.randint(1,len(jlist2)-1)   # choose the second cut position in the chromosome


# get them in order so cut1 is first, then cut2
    if cut2_choice < cut1_choice:
        temp=cut1_choice    # swap
        cut1_choice=cut2_choice
        cut2_choice=temp

   # print("pmx cut1=",cut1_choice," cut2=",cut2_choice)

    child1=""
    child2=""
    remain11=jlist1[:cut1_choice]
    remain12=jlist1[cut2_choice:]   #-1]   # last element is return dest. remove here before pmx
    swap1=jlist1[:cut2_choice]            
    swap11=swap1[cut1_choice:]

    remain21=jlist2[:cut1_choice]
    remain22=jlist2[cut2_choice:]   #-1]  # last element is return dest. remove here before pm
    swap2=jlist2[:cut2_choice]
    swap21=swap2[cut1_choice:]

  #  print("before")
  #  print("A",jlist1)
  #  print("B",jlist2)
  #  print("remain11",remain11,"+swap11+",swap11,"+ remain12",remain12)
  #  print("remain21",remain21,"+swap21+",swap21,"+ remain22",remain22)
  #  input("?")

   

    new1=jlist1   #remain11+swap11+remain12
    new2=jlist2    #remain21+swap21+remain22

    temp1=jlist1
    temp2=jlist2

   # temp1=remain11+swap21+remain12
   # temp2=remain21+swap11+remain22

  #  print("swap")
  #  print("after new1:",new1,"\n   new2:",new2)
  #  print("?")

    swaps1=list(swap11)
    swaps2=list(swap21)
  #  print("swaps1",swaps1)
  #  print("swaps2",swaps2)
   # swaps=swaps1
  #  print("swaps11 list=",swaps1)
  #  print("swaps21 list=",swaps2)
 #   print("PMX start A new1=",new1)
 #   print("PMX start B new2=",new2)
 #   print("A swaps needed in cut=",swaps1)
    for i in range(cut1_choice,cut2_choice):
        new1,new2=pmx_swap(new1,i,new2,i)

  #  print("PMX swap p1 A new1=",new1)
  #  print("PMX swap p2 B new2=",new2)
 
   # print("B swaps needed to rebalance=",swaps2)
    k=0
    for s in swaps2:
    #    print("looking for s=",s,"in new1. k=",k)
        found1=False
        found2=False
       #
        for i in range(1,len(new1)-1):
     #       print("i loop. i=",i,"checking new1 for s",s,"i=",i)
            #if i>=cut1_choice and i<cut2_choice:
            #    pass
            #else:
            if new1[i]==s:
      #          print("found s",s," at i=",i,"in new1")
                found1=True
                break
       #     else:
        #        print(s,"not found in new1")

#        print("finding the char",s," in new1 at position j",j)
        if found1:
         #   print("found1")
            for j in range(1,len(new2)-1):
          #      print("j loop. j=",j," looking in new2 in swaps1",swaps1,"k=",k,"=",swaps1[k])
#                if j>=cut1_choice and j<cut2_choice:
#                    pass
#                else:
                      #  print("not jumping over j=",j,"c1=",cut1_choice,"c2=",cut2_choice)          
                if new2[j]==swaps1[k]:

           #         print("found2. swaps1 at k=",k,"swaps[k]=",swaps1[k],"at j=",j,"in new2")
        
                    found2=True
                    break
            #    else:
             #       print(swaps1[k],"swaps1[k] not found in new2. k=",k)


                

        
        if found1 and found2:
            found1=False
            found2=False
            
         #   print("swapping new1 at i",new1,i,"and new2 at j",new2,j) 
            new1,new2=pmx_swap(new1,i,new2,j)
          #  print("swapped new1=",new1,"and new2=",new2) 

       # else:
        #    print("not found k=",k)
        k+=1

    #print("Anew1=",new1)
    #print("Bnew2=",new2)
    #input("?")
        
  #  clock_end=time.process_time()
  #  duration_clock=clock_end-clock_start
  #  ga.tsout.write("ts pmx - Clock: duration_clock ="+str(duration_clock)+"\n")
  
    
    return(new1,new2)






def pmx_loop(npmates):
   # clock_start=time.process_time()

    mates=np.ndarray.tolist(npmates)
    newpool=[]
    for i in range(0,len(mates)):
      #  print("maates=",mates[i][0],mates[i][1])
        newg1,newg2=pmx(mates[i][0],mates[i][1])
       # newg1.append(newg1[0])  #  add the start point to the end
       # newg2.append(newg2[0])  #  add the start point to the end
      #  print("pmx new:",newg1,newg2)

        # validate pmx
        hashsum=sum(newg1)-newg1[-1]
        length=len(newg1)-1
        hashsb=sum(range(1,length))
      #  print("pmx check. hash=",hashsum,"hashsb=",hashsb)
        if hashsum!=hashsb:
            print("newg1=",newg1,"newg2=",newg2)
            input("pmx error?")
        else:
            pass 
        #    print("pmx correct!")
        newpool.append(newg1)
        newpool.append(newg2)


  #  clock_end=time.process_time()
  #  duration_clock=clock_end-clock_start
  #  j.tsout.write("ts pmx loop - Clock: duration_clock ="+str(duration_clock)+"\n")
  
    return(np.array(newpool))





def mutate(genepool,mutation_rate):
    # we need to change random elements of randoms lists in the genepool by simply swapping two stops
    # we cannot change the first or last element
    #the mutation rate should ideally be about 1 in a 1000 bits.  a mutation rate of 1000 means 1 in a 1000 bits
    # number of bits going through per mutation cycle= no_of_alleles*2 ploidy * pop_size
    # 
    #clock_start=time.process_time()

    newpool=np.ndarray.tolist(genepool)
    
    mutation_count=0
    pop_size=len(newpool)
    pop_len=len(newpool[0])
     
        #int(round(gene_pool_size/mutation_rate))
    while mutation_count<mutation_rate:
  #  for m in range(0,mutation_rate):
        
        m_choice=random.randint(0,pop_size-1)   # choose a member of the popultation
        s1_choice=random.randint(1,pop_len-2)   # choose a position in the list. not the first or last
             
        s2_choice=random.randint(1,pop_len-2)
        while s2_choice==s1_choice:     # check if point already chosen
            s2_choice=random.randint(1,pop_len-2)


    #    print("mutate element",m_choice,"np=",newpool[m_choice],"swap",s1_choice,"and",s2_choice)

        
        temp1=newpool[m_choice][s1_choice]
        newpool[m_choice][s1_choice]=newpool[m_choice][s2_choice]
        newpool[m_choice][s2_choice]=temp1
        
    #    print("mutated elements at",m_choice,"pos>",s1_choice,",",s2_choice,"=",newpool[m_choice])
    #    input("?")


        mutation_count+=1  
    
  #  clock_end=time.process_time()
  #  duration_clock=clock_end-clock_start
  #  j.tsout.write("ts mutate - Clock: duration_clock ="+str(duration_clock)+"\n")
  
        
    return(np.array(newpool))









# start
# test each member of the population against the total distance travelled value
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

#
#!/usr/bin/env python
#
#######################################################

def main():
    if(len(sys.argv) < 2 ) :
        print("Usage : python shortest_xxxxx.py epoch_length")
        sys.exit()
        
##    params = dict(
##       # no_of_address_bits=2,

    class sp(object):
        pass

    sp.xsize=0
    sp.ysize=0
    
    sp.poolsize=200
    sp.epoch_length=int(sys.argv[1])
    sp.mutation_rate=20

    sp.xsize=int(input("x size?"))
    sp.ysize=int(input("y size?"))
    sp.no_of_stops=int(input("number of stops?"))
    sp.no_of_stops+=1  # add one to return to start


    

    print("\n\nGenetic algorithm Optimised with numpy- Shortest path by Anthony Paech 20/8/19")
    print("===================================================================================\n")
    print("Gene pool size",sp.poolsize)
    print("Epoch length=",sp.epoch_length)

    genepool=create_starting_genepool(sp.no_of_stops,sp.poolsize)
    print("\ngenepool=\n",genepool,genepool.shape,"len=",len(genepool))
  #  input("?")

   # distance_table=build_distance_lookup(sp.ysize,sp.xsize,sp.no_of_stops)
    stops=create_stops(sp.xsize,sp.ysize,sp.no_of_stops)
    #stops = np.array(stops1)

    print("stops=\n",stops,stops.shape)    

 #   np.set_printoptions(precision=0)
 #   np.set_printoptions(suppress=True)
 #   print("\ndistance_table\n",distance_table)

  #  stoplist=distance_table[...,1] 
   # print("\n\nstoplist from distance_table=\n",stoplist,"\n\n")

   # stoplist=stops
   
    pygame.init()
   # BLACK = (0,0,0)
  #  WIDTH = 600
  #  HEIGHT = 600
    windowSurface = pygame.display.set_mode((sp.xsize, sp.ysize))   #, 0, 32)
    font = pygame.font.SysFont('Arial', 20)

    pygame.display.set_caption('Shortest Path GA')

    sp.gen_count=1
    distance=0
    best_distance=10000000

    best_best_distance=10000000
    #path=list(np.arange(0,sp.no_of_stops-1,1))    # path=[5,4,2,3,1,0]
 #   print(path)
    best_list=[]
    best_best_list=[]

##########################################################
    
    while sp.gen_count<=sp.epoch_length:

    #  genepool consists of a list of lists of stops in a 2D numpy array

        fitness,best_distance,best_list=fitness_calc(genepool,stops)

       # print("fitness=",fitness,"\nbest_distance=",best_distance,"\nbest_list=",best_list)
        
        wheel=create_wheel(fitness)
       # print("wheel=",wheel)

       ## create a whole new generation of size poolsize based on the roulette wheel of fitness  
     #   mates=pick_mates(genepool,wheel,sp.poolsize)
        #print("mates=",mates)
        mates=pick_mates2(genepool,wheel,sp.poolsize)

        ## partially matched crossovers
        genepool=pmx_loop(mates)
      #  print("genepool=",genepool)

        ## add a small bit of mutation to the genepool to avoid getting stuck in local minimums
        genepool=mutate(genepool,sp.mutation_rate)   # mutation rate is numbner of mutations per call

      #  if sp.gen_count%50==0:
      #      unique,counts=np.unique(genepool,return_counts=True)
      #      print("\nGen=",sp.gen_count,"/",sp.epoch_length," Genepool diversity=\n",np.asarray((unique, counts)))   #,end="\r",flush=True)
        
        if best_distance<best_best_distance:
            best_best_distance=best_distance
            best_gen=sp.gen_count
            print("\nbest=",best_best_distance,"gen=",best_gen)
            #print("gp=\n",sp.genepool)
            best_best_list=best_list
      #  print("distance=",distance)

        draw_path(pygame,windowSurface,font,best_list,sp.gen_count,best_distance,best_best_distance)


        sp.gen_count+=1

######################################## 
    
  #
  # display best
    draw_path(pygame,windowSurface,font,best_best_list,best_gen,best_best_distance,best_best_distance)
    print("best distance=",best_best_distance,"best route=",best_list,"gen=",best_gen)
 


##    print("\n\nHit space to exit")
##    k=""
##    while k!="space":
##        for event in pygame.event.get():
##            if event.type == pygame.KEYDOWN: 
##                k=pygame.key.name(event.key)
             #   if k=="space":
             #       break 

    print("\nfinished")
    input("?")
    pygame.quit()
    
    return


###########################################


 

if __name__ == '__main__':
    main()


