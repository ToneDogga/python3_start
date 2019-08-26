# Genetic algorithm solution to the travelling salesman problem written by Anthony Paech 20/8/19

# initialise
# generate search space
#
# generation 1
# randomly generate a population of n encoded strings as a list of numbers representing the journey  eg [0,1,3,5,.....,67,0]
#
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
from __future__ import print_function
from __future__ import division


#import numpy as np
#import pandas as pd
import random
from statistics import mean
from multiprocessing import Pool, Process, Lock, freeze_support, Queue, Lock, Manager
from os import getpid
from time import sleep
import platform
import math
import sys
import time
import datetime
import os
import hashlib
#from pathlib import Path
import multiprocessing 
from timeit import default_timer as timer
import pickle
import numpy as np
import pygame


def numpy_pickle():
    stop_test=[(1,2),(3,4),(5,6),(7,8)]
    no_of_stops=len(stop_test)

    ##path_screen_type = np.dtype(dict(
    ##    names=['generation', 'epoch_length', 'bestgeneration', 'best_distance','stop'],
    ##    formats=[np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))],
    ##    offsets=[0, 8, 16, 16, 16]
    ##))

    ##path_screen_type = np.dtype(dict(
    ##    names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],
    ##    formats=[np.int32,np.str,np.bool,np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))],
    ##    offsets=[5,3,0,0, 8, 8, 16, 0]
    ##))


    mp_dataout=dict(
        names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],
        formats=[np.int32,'|S25',np.bool,np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))]
    )


    path_screen_type = np.dtype(mp_dataout)

    path_screen=np.zeros((1,),dtype=path_screen_type)   # one row only

    #path_screen=np.empty([1,],dtype=path_screen_type)
    print(path_screen)
    input("?")
    ##path_screen['epoch_length']=[1]  #,2,3]
    ##print(path_screen)
    ##input("?")
    path_screen['generation'][0]=53   #[1,2,3]
    print(path_screen)
    input("?")
    path_screen['message'][0]="Hello, world!"   #[1,2,3]
    print(path_screen)
    input("?")
    path_screen['redraw'][0]=True   #[1,2,3]
    print(path_screen)
    input("?")
    path_screen['pid'][0]=1234   #[1,2,3]
    print(path_screen)
    input("?")
    path_screen['best_distance'][0]=88.8   #[1,2,3]
    print(path_screen)
    input("?")
    path_screen['bestgeneration'][0]=9   #[1,2,3]
    print(path_screen)
    input("?")


    ##for i in range(0,len(genepool[bestjourneyno])):
    ##    print(i,stops[genepool[bestjourneyno][i]]) # 
    ##    path_screen["stop"][0][i]=stops[genepool[bestjourneyno][i]]  # 

    for i in range(0,no_of_stops):
        path_screen["stop"][0][i]=stop_test[i]  # 


    print(path_screen)
    input("?")







    ##
    ##path_screen['stop'][0]=[5,6]
    ##print(path_screen)
    ##input("?")
    ##path_screen['stop'][1][1]=8
    ##print(path_screen)
    ##input("?")
    ##path_screen['stop'][1][1][1]=7
    ##print(path_screen)
    ##input("?")
    ##
    ##print(path_screen_type.names)
    ##print(path_screen_type.fields)
    ##input("?")
    ##

    x_as_bytes = pickle.dumps(path_screen)
    print(x_as_bytes)
    print(type(x_as_bytes))

    y = pickle.loads(x_as_bytes)
    print(y)



    print(y["message"][0].decode("utf-8"))
    print(y["redraw"][0])
    print(y["pid"][0])
    print(y["stop"][0])
    return

########################################################



def draw_path(redraw,pygame,screen,font,pid,generation,epoch_length,best_distance,bestgeneration,stoplist):
 #      font = pygame.font.SysFont('Arial', 25)
  #      pygame.display.set_caption('Box Test')

   # blitspace=((0,0),(0,0))   # rect object

  #  pygame.draw.line(screen, (255,0,255), (20,20), (70,80), 2)


    if redraw:
        screen.fill((0,0,0))
        #windowSurface.fill((0,0,0),rect=blitspace)
    ##    for i in range(0,len(ts.genepool[bestjourneyno])-1):
    ##        t1=ts.stops[ts.genepool[bestjourneyno][i]]
    ##        t2=ts.stops[ts.genepool[bestjourneyno][i+1]]
        
            
        for i in range(0,len(stoplist)-1):
            t1=stoplist[i]
            t2=stoplist[i+1]
            pygame.draw.line(screen, (255,255,0),t1,t2)
    #        pygame.display.flip()
           
    #        i+=1

        screen.blit(font.render("Best gen:"+str(bestgeneration)+"/"+str(epoch_length)+"="+str(best_distance), True, (0,255,255)), (5, 5))  #, 2)

       
        
    blitspace=screen.blit(font.render("Gen:"+str(generation)+"/"+str(epoch_length), True, (0,255,0)), (300, 5))  #, 2)
    screen.fill((0,0,0),rect=blitspace)
    screen.blit(font.render("Gen:"+str(generation)+"/"+str(epoch_length), True, (0,255,0)), (300, 5))  #, 2)

      #  time.sleep(1)
      #  pygame.display.update(blitspace)   # clear this rectangle (ts.xsize-200, 5),(ts.xsize,80))
     
    #    pygame.draw.rect(windowSurface, WHITE, (100, 200, +300, +2), 2)
      #  pygame.display.flip()




    pygame.display.flip()
    return




def listener(xsize,ysize,q,filename):    #,l_lock):
    '''listens for messages on the q, writes to file. '''
    print("Queue listener started on:",os.getpid())

    pygame.init()
   # BLACK = (0,0,0)
  #  WIDTH = 600
  #  HEIGHT = 600
    windowSurface = pygame.display.set_mode((xsize, ysize))   #, 0, 32)
    font = pygame.font.SysFont('Arial', 20)
    pygame.display.set_caption('Shortest Path GA')


  #  windowSurface.fill(BLACK)




    
    shortestdist=10000000
   # l_lock.acquire()
    f=open(filename,"w")   #,buffering=8096)
    #f.flush()
    while True:
       # if q.full():
       #     print("q full")
       #     break
       # m = q.get()    #timeout=20)  get_nowait

        y=q.get()
        if y == "kill":
        #    print("trying to kill listener process.  Flushing q.  qsize=",q.qsize())
            #f.write("killed\n")
            while not q.empty():
                m=pickle.loads(q.get())
                msg=m["message"][0].decode("utf-8")
                pid=m["pid"][0]
            #    print("PID:",pid,":",msg)
                f.write(str(pid)+":"+str(msg)+"1:m="+str(m)+"\n")            
                f.flush()
            break
        else:
            m = pickle.loads(y)
            msg=m["message"][0].decode("utf-8")
            pid=m["pid"][0]
          #  print("PID:",pid,":",msg)
            f.write("PID:"+str(pid)+":"+str(msg)+"2:m="+str(m)+"\n")  #+":"+q.qsize()+"\n")
          #  f.write(str(q.qsize())+'\n')
            f.flush()

#              names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],

            
            redraw=m["redraw"][0]
            best_distance=m["best_distance"][0]
     #       print("bd=",best_distance,"sd=",shortestdist)
            if best_distance!=0 and best_distance<shortestdist:
                shortestdist=best_distance
                redraw=True
            generation=m["generation"][0]
            epoch_length=m["epoch_length"][0]
            bestgeneration=m["bestgeneration"][0]
            stoplist=m["stop"][0]
            draw_path(redraw,pygame,windowSurface,font,pid,generation,epoch_length,shortestdist,bestgeneration,stoplist)
        




            
       
  #  q.close()
    f.close()
    pygame.quit()
    print("Queue listener closed on:",os.getpid())
   # l_lock.release()
    return(True)


def add_to_queue(msg,q):
    temp_buffer=[]
      #  If queue is full, put the message in a temporary buffer.
      #  If the queue is not full, adding the message to the queue.
      #  If the buffer is not empty and that the message queue is not full,
      #  putting back messages from the buffer to the queue.
  #  print(q.qsize())    
    if q.full():
        temp_buffer.append(msg)
    else:
        q.put(msg)
     #   q.put(str(q.qsize())+'\n')
        if len(temp_buffer) > 0:
            add_to_queue(temp_buffer.pop())
    #history.append(q.qsize)
    return   #(history)




##        
### Python program to find SHA256 hash string of a file
##def hash_a_file(FILENAME):
##    #filename = input("Enter the input file name: ")
##    sha256_hash = hashlib.sha256()
##    with open(FILENAME,"rb") as f:
##        # Read and update hash string value in blocks of 4K
##        for byte_block in iter(lambda: f.read(4096),b""):
##            sha256_hash.update(byte_block)
##    return(sha256_hash.hexdigest())
##
##
##
##def count_file_rows(FILENAME):
##    with open(FILENAME,'r') as f:
##        return sum(1 for row in f)
##





def create_stops(x,y,no_of_stops):
    stops=[]   # starting point
    for s in range(1,no_of_stops):
        unique=False
        while not unique:
            point=(random.randint(0,x-1),random.randint(0,y-1))
            # check if point already exists
            if not stops:
                unique=True
            else:    
                for h in stops:
               #     print("h=",h,"poimnt=",point)
               #     input("?")
                    if (point==h):
                   #     print("point found",point,h)
                        unique=False
                        break
                    else:
                        unique=True       
             #   print("h loop finished")       

        stops.append(point)
   # print("stops=",j.stops)
    return(stops)  



def calc_distance(start,finish):
  #  print("calc dist from",start,"to",finish)
    startx=start[0]
    starty=start[1]
    finishx=finish[0]
    finishy=finish[1]
    return(math.sqrt((finishx-startx)**2+(finishy-starty)**2))



def create_starting_genepool(stops,poolsize):
    #stops=j.no_of_stops   # less one as we need to start and finish in the same place
    gene=[]
    for g in range(poolsize-1):
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
    return(gene)       



def calc_distances(stops,no_of_stops,poolsize,genepool):
 #   clock_start=time.process_time()
    
    best=1000000
    bestno=0
    total_dist=[]  # first entry goes nowhere.  it is -1 to 0 effectively
    for i in range(0,poolsize-1):
        dist=0
        for k in range(0,no_of_stops-2):
        #    print("i=",i,"k=",k,"g=",ts.genepool[i][k])
            dist+=calc_distance(stops[round(genepool[i][k])],stops[round(genepool[i][k+1])])
       # print("total dist=",dist,"from",ts.stops[ts.genepool[i][0]],"to",ts.stops[ts.genepool[i][k]])    
        total_dist.append(round(dist,2))
        if dist<best:
            best=dist
            bestno=i
    return(total_dist,bestno)



def spin_the_mating_wheel(poolsize,probtable_len,wheel,wheel_len,genepool):
    clock_start=time.process_time()

    mates=[]
    for i in range(1,poolsize):
        go_back=True
        while go_back:
            # pick a random string for mating
            first_mate_no=random.randint(0,probtable_len-1)
            # choose its mate from the probability_table
            second_mate_no=first_mate_no
            while second_mate_no==first_mate_no:
                second_mate_no=wheel[random.randint(0,wheel_len-1)]
            go_back=False
            if genepool[first_mate_no]==genepool[second_mate_no]:
                go_back=True

        mates.append((genepool[first_mate_no],genepool[second_mate_no]))      # mates is a list of tuples to be mated               

   # clock_end=time.process_time()
   # duration_clock=clock_end-clock_start
   # j.tsout.write("ts spin the mating wheel - Clock: duration_clock ="+str(duration_clock)+"\n")
  
    return(mates)  


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
        cut2_choice=random.randint(1,len(jlist2)-1)   # choose the first cut position in the chromosome


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



def pmx_loop(mates):
    clock_start=time.process_time()

    newpool=[]
    for i in range(0,len(mates)):
        newg1,newg2=pmx(mates[i][0],mates[i][1])
       # newg1.append(newg1[0])  #  add the start point to the end
       # newg2.append(newg2[0])  #  add the start point to the end
        #    print("pmx new:",newg1,newg2)

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
  
    return(newpool)





def mutate(newpool,mutation_rate):
    # we need to change random elements of randoms lists in the genepool by simply swapping two stops
    # we cannot change the first or last element
    #the mutation rate should ideally be about 1 in a 1000 bits.  a mutation rate of 1000 means 1 in a 1000 bits
    # number of bits going through per mutation cycle= no_of_alleles*2 ploidy * pop_size
    # 
    clock_start=time.process_time()
    
    mutation_count=0
    pop_size=len(newpool)
    pop_len=len(newpool[0])
     
        #int(round(gene_pool_size/mutation_rate))
    for m in range(0,mutation_rate):
        
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
  
        
    return(newpool)




def dataout_load_and_send(dataout,q):
########################################
# send data out on a queue.  This is the only way to communicate with the listener
   #     names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],


# load data into numpy data structure

        dataout["pid"][0]=pid
        dataout["message"][0]=""
        dataout["redraw"][0]=False
        dataout["generation"][0]=gencount
        dataout["epoch_length"][0]=epoch_length
        dataout["bestgeneration"][0]=bestjourneyno
        dataout["best_distance"][0]=bestjourneydist
        
        for i in range(0,len(genepool[bestjourneyno])):
        #    print(i,stops[genepool[bestjourneyno][i]]) # 
            dataout["stop"][0][i]=stops[genepool[bestjourneyno][i]]  #

        print("data out ready=",dataout)

        dataout_as_bytes = pickle.dumps(path_screen)
        print("dataout as bytes=",dataout_as_bytes)
        #print(type(x_as_bytes))
   
        add_to_queue(dataout_as_bytes,q)




#############################################







#################################################

def ga_ts_run(stops,no_of_stops,poolsize,epoch_length,mutation_rate,q):

# j is a copy of the variable class
# q is the multiprocessing queue that the messages are sent back through

#####################################################
# define the numpy data type that this function in multiprocessor mode will use to send data through a queue to the listener function
    mp_dataout=dict(
        names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],
        formats=[np.int32,'|S25',np.bool,np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))]
    )


    mp_dataout_type = np.dtype(mp_dataout)

    dataout=np.zeros((1,),dtype=mp_dataout_type)   # one row only


#######################################################3

    pid=str(os.getpid())
  #  tsout=open("TSGAresults1.txt","a")

    dataout["pid"][0]=pid
    dataout["message"][0]="Shortest path GA function started."
    dataout["redraw"][0]=False

    dataout_as_bytes = pickle.dumps(dataout)
   #print("dataout as bytes=",dataout_as_bytes)
    #print(type(x_as_bytes))


    add_to_queue(dataout_as_bytes,q)


    bestjourneydist=100000000
    bestjourneyofepoch=[]
    bestjourneyno=0

    scaling=1000
    journey=[]
    
    dataout["pid"][0]=pid
    dataout["message"][0]=" :  Create starting genepool."
    dataout["redraw"][0]=False

    dataout_as_bytes = pickle.dumps(dataout)
   #print("dataout as bytes=",dataout_as_bytes)
    #print(type(x_as_bytes))

    add_to_queue(dataout_as_bytes,q)
    
    genepool=create_starting_genepool(no_of_stops,poolsize)
  #  print("genes=",ts.genepool)

    

    t="PID:"+pid+" :Gene pool size "+str(poolsize)+"\n"

    t=t+"PID:"+pid+" :Epoch length="+str(epoch_length)+"\n"
    starttime=str(datetime.datetime.now())

    t=t+"PID:"+pid+" Genetic algorithm. Started at: "+starttime+"\n"
    print("PID:"+pid+":starting....")
    for gencount in range(1,epoch_length+1):


        #  genepool is a list of lists
        # each list genepool[x] is a journey
        # there are ts.poolsize journeys

      #  print("calc total distance for each journey in genepool")
        distances,bestjourneyno=calc_distances(stops,no_of_stops,poolsize,genepool)
        
       # print("best dist=",ts.distances[bestjourneyno],"bestjn=",bestjourneyno)
        totalgenepool_dist=distances[bestjourneyno]
     #   print("bestjourneyno=",bestjourneyno)
       # bestdist=ts.distances
       # bestdist.sort()
     #   print("best=",bestdist[0])
        
        dataout["redraw"][0]=False
       # print("genepool dist=",ts.totalgenepool_dist)
        if totalgenepool_dist<bestjourneydist:
            bestjourneydist=totalgenepool_dist
            bestjourneygen=gencount
            dataout["redraw"][0]=True
        #    print("")
          #  print("best journey=",ts.genepool[bestjourneyno],"dist=",bestjourneydist)
            t=t+"PID:"+pid+" best journey= "+str(genepool[bestjourneyno])+" dist="+str(round(bestjourneydist,2))+" gen:"+str(gencount)+" Mutation rate: "+str(mutation_rate)+" \n\n"
          
        if gencount==epoch_length:
            print("")
        print("PID:",pid,"Generation:",gencount," shortest distance:",round(bestjourneydist,2))  #,end='\r',flush=True)
     #   w="PID:"+str(os.getpid())+" Generation:"+str(gencount)," shortest distance:",str(round(bestjourneydist,2))+"\n"
     #   add_to_queue(w,q)


########################################
# send data out on a queue.  This is the only way to communicate with the listener
   #     names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],


# load data into numpy data structure

        dataout["pid"][0]=pid
        dataout["message"][0]=""    
        dataout["generation"][0]=gencount
        dataout["epoch_length"][0]=epoch_length
        dataout["bestgeneration"][0]=bestjourneygen
        dataout["best_distance"][0]=bestjourneydist
        
        for i in range(0,len(genepool[bestjourneyno])):
        #    print(i,stops[genepool[bestjourneyno][i]]) # 
            dataout["stop"][0][i]=stops[genepool[bestjourneyno][i]]  #

      #  print("data out ready=",dataout)

        dataout_as_bytes = pickle.dumps(dataout)
        #print("dataout as bytes=",dataout_as_bytes)
        #print(type(x_as_bytes))
   
        add_to_queue(dataout_as_bytes,q)




#############################################

        # reproduction
        # total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
        # create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
        # spin the probability_table m times each time yielding a reproduction candidate of the population
        # in this way more highly fit strings have more offspring in the next generation.



        # calc payoff probabilities
        probability_table=[]
        for x in range(0,len(distances)):
            probability_table.append((distances[x]/totalgenepool_dist)*scaling)
    #    print("prob table=",ts.probability_table,"sum=",sum(ts.probability_table))

        # setup a roulette wheel with the smallest distances having the largest slice of the wheel

        probtable_len=len(probability_table)
        if probtable_len<5:
            print("Warning, increase scaling.  Probability table",probtable_len,"too small.   <5 ")
        elif probtable_len>500:
            print("WARNING, decrease scaling, Probability table",probtable_len,"too big.  > 500")
            
        mpt=round(mean(probability_table))
        
        wheel=[]
        n=0 
        while n<=probtable_len-1:
            piesize=round(probability_table[n])
            if piesize<0:
                piesize=0
                
                # invert probabilities
            wheel=wheel+([n] * abs((2*mpt)-piesize))   # invert across mean
            n=n+1
          
    #    print("wheel=",ts.wheel)
        
        wheel_len=len(wheel)
        
        # spin the mating wheel and create a new population
    #    print("spin the mating wheel and create a new population")

        #j.mates=spin_the_mating_wheel(j)
        mates=spin_the_mating_wheel(poolsize,probtable_len,wheel,wheel_len,genepool)

    #   print("mates=",ts.mates)

    #    print("PMX crossover")
        # turn the list of tuples of lists into a list of lists for the next round
        genepool=pmx_loop(mates)

        genepool=mutate(genepool,mutation_rate)   # mutation rate is numbner of mutations per call
        
    #    print("new generation=")

################################################

    t=t+"\n\n"
    print("\n\nFinished PID:",pid)   # genepool length=",len(ts.genepool))
    print("PID:",pid,"Best journey dist=",round(bestjourneydist,2),"Mutation rate=",mutation_rate)
    t=t+"Best journey dist="+str(round(bestjourneydist,2))+" Mutation rate="+str(mutation_rate)+" \n"
 
    print("PID:",pid,"Best route")
   # add_to_queue("PID:"+pid+" Best route",q)
    starttime=str(datetime.datetime.now())
   # clock_end=timer()  #.process_time()
   # duration_clock=clock_end-clock_start

  #  print("payoff pid:[",os.getpid(),"] finished chunk",rowno,"rows in:",duration_clock,"secs.")

    t=t+"PID:"+pid+" Best route"+" \n"
    for i in range(0,len(genepool[bestjourneyno])):
        print("PID:",pid,"move (",i,"):",str(stops[genepool[bestjourneyno][i]]))
        t=t+("PID:"+pid+" move ("+str(i)+") :"+str(stops[genepool[bestjourneyno][i]])+"\n")
        i+=1

    dataout["pid"][0]=pid
    dataout["message"][0]=t
    dataout["redraw"][0]=False

    dataout_as_bytes = pickle.dumps(dataout)
   #print("dataout as bytes=",dataout_as_bytes)
    #print(type(x_as_bytes))

    add_to_queue(dataout_as_bytes,q)   # send results of run to file and print

   # return
   # tsout.flush()
    time.sleep(4)  # wait for other processes
    
    finishtime=str(datetime.datetime.now())
    print("PID:",pid," Finished at",finishtime)
    msg="PID:"+pid+" Finished at "+str(finishtime)+"\n"
    
    dataout["pid"][0]=pid
    dataout["message"][0]=msg
    dataout["redraw"][0]=False

    dataout_as_bytes = pickle.dumps(dataout)
   #print("dataout as bytes=",dataout_as_bytes)
    #print(type(x_as_bytes))

    add_to_queue(dataout_as_bytes,q)

    return











#######################################################

def main():
    freeze_support()

    if(len(sys.argv) < 2 ) :
        print("Usage : python TSGA_xxxxx.py epoch_length")
        sys.exit()

   # class ts(object):
   #     pass

  #  ts.xsize=0
  #  ts.ysize=0
  #  ts.no_of_stops=0
    #ts.stops=[]
    #ts.journey=[]
    #ts.distance=0
    #ts.distances=[]
   
    #ts.probablilty_table=[]
    #ts.probtable_len=0
    #ts.wheel=[]
    #ts.wheel_len=0
    #ts.mates=[]

    poolsize=200
    mutation_rate=[1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8]   # different mutation rates for the different parallel processes
    
    #ts.genepool=[]

    #ts.scaling=1000    
    filename="TSGAresults.txt"

    print("\n\nGenetic algorithm - Shortest path by Anthony Paech 20/8/19")
    print("==========================================================\n")
    
    xsize=int(input("x size?"))
    ysize=int(input("y size?"))
    no_of_stops=int(input("number of stops?"))

    no_of_stops+=1   #  has to return back to the start

    print("create stops")
    stops=create_stops(xsize,ysize,no_of_stops)
    print("stops=",stops)
  #  print("create distance table")
  #  for x in range(0,ts.no_of_stops-2):
  #      ts.distance=calc_distance(ts.stops[x],ts.stops[x+1])
  #      print("distance=",x,"->",x+1,"=",ts.distance)

  #  tsout=open("tsout.txt","a")

  # clear out report file 
   # t=open("TSGAresults1.txt","w")
   # t.close()
    
####################################################

##    pygame.init()
##   # BLACK = (0,0,0)
##  #  WIDTH = 600
##  #  HEIGHT = 600
##    windowSurface = pygame.display.set_mode((xsize, ysize))   #, 0, 32)
##    font = pygame.font.SysFont('Arial', 20)
##    pygame.display.set_caption('Shortest Path GA')
##

  #  windowSurface.fill(BLACK)

#####################################################

    manager = multiprocessing.Manager()
    q = manager.Queue()    
   
    cpus = multiprocessing.cpu_count()
####################################################
    
   
    windows=platform.system().lower()[:7]
    print("platform=",windows)
    if windows=="windows":
        EOL="\r\n"
    else:
        EOL='\n'

    print("cpus=",cpus)
  #  add_to_queue("cpus="+str(cpus)+"\n",q)
    clock_start=timer() #time.process_time()

   # chunk=int(row_numbers/(cpus-1))   # less one cpu for listener
   # remainder=row_numbers%(cpus-1)

   # lwremainder=row_numbers%linewrite
    
   # print("rownumbers=",row_numbers," chunk size=",chunk," remainder=",remainder)
   # print("lines per write=",linewrite," remainder=",lwremainder)

    multiple_results=[]
    epoch_length=int(sys.argv[1])
    i=0
    with Pool(processes=cpus) as pool:  # processes=cpus-1
         #put listener to work first
        watcher = pool.apply_async(listener, args=(xsize,ysize,q,filename ))


    #    if remainder!=0:
    #        multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,args=(0,remainder,q,linewrite,lwremainder ))]
    #    else:
    #        multiple_results=[]

    
        for i in range(0,cpus-1):          
            multiple_results.append(pool.apply_async(ga_ts_run,args=(stops,no_of_stops,poolsize,epoch_length,mutation_rate[i],q )))  # stops, journey and poolsize, epoch length and name of q
 #           i+=1
            
        for res in multiple_results:
            result=res.get(timeout=None)
            res.wait()

 


        


   #     epoch_length=100
   #     bestjourneyno=ga_ts_run(ts,epoch_length,q)






        print("Generate results finished")
        
   #     add_to_queue("Generate results finished\n",q)

   #     print("killing listener")
        q.put("kill")
        result=watcher.get(timeout=None) 
        watcher.wait()

        
   # print("try to close pool")
    pool.close()
  #  print("pool closed.  trying to join() pool")
    pool.join()
    print("pool closed. join() complete")

  #  print("result=",result)   
    
##    for i in range(0,len(genepool[bestjourneyno])):
##        print("move (",i,"):",str(stops[genepool[bestjourneyno][i]]))
##        tsout.write("move ("+str(i)+") :"+str(stops[genepool[bestjourneyno][i]])+"\n")
##        i+=1
##        
 #   tsout.close()    
        

    return





###########################################


 

if __name__ == '__main__':
    main()


