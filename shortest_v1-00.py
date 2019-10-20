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
import pandas as pd
import datetime
import random
import sys
from timeit import Timer
import pygame

#
#  new logic optimised for speed
# use numpy and vectorization
# avoid for loops
#
#  numpy array
#  [ [index,(x,y),distancetoindex0,distancetoindex1,....,distancetoindexn],..]





def create_stops(x,y,no_of_stops):
    stops=[0.0]   # starting point - distance
    for s in range(1,no_of_stops):
        unique=False
        while not unique:
            point=(random.randint(0,x-1),random.randint(0,y-1))
            if s==1:
                endp=point
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

        stops.append([s-1,point])
    stops.append([s,endp])
   # print("stops=",j.stops)
    return(stops)  




def build_distance_lookup(ysize,xsize,no_of_stops):
    stops=create_stops(xsize,ysize,no_of_stops)
    s = np.arange(0,no_of_stops,1)
    a = np.array([stops[i][1] for i in s])   
    d = np.around(distance.cdist(a,a,'euclidean'),2)                           
    return(np.concatenate((stops,d),axis=1))

    
def path_distance(dpath):
    no_of_steps=len(dpath)
  #  print("\nsorted dpath=\n",dpath,"steps=",no_of_steps)
    totaldist=0
    for index in range(0,no_of_steps-1):
       # nextindex=dpath[index+1][0]
        totaldist+=dpath[index][dpath[index+1][0]+2]   # column 2 is where the distances start
   #     print("nextindex=",nextindex,"totaldist=",totaldist)
    return(totaldist)    




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
        #gene[g].append(gene[g][0])   # return to end
    return(gene)       




    

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










def spin_the_mating_wheel(poolsize,probtable_len,wheel,wheel_len,genepool):
   # clock_start=time.process_time()

    mates=[]
    for i in range(1,poolsize):
        go_back=True
        while go_back:
            # pick a random string for mating
         #   first_mate_no=random.randint(0,probtable_len-1)
            first_mate_no=wheel[random.randint(0,wheel_len-1)]
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
    sp.genepool=[]
    sp.poolsize=1000
    sp.epoch_length=int(sys.argv[1])

    sp.xsize=int(input("x size?"))
    sp.ysize=int(input("y size?"))
    sp.no_of_stops=int(input("number of stops?"))


    

    print("\n\nGenetic algorithm Optimised with numpy- Shortest path by Anthony Paech 20/8/19")
    print("===================================================================================\n")
    print("Gene pool size",sp.poolsize)
    print("Epoch length=",sp.epoch_length)

    sp.genepool=create_starting_genepool(sp.no_of_stops,sp.poolsize)
    print("\ngenepool=\n",sp.genepool,"len=",len(sp.genepool))
 

    dist=build_distance_lookup(sp.ysize,sp.xsize,sp.no_of_stops)
    np.set_printoptions(precision=0)
    np.set_printoptions(suppress=True)
    print("\ndist=\n",dist)
    #dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1]})
   # dataframe = pd.DataFrame.from_records(dist)
   # print("pd=",dataframe)
    stoplist=dist[...,1] 
    print("\n\nstoplist=\n",stoplist)

   
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
    #path=list(np.arange(0,sp.no_of_stops-1,1))    # path=[5,4,2,3,1,0]
 #   print(path)
    best_list=[] 
    while sp.gen_count<=sp.epoch_length:
        i=random.randint(0,sp.poolsize-1)
        path=sp.genepool[i]

      #  print("path=",path)
        loop=path+[path[0]]
     #   print("loop=",loop)
        dpath=dist[loop]
      #  print("\npath sorted=\n",dpath)
        stoplist=dpath[...,1] 
      #  print("\n\nstoplist=\n",stoplist)
        distance=round(path_distance(dpath),2)
        sp.genepool[i].insert(0,distance)
        
        if distance<best_distance:
            best_distance=distance
            print("best=",best_distance)
            print("gp=\n",sp.genepool)
            best_list=stoplist
      #  print("distance=",distance)

        draw_path(pygame,windowSurface,font,best_list,sp.gen_count,distance,best_distance)


##        distances,bestjourneyno=calc_distances(stops,no_of_stops,poolsize,genepool)
##        
##       # print("best dist=",ts.distances[bestjourneyno],"bestjn=",bestjourneyno)
##        totalgenepool_dist=distances[bestjourneyno]
##
##
##
##        probability_table=[]
##        for x in range(0,len(distances)):
##            probability_table.append((distances[x]/totalgenepool_dist)*scaling)
##    #    print("prob table=",ts.probability_table,"sum=",sum(ts.probability_table))
##
##        # setup a roulette wheel with the smallest distances having the largest slice of the wheel
##
##        probtable_len=len(probability_table)
##
##        probtable_set=len(set(probability_table))
##        
##        if probtable_len<5:
##            print("Warning, increase scaling.  Probability table",probtable_len,"too small.   <5 ")
##        elif probtable_len>4000:
##            print("WARNING, decrease scaling, Probability table",probtable_len,"too big.  > 500")
##            
##        mpt=int(round(mean(probability_table)))
##        
##        wheel=[]
##        n=0 
##        while n<=probtable_len-1:
##            piesize=int(round(probability_table[n]))
##            if piesize<0:
##                piesize=0
##                
##                # invert probabilities
##            wheel=wheel+([n] * abs((2*mpt)-piesize))   # invert across mean
##            n=n+1
##          
##    #    print("wheel=",ts.wheel)
##        
##        wheel_len=len(wheel)
##        
##        # spin the mating wheel and create a new population
##    #    print("spin the mating wheel and create a new population")
##
##        #j.mates=spin_the_mating_wheel(j)
##        mates=spin_the_mating_wheel(poolsize,probtable_len,wheel,wheel_len,genepool)
##
##    #   print("mates=",ts.mates)
##
##    #    print("PMX crossover")
##        # turn the list of tuples of lists into a list of lists for the next round
##        genepool=pmx_loop(mates)
##
##        genepool=mutate(genepool,mutation_rate)   # mutation rate is numbner of mutations per call
 












     #   np.random.shuffle(path)
        sp.gen_count+=1

 
    
  #
    print("\nfinished")
    input("?")
    pygame.quit()
    return


###########################################


 

if __name__ == '__main__':
    main()


