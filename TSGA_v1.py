# Genetic algorithm solution to the travelling salesman problem

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
# find a random split point in the string a swap the remaining information over between the mating pairs
#
# mutation
# this is the occasional and small chance (1/1000) that an element of a journey list is swapped randomly.
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
import time
#import csv
import math
#import linecache
import sys
from statistics import mean




def mutate(newpopulation,no_of_alleles,ploidy,pop_size,mutation_rate):
    # we need to change random bits on chromo1 or chromo2 column to a random selection - 25%="%" (recessive 1), 25%="1" , 50% = "0"
    #the mutation rate should ideally be about 1 in a 1000 bits.  a mutation rate of 1000 means 1 in a 1000 bits
    # number of bits going through per mutation cycle= no_of_alleles*2 ploidy * pop_size
    # 7*2*128=1792
    
    mutation_count=0

    c1_choice_list=[]
    c2_choice_list=[]
    old_chromo_list=[]
    new_chromo_list=[]
    
    gene_pool_size=no_of_alleles*ploidy*pop_size
    number_of_mutations_needed=int(round(gene_pool_size/mutation_rate))
    for m in range(0,number_of_mutations_needed):
        mutated_bit=""
        chromo=""
   
        c1_choice=random.randint(1,2)   # choose a chromo column
        c1_choice_list.append(c1_choice)
        c2_choice=random.randint(0,pop_size-1)   # choose a member of the popultation
        c2_choice_list.append(c2_choice)
        c3_choice=random.randint(0,no_of_alleles-1)   # choose a position in the chromosome            # chromo1
        c4_choice=random.randint(-1,1)   # choose a new bit  -1=%,0=0,1=1

        if c4_choice==-1:
            mutated_bit="%"
        elif c4_choice==0:
            mutated_bit="0"
        elif c4_choice==1:
            mutated_bit="1"
        else:
            print("bit mutation error.  c4_choice=",c4_choice)
            

        if c1_choice==1:
            # chromo1
        #    old_chromo_list.append(newpopulation.loc[c2_choice,"chromo1"])
            chromo=newpopulation.loc[c2_choice,"chromo1"]
           
        else:
            # chromo2
       #     old_chromo_list.append(newpopulation.loc[c2_choice,"chromo2"])
            chromo=newpopulation.loc[c2_choice,"chromo2"]


        newc=chromo[:c3_choice]+mutated_bit+chromo[c3_choice+1:]


        if c1_choice==1:
            # chromo1
            newpopulation.loc[c2_choice,"chromo1"]=newc
            
        else:
            # chromo2
            newpopulation.loc[c2_choice,"chromo2"]=newc

        mutation_count+=1  
        
    return(newpopulation,mutation_count, c1_choice_list, c2_choice_list)



def create_stops(j):
    j.stops=[]   # starting point
    for s in range(1,j.no_of_stops):
        unique=False
        while not unique:
            point=(random.randint(0,j.xsize-1),random.randint(0,j.ysize-1))
            # check if point already exists
            if not j.stops:
                unique=True
            else:    
                for h in j.stops:
               #     print("h=",h,"poimnt=",point)
               #     input("?")
                    if (point==h):
                   #     print("point found",point,h)
                        unique=False
                        break
                    else:
                        unique=True       
             #   print("h loop finished")       

        j.stops.append(point)
   # print("stops=",j.stops)
    return(j)  



def calc_distance(start,finish):
  #  print("calc dist from",start,"to",finish)
    startx=start[0]
    starty=start[1]
    finishx=finish[0]
    finishy=finish[1]
    return(math.sqrt((finishx-startx)**2+(finishy-starty)**2))



def create_starting_genepool(j):
    stops=j.no_of_stops   # less one as we need to start and finish in the same place
    gene=[]
    for g in range(j.poolsize-1):
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



def calc_distances(ts):
    best=1000000
    bestno=0
    total_dist=[]  # first entry goes nowhere.  it is -1 to 0 effectively
    for i in range(0,ts.poolsize-1):
        dist=0
        for k in range(0,ts.no_of_stops-1):
        #    print("i=",i,"k=",k,"g=",ts.genepool[i][k])
            dist+=calc_distance(ts.stops[round(ts.genepool[i][k])],ts.stops[round(ts.genepool[i][k+1])])
       # print("total dist=",dist,"from",ts.stops[ts.genepool[i][0]],"to",ts.stops[ts.genepool[i][k]])    
        total_dist.append(round(dist,2))
        if dist<best:
            best=dist
            bestno=i
    return(total_dist,bestno)



def spin_the_mating_wheel(j):
    mates=[]
    for i in range(1,j.poolsize):
        go_back=True
        while go_back:
            # pick a random string for mating
            first_mate_no=random.randint(0,j.probtable_len-1)
            # choose its mate from the probability_table
            second_mate_no=first_mate_no
            while second_mate_no==first_mate_no:
                second_mate_no=j.wheel[random.randint(0,j.wheel_len-1)]
            go_back=False
            if j.genepool[first_mate_no]==j.genepool[second_mate_no]:
                go_back=True

        mates.append((j.genepool[first_mate_no],j.genepool[second_mate_no]))      # mates is a list of tuples to be mated               

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
    remain12=jlist1[cut2_choice:]
    swap1=jlist1[:cut2_choice]
    swap11=swap1[cut1_choice:]

    remain21=jlist2[:cut1_choice]
    remain22=jlist2[cut2_choice:]
    swap2=jlist2[:cut2_choice]
    swap21=swap2[cut1_choice:]

   # print("before")
   # print("A",jlist1)
   # print("B",jlist2)
 #   print("remain11",remain11,"+swap11+",swap11,"+ remain12",remain12)
 #   print("remain21",remain21,"+swap21+",swap21,"+ remain22",remain22)
   # input("?")

   

    new1=jlist1   #remain11+swap11+remain12
    new2=jlist2    #remain21+swap21+remain22

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
    #print("A swaps needed in cut=",swaps1)
    for i in range(cut1_choice,cut2_choice):
        new1,new2=pmx_swap(new1,i,new2,i)

   # print("B swaps needed to rebalance=",swaps2)
    k=0
    for s in swaps2:
        i=1
        j=1
   #     k=0
        found1=False
        found2=False
       # print("checking new1 for char",s)
        for i in range(1,len(new1)-2):
           # if i>=cut1_choice and i<cut2_choice:
           #     pass
           # else:
            if new1[i]==s:
             #   print("found s",s," at i=",i,"in new1")
                found1=True
                break
         #   else:
            #    print(s,"not found in new1")

#        print("finding the char",s," in new1 at position j",j)
        if found1:
            for j in range(1,len(new2)-2):
         #       print("looking in new2 in swaps1",swaps1,"k=",k,"=",swaps1[k])
              #  if j>=cut1_choice and j<cut2_choice:
              #      pass
              #  else:
              #  print("not jumping over j=",j,"c1=",cut1_choice,"c2=",cut2_choice)
                
                if new2[j]==swaps1[k]:
          #          print("found swaps1",swaps1[k],"at j=",j,"in new2 k=",k)
            
                    found2=True
                    break
          #      else:
           #         print(swaps1[k],"not found in new2")


                


        if found1 and found2:
            found1=False
            found2=False
            k+=1
       #     print("swapping new2 at i",new1,i,"and new1 at j",new2,j) 
            new1,new2=pmx_swap(new1,i,new2,j)
     #   else:
        #    print("pmx error")

    #print("Anew1=",new1)
    #print("Bnew2=",new2)
    #input("?")

    
    return(new1,new2)



def pmx_loop(mates):
    newpool=[]
    for i in range(0,len(mates)):
        newg1,newg2=pmx(mates[i][0],mates[i][1])
        newpool.append(newg1)
        newpool.append(newg2)
    return(newpool)


#######################################################

def main():

    scaling=100


    class ts(object):
        pass

    ts.xsize=0
    ts.ysize=0
    ts.no_of_stops=0
    ts.stops=[]
    ts.journey=[]
    ts.distance=0
    ts.distances=[]
    ts.totalgenepool_dist=0
    ts.probablilty_table=[]
    ts.probtable_len=0
    ts.wheel=[]
    ts.wheel_len=0
    ts.mates=[]

    ts.poolsize=30
    ts.chromo_length=0
    ts.genepool=[]

    epoch_length=20
    

    ts.xsize=int(input("x size?"))
    ts.ysize=int(input("y size?"))
    ts.no_of_stops=int(input("number of stops?"))

    ts.no_of_stops+=1   #  has to return back to the start

    print("create stops")
    create_stops(ts)
    print("stops=",ts.stops)
    print("create distance table")
  #  for x in range(0,ts.no_of_stops-2):
  #      ts.distance=calc_distance(ts.stops[x],ts.stops[x+1])
  #      print("distance=",x,"->",x+1,"=",ts.distance)

        
    print("create starting genepool")
    ts.genepool=create_starting_genepool(ts)
    print("genes=",ts.genepool)

    bestjourneydist=1000000
    bestjourneyofepoch=[]
    bestjourneyno=0
   
    print("starting....")
    for gencount in range(1,epoch_length+1):


#  genepool is a list of lists
# each list genepool[x] is a journey
# there are ts.poolsize journeys

      #  print("calc total distance for each journey in genepool")
        ts.distances,bestjourneyno=calc_distances(ts)
        ts.totalgenepool_dist=sum(ts.distances)
     #   print("bestjourneyno=",bestjourneyno)
        bestdist=ts.distances
        bestdist.sort()
     #   print("best=",bestdist[0])
        

       # print("genepool dist=",ts.totalgenepool_dist)
        if bestdist[0]<bestjourneydist:
            bestjourneydist=bestdist[0]
            
      #  print("distances=",ts.distances,"total=",ts.totalgenepool_dist)
        print("best journey=",ts.genepool[bestjourneyno],"dist=",bestjourneydist)



        # reproduction
        # total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
        # create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
        # spin the probability_table m times each time yielding a reproduction candidate of the population
        # in this way more highly fit strings have more offspring in the next generation.



        # calc payoff probabilities
        ts.probability_table=[]
        for x in range(0,len(ts.distances)):
            ts.probability_table.append((ts.distances[x]/ts.totalgenepool_dist)*scaling)
    #    print("prob table=",ts.probability_table,"sum=",sum(ts.probability_table))

        # setup a roulette wheel with the smallest distances having the largest slice of the wheel

        ts.probtable_len=len(ts.probability_table)

        mpt=round(mean(ts.probability_table))
        ts.wheel=[]
        n=0 
        while n<=ts.probtable_len-1:
            piesize=round(ts.probability_table[n])
            if piesize<0:
                piesize=0
                
                # invert probabilities
            ts.wheel=ts.wheel+([n] * abs((2*mpt)-piesize))   # invert across mean
            n=n+1
          
    #    print("wheel=",ts.wheel)
        
        ts.wheel_len=len(ts.wheel)
        
        # spin the mating wheel and create a new population
    #    print("spin the mating wheel and create a new population")

        ts.mates=spin_the_mating_wheel(ts)

    #   print("mates=",ts.mates)

    #    print("PMX crossover")
        # turn the list of tuples of lists into a list of lists for the next round
        ts.genepool=pmx_loop(ts.mates)


    #    print("new generation=")

    print("finished:",len(ts.genepool))
        
        
               

##        mates,len_wheel=spin_the_mating_wheel(probability_table,population,r.pop_size,r.direction)  # wheel_len is the size of the unique gene pool to select from in the probability_table
##
##           
##        population=crossover(mates,r.no_of_alleles,individual)   # simple crossover
##          
##
##        population, mutation_count, whichchromo,whichmember=mutate(population,r.no_of_alleles,r.ploidy,r.pop_size,r.mutation_rate)   # 1000 means mutation 1 in a 1000 bits processed
##
##
## 
##        for p in range(0,r.pop_size):    #fill out the express column for the population:
##            population.loc[p,"expressed"]=express(population.loc[p,"chromo1"],population.loc[p,"chromo2"])
##
##        mutations=mutations+mutation_count
##        
##
##
##        total_generations+=1
##        r.total_total_generations+=1
##
##        if r.advanced_diag!="s":    
##            print("")
##
##
##    r.results.write("\nOptimisation TS-GA finished.\n\n")
##    r.results.write("==========================================\n\n")
##    r.outfile.write("\n\nBest Population:")
##    r.outfile.write("\n\n\n"+bestpopulation.to_string())
##    r.outfile.write("\n\n")


    return


###########################################


 

if __name__ == '__main__':
    main()


