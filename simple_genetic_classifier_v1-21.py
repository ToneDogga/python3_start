# Genetic algorithms 
# machine learning using genetic algorithms
# a genetics based classifier system    started 24/8/19
#
#
#  A simple genetic based machine learning classifier system
#
# 1) a rule and message system
# 2) an apportionment of credit system
# 3)a genetic algorithm
#
# a rules and message system
#  index            classifier:message
# 1                      01##:0000
# 2                      00#0:1100
# 3                      11##:1000
# 4                      ##00:0001
#
# matching activity of a classifier system
# environmental message appears on a list 0111
# this messages matches only classifier 1, which then posts its message 0000
# this message matches classifier 2 and 4 which in turn post thier messages
# eventually the process terminates
# 
# a credit apportionment scheme
# bucket brigade algorithm
# an information economy where the right to trade information is bought and sold by classifiers
# environment -> classifiers -> effectors (information consumers)
# auction and a clearinghoue
# when classifiers are matched, they do not directly post thier messages
# the classifers have to participate in a activation auction
# each classifier maintains a record of its net worth or strength S
# each classifier makes a bid B porportional to its strength S
# if is it selected by the auction
# it needs to make its payment though the clearing house
# paying its bid to other classifiers for matching messages rendered
# a matched and activated classifier sends its bid B to those classifers responsible for sending the messages
# that matched the bidding classifiers condition.
# remember we are not searching for one master rule
#
# starting point
#  index            classifier:message     Strength
# 1                      01##:0000          200
# 2                      00#0:1100          200
# 3                      11##:1000          200
# 4                      ##00:0001          200
#   environment                               0  
#
# we assume a bid coefficient of 0.1
# therefore a bid is 0.1 * S = 20
# so message 0111 arrives from the environment
# classifier 1 matches this and bids 20 units and then waits for the next time step
# classifier 1 pays its bid the the party that sent the message, in this case the environment stregnngth is increased by 20 units
# from then on payments go back to other classifiers
# finally at time step 5 a reward comes into the system and is paid to the last active classifier 4.
#
# the auctio n is held in the presence of random noise.  standard deviation.
# so the each bid has a random element of noise added or taken from it.  So they bid is EB , effective bid.
# The winners pay thier bids (without the noise effect) to the clearing house, where payment is divided amoung all
# classifiers responsible for sending a matching (and winning) message
#
# each classifier is taxed to prevent freeeloading
# collect a tex at each time step proportional to the classifiers strength
# 
#
# genetic algorithm
# we need a way of injecting new, possibily better rules into the system
# new rules are created by the standard method of reproduction, crossover and mutation
#  But we don't replace the whole population
# we have a selection proportion
# we also define the number of time steps between each GA call.  This can be fixed or random
# usually we use a roulette wheel based on each classifiers strength as used for fitness
# mutation is slightly different , we change the mutated character to one of the other two with equal probability
# (0=>[1,#],1=>[0,#],#=>[0,1])  we will use -1 to represent # and vice versa
#
# these algorithms can learn boolean functions, a multiplexer.
# 

#
#!/usr/bin/env python
#
from __future__ import print_function
from __future__ import division


import numpy as np
#import numpy.ma as ma
import pandas as pd
import random
import time
import csv
import math
#import pygame
import linecache
import sys
import platform
import itertools
import datetime
import subprocess as sp
from collections import Counter
from statistics import mean
from multiprocessing import Pool, Process, Lock, freeze_support, Queue, Lock, Manager
from os import getpid
import os
import hashlib
import multiprocessing 
from timeit import default_timer as timer
import pickle
#from difflib import SequenceMatcher


################################################################3

# multiplexer
# 6 signals come into the multiplexer
# last two lines are address bits are decoded as a unsigned binary integer
# this address values is then used to work out which of the four remaining leading bits signels
# on the Data or d lines is to be passed thwough to the mulriplexer output as a single bit


def generate_random_signal(no_of_bits):
    data=""
    for i in range(0,no_of_bits):
        data=data+str(random.randint(0,1))
        
    return(data)

def decode_signal(signal,no_of_data_bits,no_of_address_bits):   # multiplexer in form   ddddaa -> o
    address=int(signal[-no_of_address_bits:],2)
  #  print("address=",address)
    data=signal[:no_of_data_bits]
    data2=data[::-1]
  #  print("data=",data)
    result=data2[address]
 #   print("\nsignal=",signal,"address=",address,"data=",data,"result=",result,"\n")
 #   input("?")
    return(result)

########################


def listener(params,q):
    '''listens for messages on the q, writes to file. '''

    pid=os.getpid()
    
    f=open(params["results_file"],"w")
    f.close()
    f=open(params["results_file"],"a")
    print("Queue listener started on:",pid)
    f.write("Queue listener started on:"+str(pid)+"\n")

#   pygame.init()
    while True:

        y=q.get()
        if y == "kill":
        #    print("trying to kill listener process.  Flushing q.  qsize=",q.qsize())
            #f.write("killed\n")
            while not q.empty():
                m=pickle.loads(q.get())
                print("\r"+str(m)) 
                f.write(str(m)+"\n")                  
                f.flush()
            break
        else:
            m = pickle.loads(y)
            print("\r"+str(m))
            f.write(str(m)+"\n")
            f.flush()
              
    
##    print("Hit space to exit")
##    k=""
##    while k!="space":
##        for event in pygame.event.get():
##            if event.type == pygame.KEYDOWN: 
##                k=pygame.key.name(event.key)
##             #   if k=="space":
##             #       break 
##
##    pygame.quit()
    print("Queue listener closed on:",os.getpid())
    f.write("PID:"+str(pid)+" Queue listener closed\n")
   # l_lock.release()
    f.close()
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





####################################

# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the probability_table m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

  

def reproduce(newpopulation,params):
# total all the strength of the genepool
# calculate the % strength of each one
# add that as a column to pandas
# create a list as a roulette wheel with a more entries for stronger genes

    total=newpopulation["strength"].sum()
 #   print("total",total)
    newpopulation["brkdown"]=round((newpopulation["strength"]/total)*params["scaling"],0)
    #newpopulation["brkdown"]=newpopulation["strength"]/total

   # print("newpopulation\n")
   # print(newpopulation)
   # print("brkdown total=",newpopulation["brkdown"].sum())
   # input("?")

    # create wheel list
    # put in index order
    newpopulation.sort_index(inplace=True)
 #   print("newpopulation\n")
 #   print(newpopulation)
    
    wheel_slices=newpopulation["brkdown"].values.tolist()
    piesize=len(wheel_slices)
    wheel=[]
  #  print("wheel slices=",wheel_slices)
    for w in range(0,piesize):
        wheel=wheel+([w] * abs(int(wheel_slices[w])))
  #  print("wheel=",wheel)
    wheel_len=len(wheel)
#    print("wheel len=",wheel_len)

# spin the mating wheel twice.
    giveup_count=0
    giveup=False
    not_found=True
    while not_found:
        if giveup_count<wheel_len*2:
            giveup_count+=1
            first_mate_index=wheel[random.randint(0,wheel_len-1)]
            if first_mate_index in newpopulation.index:  # make sure the indexed record exisits
                not_found=False
        else:
            giveup=True
            break

    giveup_count=0
    if not giveup:
        not_found=True
          # make sure it doesnt mate with itself
        while not_found:
            if giveup_count<wheel_len*2:
                giveup_count+=1
                second_mate_index=first_mate_index
                while second_mate_index==first_mate_index:
                    second_mate_index=wheel[random.randint(0,wheel_len-1)]    
                if second_mate_index in newpopulation.index:   # make sure the indexed record exists
                    not_found=False
            else:
                giveup=True
                break
    if not giveup:
   #     print("mate1=",first_mate_index,"mate2=",second_mate_index)


    # crossover the condition at a random (non-zero) point on both alleles of the mates

        gene1=newpopulation.loc[first_mate_index,"condition"]
        gene2=newpopulation.loc[second_mate_index,"condition"]

        message1=newpopulation.loc[first_mate_index,"message"]   # swap the messages
        message2=newpopulation.loc[second_mate_index,"message"]


    #    print("a1=",gene1,"a2=",gene2)

        gene_len=len(gene1)

        split_point1=random.randint(0,gene_len-1)   #-params["no_of_address_bits"])   # only crossover the data, not the address
        split_point2=random.randint(1,gene_len)   #-params["no_of_address_bits"])   # only crossover the data, not the address
        if split_point2<split_point1: # Swap
            temp=split_point1
            split_point1=split_point2
            split_point2=temp
       
        child1=""
        child2=""
        remain1=gene1[:split_point1]
        swap1=gene1[split_point1:split_point2]
        remain11=gene1[split_point2:]
        
        remain2=gene2[:split_point1]
        swap2=gene2[split_point1:split_point2]
        remain21=gene2[split_point2:]

        child1=remain1+swap2+remain11
        child2=remain2+swap1+remain21

        

        #print("sp1=",split_point1,"sp2=",split_point2,"new1=",child1,"new2=",child2)
        #print("gene1=",gene1,"r1=",remain1,"swap1=",swap1,"r11=",remain11)
        #print("gene2=",gene2,"r2=",remain2,"swap2=",swap2,"r21=",remain21)

        ##    # strength needs to start at the mean of the population so it doesnt die or dominate
        new_strength=newpopulation["strength"].mean()
        specificity1=count_specificity(child1,params["condition_bits"])
        new_row1={"condition":child1,"message":message2,"match_flag":False,"bid":0.0,"ebid":0.0,"strength":new_strength,"specificity":specificity1,"crowding":0.0,"payments":0.0, "taxes":0.0,"receipts":0.0,"winners":False,"brkdown":0.0,"address":"  ","keeping":False}
        newpopulation = newpopulation.append(new_row1, ignore_index=True)
        specificity2=count_specificity(child2,params["condition_bits"])
        new_row2={"condition":child2,"message":message1,"match_flag":False,"bid":0.0,"ebid":0.0,"strength":new_strength,"specificity":specificity2,"crowding":0.0,"payments":0.0, "taxes":0.0,"receipts":0.0,"winners":False,"brkdown":0.0,"address":"  ","keeping":False}
        newpopulation = newpopulation.append(new_row2, ignore_index=True)
     #   print("ns=",specificity1,specificity2)
     #   print("2 new genes born. len=",len(newpopulation))

    else:
        print("\ngave up finding a mating pair.  Population too small")
    
    
 #   input("?")
    return(newpopulation)


def mutate(population,params):
# mutation is slightly different , we change the mutated character to one of the other two with equal probability
# (0=>[1,#],1=>[0,#],#=>[0,1])  we will use -1 to represent # and vice versa
#    
    
    mean=population["strength"].mean()  

    temppop=population.query("winners==False & strength<="+str(mean)) #,inplace=True)
    templen=len(temppop)
  #  print("mutate.  templen=",templen)
    # pick a random row.  don't mutate a winner
    if templen>0:
        row=random.randint(0,templen-1)
        
        gene=temppop.iloc[row].condition
        ginx=temppop.iloc[row].name
     #   print("gene to mutate",gene,"from row number",row,"index=",ginx)
        # pick a random position in a condition which has not won in the last 3 cycles
        genelen=len(gene)   #-params["no_of_address_bits"]   # only mutate the data, not the address.
        newgene=""
        pos=random.randint(0,genelen)
        for elem in range(0,genelen):
            if elem==pos:
                newgeneno=random.randint(0,1)
                if gene[pos]=="0":
                    if newgeneno==0:
                         newgene=newgene+"#"
                    else:
                         newgene=newgene+"1"

                elif gene[pos]=="1":
                    if newgeneno==0:
                         newgene=newgene+"0"
                    else:
                         newgene=newgene+"#" 

                elif gene[pos]=="#":
                    if newgeneno==0:
                         newgene=newgene+"0"
                    else:
                         newgene=newgene+"1" 
                   
            else:    
                newgene=newgene+gene[elem]
                
        population.loc[ginx,"condition"]=newgene
    else:
        pass
        #print("no mutation, population is all winners.")
  #  print("population=",population)
  #  input("?")
    return(population)

  
def count_file_rows(filename):
    with open(filename,'r') as f:
        return sum(1 for row in f)


def read_env_file_in(filename):
    environment=pd.read_csv(filename,dtype={"condition":np.str,"message":np.str,'match_flag': np.bool,"winners":np.bool})
    print(environment)
  #  input("?")
    return(environment)




def read_pop_file_in(filename,condition_type,message_type):
    population=pd.read_csv(filename,dtype={"condition":np.str,"message":np.str,'match_flag': np.bool,"winners":np.bool})
  #  population=pd.read_csv(filename,dtype={"condition":condition_type,"message":message_type,'match_flag': np.bool,"winners":np.bool})


    print(population)
  #  input("?")
    return(population)


def return_a_row_from_env(filename,size):
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        random_line=random.randint(0,size-1)+1
        try:
            l=linecache.getline(filename,random_line).rstrip()
          #  print("return a row from env",random_line,l)
            if l:
                p=l.split(',')  # add an extra element to the row because the concatno field is not in the payoff file
             #   print("linecache p=",p)
                if p:
                    return(p,True)
                #    return(list(itertools.chain(*p)),True)
                else:
                    return(["0","0"],False)
            else:
                return(["0","0"],False)
               
 
            
        except IndexError:
            print("\nreturn a row from env using line cache. index error at random line",random_line)
            return(["0","0"],False)
        except ValueError:
            print("\nreturn a row from env using line cache. value error random line=",random_line) 
            return(["0","0"],False)
        except IOError:
            print("\nreturn a row from env using line cache. IO error")
            return(["0","0"],False)



    

def extract_messages_to_list(population):
    sortedpop=population.sort_values("strength",ascending=False)  # message list comes out with the strongest message first on the list
 #  message_list=list(set(sortedpop["message"].values.tolist()))    # removes non-unique list elements
    message_list=list(sortedpop["message"].unique())    # removes non-unique list elements
    if len(message_list)>3:
        del message_list[3:]
   # print("\ne=",sortedpop,"\nmess=",message_list)
   # input("?")
    return(message_list)

def get_message_off_message_list(message_list,params):
    if len(message_list)>0:
        message=message_list[0]
        params["messages_applied"].append(message)
        del message_list[0]
        return(message)
    else:
        return("")


def add_message_to_message_list(messages,message_list):
 #   message_list.append(messages)
 #   return(list(itertools.chain(*message_list)))
     if len(message_list)>6:
        del message_list[6:]

     return(message_list+list(messages))


def create_a_unique_condition(population,no_of_condition_bits):
    not_unique=True
    while not_unique:
        cl=""    
        for alleles in range(0,no_of_condition_bits):
            bit=random.randint(-1,1)
            if bit==-1:
                b="#"
            else:
                b=str(bit)
            cl=cl+b

        not_unique= not (population[population["condition"]==cl].empty)
    return(cl)


def create_a_non_unique_condition(population,no_of_condition_bits):
   # not_unique=True
   # while not_unique:
    cl=""    
    for alleles in range(0,no_of_condition_bits):
        bit=random.randint(-1,1)
        if bit==-1:
            b="#"
        else:
            b=str(bit)
        cl=cl+b

    #    not_unique= not (population[population["condition"]==cl].empty)
    return(cl)



def create_a_message(no_of_message_bits): 
    ml=""    
    for alleles in range(0,no_of_message_bits):
        ml=ml+str(random.randint(0,1))   
    return(ml)   

def numberToBase(n, b):
    if n == 0:
        return(["0"])
    digits = []
    while n:
        digits.append(str(int(n % b)))
        n //= b
    return(digits[::-1])

 
def create_entire_classifier_space(population,params):
    pid=os.getpid()
    psize=(3**params["condition_bits"])*(2**params["message_bits"])
    csize=3**params["condition_bits"]
    msize=2**params["message_bits"]
    print("PID:{",pid,"} Entire classifer space being created.",psize,"records.")             
    pos=0          
    for c_allele in range(0,csize):
        if pos%100==0:
            print("PID:{",pid,"} Progress:[",int((pos/psize)*100),"%].")             
        cbit="".join(numberToBase(c_allele,3)).zfill(params["condition_bits"])   # base 3
        cl=cbit.replace("2", "#")   #.zfill(params["condition_bits"])
        for mcount in range(0,msize):
            ml=str("{00:b}".format(mcount)).zfill(params["message_bits"])
         #   print(cl,":",ml) 
            population.loc[pos,"condition"]=cl            
            population.loc[pos,"message"]=ml
            pos+=1
    return(population)

def create_classifiers(population,condition_bits,size):
    population_len=size
    pid=os.getpid()
    for p in range(0,population_len):
      #  cl=create_a_unique_condition(population,condition_bits)
        cl=create_a_non_unique_condition(population,condition_bits)
        

        population.loc[p,"condition"]=cl
        if p%100==0:
            print("PID:{",pid,"} Condition creation process:",int(p/population_len*100),"%.     ")
    return(population)   
 

def create_messages(population,message_bits,size):
    population_len=size
    pid=os.getpid()
    for p in range(0,population_len):
        ml=create_a_message(message_bits)
        #ml=""    
        #for alleles in range(0,message_bits):
        #    ml=ml+str(random.randint(0,1))
           
        population.loc[p,"message"]=ml
        if p%100==0:
            print("PID:{",pid,"} Message creation process:",int(p/population_len*100),"%.      ")
        
    return(population)


##def similarity(row,condition):
##  #  from difflib import SequenceMatcher
##      return(SequenceMatcher(None, row["condition"], condition).ratio())
##    
##
def cleanup_crowding(temppop,params):
 #   population=population.sort_values("condition",ascending=False)
    temppop2=temppop
    temppop2["address"]=temppop["condition"].str[-params["crowding_depth"]:]
    temppop2["keeping"]=False
    temppop2=temppop2.query("winners==False")
    # find the weakest in each crowd for each address - low strength, low specificity
    temppop2=temppop2.sort_values(["address","strength","specificity"],ascending=[True,False,False])    #.drop_duplicates(subset=["address"])
    temppop2["keeping"]=temppop.duplicated(["address"],keep="first")
  #  print("\ncrowding.  sorted by temppop2=\n",temppop2.to_string())
    temppop3=temppop.drop(temppop2.loc[temppop2["keeping"]==False].index)   #name)  #index)
 #   print("\ncrowding3.  sorted by temppop3=\n",temppop3.to_string())
  #  input("?")
   
    return(temppop3)




def update_specificity(population,params):   # a very specific condition has no #'s in it.  It means it gets a higher reward if it wins
    #p=population["condition"]   #.ma.count("#",axis=1))
    #q=p.np.count("#",axis=0)
    


  #  p =population.condition.applymap(lambda x: str.count(x, '#'))
 #   q=population.condition.str.count("#")
 #   print("pop=\n",q,population)
    population["specificity"]=(params["condition_bits"]-population.condition.str.count("#"))/params["condition_bits"]
    return(population)

def update_strengths(population):
    # strength(t+1)=strength(t)-payments(t)-taxes(t)+receipts(t)
    population["strength"]=population["strength"]-population["payments"]-population["taxes"]+population["receipts"]
    return(population)

##def bid_adjust(row,coeff1,coeff2):
##    return(row["strength"]*coeff1+row["strength"]*row["specificity"]*coeff2)   #This prevents non specific rules dominating

def bid(population,params):
    # each condition that matching the message makes a bid proportional to thier strength ( bidcoeff)
    population["bid"]=population["strength"]*params["bidcoeff1"]+population["strength"]*population["specificity"]*params["bidcoeff2"]
    mask=population.match_flag == False
  #  column_name="match_flag"
 #   population.loc[mask,"bid"]=population.apply(bid_adjust,axis=1,args=[3])
    population.loc[mask,"bid"]=0

    return(population)
    
##def noise(row,mu,sigma):  # normal distribution
##    return(row["bid"]+np.random.normal(0.0,params["bidsigma"]))


def effective_bid(population,params):
    # there is a tax on each transaction which reduces the bid to an effective bid which is ebid
    population["ebid"]=population["bid"]+np.random.normal(0.0,params["bidsigma"])
    mask=population.match_flag == False
    population.loc[mask,"ebid"]=population["bid"]
    return(population)




def tax(population,params):

    if params["bidtax"]==0:
        pass
    else:
    # apply bidtax to bidders only
        population["taxes"]=population["taxes"]+population["bid"]*params["bidtax"]
        mask=population.match_flag == False
        population.loc[mask,"taxes"]=0


    if params["lifetax"]==0.0:
        pass
    else:
  # apply exisitance tax to everyone
        population["taxes"]=population["taxes"]+population["strength"]*params["lifetax"]
        
    
    return(population)



def save_winners(temppop):

    winner=temppop.index.tolist()[:1]   # the top 1 only



    return(winner)



def mark_winners(winner,temppop):
   # temppop["winners"]=False

    temppop["winners"]=temppop["winners"].isin([winner])
  #  print("maerk winners temppop=",temppop)
  #  input("?")
    return(temppop)




def clearing_house(temppop,params):
    # randomise the order to prevent always choosing first one when there are multiple identical ebids that won't sort out
    previous_winner=0
   # winner_before_previous=[]

    temppop = temppop.sample(frac=1)   #.reset_index(drop=True)

    params["winner_list"].append(save_winners(temppop[temppop["match_flag"]==True].sort_values("ebid",ascending=False)))   # a list of a list of winners sorted in order by the size of ebid highest to lowest

    current_winner=params["winner_list"][-1]
  #  print("winner list=",params["winner_list"],"current winners=",current_winner)
    # the winners list of lists contains the sorted order of the winners
  #  print("spoils=",spoils_function(params["winner_spoils"],1000))
  #  input("?")

    if len(params["winner_list"])>1:
        previous_winner=params["winner_list"][-2]
    #    print("previous winner",previous_winner)
     #   if len(params["winners"])>2:
      #      winner_before_previous=params["winners"][-3]
      #      print("winner before previous",winner_before_previous)

      #  else:
       #     winner_before_previous=[]
      #      print("winner before prevous list empty")
    else:        
        previous_winner=0
     #   print("previous winners list empty")
    params["old_winners"].append(previous_winner)    #+winner_before_previous
  #  params["all_winners"].append(current_winner+previous_winner)    #+winner_before_previous


############

    
    temppop=mark_winners(current_winner,temppop)   # set the winners flag for each row for current winners to divvy up the payment
  #  print("all winners=",params["all_winners"],"previous winner-",previous_winner,"winner_before previous=",winner_before_previous)

    temppop2=temppop.query("match_flag==True")
    if len(temppop2)>0:
        value=temppop2["bid"].max()  # payment value.  pay the actual bid, not the effective bid
    else:
        value=0.0

###############################################################
        # divvy up payment.  Find the highest ebid
  #  print("\npayment for ",current_winners," of",value)
   
  #  for elem in range(0,len(current_winners)):   #len(params["winner_spoils"])):
  #      temppop.loc[current_winners[elem],"payments"]=params["winner_spoils"][elem]*value


   # print(temppop)
   # input("?")
 

###############################################
# update receipts based on previous cycle winners
  #  print("\nreceipts for ",previous_winner," of ",value*params["minus1factor"])


    if previous_winner:   # and len(winner_before_previous)>0:

                # divvy up payment.  Find the highest ebid
  
   #     for elem in range(0,len(current_winners)):   #len(params["winner_spoils"])):
       #     print("payment=",elem,current_winners,value)
        temppop.loc[current_winner,"payments"]=params["winner_spoils"]*value


        temppop=mark_winners(previous_winner,temppop)   # set the winners flag for each row for last cycles winners (1 cycle back)

     #   for elem in range(0,len(previous_winner)):   #params["winner_spoils"])):
        #    print("receipt=",elem,previous_winner,value)
        try:
            temppop.loc[previous_winner,"receipts"]=params["winner_spoils"]*value*params["minus1factor"]*params["reward"]
        except KeyError:
            # no receipt so
            # undo payment.
            print("key error on pw=",previous_winner,"in",temppop)
            temppop.loc[current_winner,"payments"]=0.0
            
       # print("pw=",previous_winner)
         

    return(temppop)




def check_signal_and_output(row,signal,output,rowindex,cms,cms_flag,mms,mms_flag,condition_bits):
  #  if row["condition"].isnan():
   #     print("\nmatching signal row empty")
    #    return(False)

  #  print("rc=",row["condition"],"mess=",row["message"])
    condition=row["condition"]
    message=row["message"]

# test in environment
  #  output=decode_signal(signal,no_of_data_bits,no_of_address_bits)

#  The multiplexer matchs condition ddddaa
#with line 0 the furthest right


    
    try:
        alen=len(condition)
        slen=len(signal)
    except TypeError:
        alen=0
        slen=0
        print("\ntype error condition",condition)
        input("?")
        return(False)
        
    clen=min(alen,slen)
  #  print("signal=",signal,"condition=",condition,"clen=",clen)

    try:
        mlen=len(message)
        tlen=len(output)
    except TypeError:
        mlen=0
        tlen=0
        print("\ntype error message",message)
      #  input("?")
        return(False)





    olen=min(mlen,tlen)
 
    #print("condition=",condition,"signal=",signal,"clen=",clen)
    #print("message=",message,"output=",output,"olen=",olen)
    
 #   print("check alen=",alen,",mlen=",mlen)
 
  #  print("r=",r)

    #condition_match_specificity
    cmatch=False
    cmatch_count=0
    for condition_elem in range(0,clen):  #clen-1,-1,-1):   # start with the left most right most
  #      print("e=",message_elem)

        m=signal[condition_elem]
        a=condition[condition_elem]   # note classifer is on column 0
   #     print("check condition e=",condition_elem,"m=",m,"a=",a)
        if a=="#" or a==m:
           cmatch=True
           cmatch_count+=1
        else:
           cmatch=False
           if cms_flag:
               break
           
    if cmatch_count>=clen*cms:       #params["condition_match_specificity"]
        cmatch=True
    else:
        cmatch=False

   # print("conditions matching",cmatch,"cmatch_count=",cmatch_count,"clen=",clen)
    mmatch=False
    mmatch_count=0
    for message_elem in range(0,olen):   #olen-1,-1,-1):   # start with the left most
  #      print("e=",message_elem)

        m=message[message_elem]
        a=output[message_elem]   # note classifer is on column 0
     #   print("check message e=",message_elem,"m=",m,"a=",a)
        if a=="#" or a==m:
           mmatch=True
           mmatch_count+=1
        else:
           mmatch=False
           if mms_flag:
               break
        
        
    if mmatch_count>=olen*mms:          #params["message_match_specificity"]
        mmatch=True
    else:
        mmatch=False



  #  print("message matching",mmatch,"mmatch count=",mmatch,"olen=",olen)

    match=cmatch and mmatch

   # if match:
   #     pass
   #     print("both matching",match)   #,"row=",rowindex)    
 #   print("signal",signal," output=",output,"condition=",condition,"message=",message,"match=",match,"rowno=",row.name,"row index=",rowindex[row.name])
 #   input("?")
    return(match)



##def check_for_nans_in_rows(population):
##    df1 = population[population.isna().any(axis=1)]
##    print("check for nans=",df1)
##    return


def find_matches(signal,output,df,condition_bits,message_bits):
    df["matching"]=True
  
    c=min(condition_bits,len(signal))-1   #len(df["condition"].max())-1
    m=min(message_bits,len(output))-1   #len(df["condition"].max())-1
  
    while c>=0 and df.size>0:
        s=signal[c]     
        newcol = df.condition.str[c]

        df["matching"]=(df["matching"] & ((newcol==s) | (newcol=="#")))
    
        c-=1

    df["matching"]=(df["matching"] & (df.message.str[-m]==output[-m]))
    df["match_flag"]=(df["matching"]==True)
    df = df.replace(np.nan, False, regex=True)
    return(df)

                                    


def kill_a_non_winning_random_condition(population,params):
# delete a non matching, non winner at random
  #   print("killing the least fit")   
    population=mark_winners(params["all_winners"],population)   # set the winners flag for each row for current winners to divvy up the payment

    # cant kill a winner, previous winner (1 cycle) or previous winner (2 cycles)  they are marked as winners in the dataframe field
  #  population.drop(population['strength'].idxmin())
  
  #  mean=population["strength"].mean()
    started=str(params["starting_strength"])
    temppop=population.query("winners==False & match_flag==False & strength="+started)    #str(mean))  #,inplace=True)
 #   print("query before delete=",temppop)
 #   input("?")
    if len(temppop)>0:
       # temppop2=temppop.sample(n=1)
        temppop2 = temppop.loc[np.random.choice(temppop.index, size=1)]
       # temppop2
        #population=population.drop(temppop['strength'].nsmallest(3, 'strength', keep='last'))
        #population=temppop.drop(temppop.nsmallest(3, columns='strength'))
        temppop3=population.drop(temppop2.index)
     #   population["keeping"]=population.loc[temppop['strength'].idxmin()]
        return(temppop3) 
    else:
        return(population)
     

 

def kill_the_most_unfit_condition(population,params):
 #   print("killing the least fit")   
    population=mark_winners(params["all_winners"],population)   # set the winners flag for each row for current winners to divvy up the payment

    # cant kill a winner, previous winner (1 cycle) or previous winner (2 cycles)  they are marked as winners in the dataframe field
  #  population.drop(population['strength'].idxmin())
    temppop=population.query("winners==False & match_flag==False")  #,inplace=True)
 #   print("query before delete=",temppop)
 #   input("?")
    if len(temppop)>0:        
        #population=population.drop(temppop['strength'].nsmallest(3, 'strength', keep='last'))
        #population=temppop.drop(temppop.nsmallest(3, columns='strength'))
        temppop2=population.drop(temppop['strength'].idxmin())
     #   population["keeping"]=population.loc[temppop['strength'].idxmin()]
        return(temppop2) 
    else:
        return(population)

  #  df.nsmallest(3, columns = 'a')
       # population=(population.query("winners==False",inplace=True)).drop(population['strength'].idxmin())

  #  population=population.query("winners==False | winners==True")
 #   print("after delete pq=",temppop2)
 #   input("?")

 #   return(temppop2)
    

def reset_fields_for_next_cycle(population):
     population["match_flag"]=False
     population["bid"]=0.0
     population["ebid"]=0.0
     population["payments"]=0.0
     population["taxes"]=0.0
     population["receipts"]=0.0
     population["winners"]=False
     population["brkdown"]=0.0
     population["address"]=""
     population["keeping"]=False


     return(population)



def timer(count,params):
    # this times the mutations, environmental messages, deaths and births

    mutate_flag=False
    env_flag=False
    crowding_flag=False
    feedback_flag=False
    death_flag=False
    birth_flag=False

    if count%params["mutrate"]==0:
        mutate_flag=True

    if count%params["envrate"]==0:
        env_flag=True

    if count%params["crowding_rate"]==0:
        crowding_flag=True

    if count%params["feedrate"]==0:
        feedback_flag=True


    if count%params["deathrate"]==0:
        death_flag=True
        
    if count%params["birthrate"]==0:
        birth_flag=True



    return(mutate_flag,env_flag,crowding_flag,feedback_flag,death_flag,birth_flag)


##    individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
##    poparray = np.zeros(params["pop_size"], dtype=individual) 
##    population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")


# set up numpy structure and pandas dataframe
# strength(t+1)=strength(t)-payments(t)-taxes(t)+receipts(t)

def setup_df(params):
    condition_len="S"+str(params["condition_bits"])
    message_len="S"+str(params["message_bits"])    
 
    if params["import_flag"]:
        population=read_pop_file_in(params["starting_pop_filename"],condition_len,message_len)
   #     print(population.dtypes)
    #    population = pd.DataFrame("match_flag", dtype=np.bool)
     #   population["match_flag"].dtype=np.bool
        #individual = np.dtype([("condition",condition_len),("message",message_len),("bid",'f'),("ebid",'f'),("match_flag",'b'),("strength",'f'),("payments","f"),("taxes","f"),("receipts","f")])

    else:    
      #  condition_len="S"+str(params["condition_bits"])
      #  message_len="S"+str(params["message_bits"])    
        
        individual = np.dtype([("condition",condition_len),("message",message_len),("bid",'f'),("ebid",'f'),("match_flag",'b'),("strength",'f'),("specificity",'f'),("crowding",'f'),("payments","f"),("taxes","f"),("receipts","f"),("brkdown","f"),("winners","b"),("address","S2"),("keeping","b")])

        if params["create_unique_classifiers_flag"]:
            poparray = np.zeros((3**params["condition_bits"])*(2**params["message_bits"]), dtype=individual) 
            population = pd.DataFrame({"condition":poparray["condition"],"message":poparray["message"],"bid":poparray["bid"],"ebid":poparray["ebid"],"match_flag":poparray["match_flag"],"strength":poparray["strength"],"specificity":poparray["specificity"],"crowding":poparray["crowding"],"payments":poparray["payments"],"taxes":poparray["taxes"],"receipts":poparray["receipts"],"winners":poparray["winners"],"brkdown":poparray["brkdown"],"address":poparray["address"],"keeping":poparray["keeping"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")
            population=create_entire_classifier_space(population,params)
            population.to_csv(params["starting_pop_filename"],columns=["condition","message"],index=False)
        else:
            poparray = np.zeros(params["size"], dtype=individual) 
            population = pd.DataFrame({"condition":poparray["condition"],"message":poparray["message"],"bid":poparray["bid"],"ebid":poparray["ebid"],"match_flag":poparray["match_flag"],"strength":poparray["strength"],"specificity":poparray["specificity"],"crowding":poparray["crowding"],"payments":poparray["payments"],"taxes":poparray["taxes"],"receipts":poparray["receipts"],"winners":poparray["winners"],"brkdown":poparray["brkdown"],"address":poparray["address"],"keeping":poparray["keeping"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")
            population=create_classifiers(population,params["condition_bits"],params["size"])
            population=create_messages(population,params["message_bits"],params["size"])

    
    population["strength"]=params["starting_strength"]   #+np.random.normal(0.0,1)
    population["match_flag"]=False
    population["bid"]=0.0
    population["ebid"]=0.0
    population["payments"]=0.0
    population["taxes"]=0.0
    population["receipts"]=False
    population["winners"]=False
    population["address"]=""
    population["brkdown"]=0.0
    population["keeping"]=False
    population["specificity"]=0.0
    population["crowding"]=False


    #print(population.to_string())
    
    
    
    #input("?")

    return(population)



#def startup(params):

   # df=setup_df(params)   # setup dataframe
    
  #  message_temp=[]

  #  total_inputs=params["no_of_address_bits"]+params["no_of_data_bits"]
  #  print("Address bits=",params["no_of_address_bits"]," Data bits=",params["no_of_data_bits"]," Total number of inputs=",total_inputs)

  #  params["mplx_data"]=multiplexer_init(params["no_of_address_bits"],params["no_of_data_bits"])
  #  print("mplx_data=",params["mplx_data"])
    # create first environmental message (from multiplexer)
   # message_temp=add_message_to_message_list(multiplexer(params["no_of_address_bits"],params["no_of_data_bits"]),message_temp)

    # OR read in from environment CSV file
    #message_temp=add_message_to_message_list([read_env_file_in(params["env_filename"])],message_temp)

   # return(df)


#####################################################

def classifier_mp_GA(params):   # multiprocessing stuff will go here later
    
    tmp=sp.call(params["cls"],shell=True)  # clear screen 'use 'clear for unix, cls for windows

    print("\n\nGeneral Classifier genetic algorithm written by Anthony Paech 15/9/19")
    print("=====================================================================")

    print("Parameters=\n",params)
    if params["create_unique_classifiers_flag"]:
        print("\n\nPlease wait. Creating entire space of",(3**params["condition_bits"])*(2**params["message_bits"]),"unique classifiers.")
    else:  
        print("\n\nPlease wait. Creating",params["size"],"random classifiers.")
    
    manager = multiprocessing.Manager()
    q = manager.Queue()    
   
    cpus = multiprocessing.cpu_count()

    # delete old csv outfile
    g=open(params["outfile_csv"], 'w')
    g.close()
   # dataout_as_bytes = pickle.dumps(dataout)
   #print("dataout as bytes=",dataout_as_bytes)
    #print(type(x_as_bytes))

   # add_to_queue(dataout_as_bytes,q)   # send results of run to file and print

    add_to_queue(pickle.dumps("Multiprocessing starting. cpus="+str(cpus-1)+"\n"),q)

 #   print("Multiprocessing starting. cpus=",cpus)
 
    multiple_results=[]
    
    i=0
    with Pool(processes=cpus) as pool:  # processes=cpus-1
         #put listener to work first
        watcher = pool.apply_async(listener, args=(params,q, ))

    
        for i in range(0,cpus-1):
            params["bidcoeff1"]=params["bidcoeff_mp"][i][0]
            params["bidcoeff2"]=params["bidcoeff_mp"][i][1]
            add_to_queue(pickle.dumps("Specificity="+str(params["bidcoeff2"])+" \n"),q)
            
            multiple_results.append(pool.apply_async(classifier_GA,args=(params,q, )))  # stops, journey and poolsize, epoch length and name of q
 
            
        for res in multiple_results:
            result=res.get(timeout=None)
            res.wait()



        print("\n\nGenerate results finished")
        
        add_to_queue(pickle.dumps("\n\nGenerate results finished\n"),q)

        print("killing listener")
        q.put("kill")
        result=watcher.get(timeout=None) 
        watcher.wait()

        
    print("try to close pool")
    pool.close()
    print("pool closed.  trying to join() pool")
    pool.join()
    print("pool closed. join() complete")


    return


def classifier_GA(params,q):
    pid=os.getpid()
    clock_start=time.process_time()
    count=1
    population=setup_df(params)   # setup dataframe

   # print("\n\nGeneral Classifier genetic algorithm written by Anthony Paech 15/9/19")
   # print("=====================================================================")

    #print("\n\nPID:",pid,"Starting classifiers")
  #  print(population.loc[:,["condition","message"]].to_string())
  #  print(population.loc[:,["condition","message"]].to_string())

   # print("\n\n")

    msg="\n\nPID:"+str(pid)+" Input Parameters=\n"+str(params)+" \n\n"
    add_to_queue(pickle.dumps(msg),q)

    population=update_specificity(population,params)         # the more general the condition (ie the more #'s) the weaker the reward for winning       


    if params["use_big_env_file"]:
        env_size=count_file_rows(params["env_filename"])
        print("Use linecache on environment file:",params["env_filename"],"size=",env_size)
    else:                             
        environmentin=read_env_file_in(params["env_filename"])
        

   # print("\n\n",population.to_string())
   # input("?")

      #  params["message_bits"]=len(environmentin["message"].max())
      #  params["condition_bits"]=len(environmentin["condition"].max())
    print("condition bits=",params["condition_bits"],"message bits=",params["message_bits"])
    

    while count<=params["epoch"] and len(population)>params["min_pop"]:   

        mutate_flag,env_flag,crowding_flag,feedback_flag,death_flag,birth_flag=timer(count,params)   # time the actions based on the count


        if env_flag:   # inject an environmental signal
            if params["use_big_env_file"]:
                envline,success=return_a_row_from_env(params["env_filename"],env_size)
                if success:
                    signal=str(envline[0]).strip()
                    output=str(envline[1]).strip()
                #    print("signal=",signal,"output=",output)
                else:
                    print("return a row from env",params["env_filename"],"failure")
            else:                     
                rand=np.random.choice(environmentin.index, size=1)
                signal= str(environmentin.loc[rand].condition.max())   #,["condition","message"]]
                output= str(environmentin.loc[rand].message.max())   #,["condition","message"]]

   # msg="\n\nPID:"+str(pid)+" Final classifier solution\n"
   # add_to_queue(pickle.dumps(msg),q)        
     #   signal=generate_random_signal(params["no_of_data_bits"]+params["no_of_address_bits"])
    #    print("\nsignal=",signal)
     #   output=decode_signal(signal,params["no_of_data_bits"],params["no_of_address_bits"])
    #    print("output=",output)
        
       # params["message_list"].append(signal)   # add 

    #   print("len5=",len(population))
        #message=get_message_off_message_list(params["message_list"],params)
     #   print("len7=",len(population))
 #       matches=find_matches(signal,output,population)
      #  population=find_matches_old(signal,output,population,params)
        population=find_matches(signal,output,population,params["condition_bits"],params["message_bits"])
 
     #   print("\n",population.to_string())
     #   input("?")

     #   print("len8=",len(population))
    #    print("message=",message)
    #    print("matches=\n",matches)
     #   input("?")

        
  #      print("message from environment detector, message_list=",params["message_list"])   # insert the env message first on the list
    #    if len(params["message_list"])==0:
    #    else:    
    #        params["message_list"]=add_message_to_message_list(multiplexer(generate_signal(,params["mplx_data"]),params["message_list"]) # random starting message
    #    else:
 #       params["message_list"].insert(0,multiplexer(message_list,params["mplx_data"]))   #   send a message to the multiplexer, process the output as a message
  #  else:


  
  #      if feedback_flag:  
  #          params["message_list"]=add_message_to_message_list(extract_messages_to_list(matches),params["message_list"])




      #  print("len6=",len(population))
      #  print("message list=",params["message_list"])
        #if len(params["message_list"])>0:
          #      params["message_list"].insert(0,multiplexer(messageparams["no_of_address_bits"],params["no_of_data_bits"]))   #        else:
     
                #print("message list empty")
                #break
       # if not params["condition_match_sensitivity_test_disabled"]:
       
      #  population=update_specificity(population,params)         # the more general the condition (ie the more #'s) the weaker the reward for winning       

       
 
          #  print("len=",len(population))
        population=bid(population,params)
     #   print(population)
        population=effective_bid(population,params)
     #   print("len2=",len(population))
        population=tax(population,params)
     #   print(population)
     #   input("?")
     
     #   add_to_queue(pickle.dumps("PID"+str(pid)+" Before clearing house\n\n"+population.to_string()+"\n\n"),q)

        population=clearing_house(population,params)


       # add_to_queue(pickle.dumps("PID"+str(pid)+" After clearing house\n\n"+population.to_string()+"\n\n"),q)

   #     print(population.to_string())
   #     print(population.sort_values("strength",ascending=False).to_string())
      #  input("?")
       
     #   print(count,":winners=",params["winner_list"])
        
     #   print(count,":all winners=",params["winners"])
      #  input("?")
#        print("messages applied=",params["messages_applied"])
      #  input("?")
     #   print("message list=",params["message_list"])
##        if False: #True: #False: 
##            print(population.sort_values("strength",ascending=False).to_string())
##            print("signal=",signal,"output=",output)
##            input("?")



        population=update_strengths(population)
    #    print(population.sort_values("strength",ascending=False).to_string())
    #    input("?")


    #    print(population.to_string())
     #   input("?")
   #     print(population.to_string())
##        print(population.sort_values("strength",ascending=False).to_string())
##        print("signal=",signal,"output=",output)
##        input("?")
      #  print("len before=",len(population),"pop before=",population)
     #   add_to_queue(pickle.dumps("PID:"+str(pid)+" After update strengths\n\n"+population.to_string()+"\n\n"),q)


        nan_rows = population[population.isnull().any(1)]
        if len(nan_rows)>0:
            pn="\nnan rows to drop="+str(nan_rows.to_string())+"\n"
            add_to_queue(pickle.dumps(pn),q)
            population.dropna(axis='rows',inplace=True)
         #   population=population.drop(nan_rows)   #[population.isnull().any(1)]



       
        if params["death_enabled"] and death_flag and count>params["deathstart"] and count<params["deathfinish"]:
            pass
          #  print("death flag count=",count)

           # population=kill_the_most_unfit_condition(population,params)

            
            population=kill_a_non_winning_random_condition(population,params)
            population=kill_a_non_winning_random_condition(population,params)
            population=kill_a_non_winning_random_condition(population,params)
            population=kill_a_non_winning_random_condition(population,params)
            population=kill_a_non_winning_random_condition(population,params)
##            population=kill_a_non_winning_random_condition(population,params)
##            population=kill_a_non_winning_random_condition(population,params)

 
           
          #  print("duplicates:",population[population.duplicated(subset=["condition","message"],keep=False)])

           # remove duplicates
##            population=population.sort_values("strength",ascending=False)
##       #     temppop=population[population.duplicated(subset=["condition","message"],keep=False)]
##            df2=population.drop_duplicates(subset=["condition"])   # remove non-unique conditions and messages
##            population=df2

         #   population=kill_a_non_unique_condition(population,params)
      #      print("death len after=",len(population),"pop after=",population)


      
        # crowding factor.  Kill a condition which is very similar to others
        if params["crowding_enabled"] and crowding_flag and count>params["crowdingstart"] and count<params["crowdingfinish"]:
            population=cleanup_crowding(population,params)   # the similarity of each condition to the rest of the population
            # this is used to select similar conditions and kill them 



        if params["mutation_enabled"] and mutate_flag and count>params["mutstart"] and count<params["mutfinish"]:
         #   print("mutate flag")
            population=mutate(population,params)

            

        if params["birth_enabled"] and birth_flag and count>params["birthstart"] and count<params["birthfinish"]:
          #  print("birth flag")
            population=reproduce(population,params)   # make two new random conditiosn and messages
        #    print(population.sort_values("strength",ascending=False).to_string())
        #    input("?")

       #     print("birth len after=",len(population),"pop after=",population)


##        print("\nbefreo. len=",len(population))
##        print(population.sort_values("strength",ascending=False).to_string())
##        input("?")


##        # remove duplicates
   #     population=population.sort_values("strength",ascending=False)
        
        if params["drop_duplicates_enabled"]:
            population=population.sort_values("strength",ascending=False)
            df2=population.drop_duplicates(subset=["condition","message"])   # remove non-unique conditions and messages
            population=df2
 


    #    print("\nafter len=",len(population))
    #    print("duplicates=",temppop.sort_values("strength",ascending=False).to_string())
    #    input("?")

        match_count=population[population.match_flag==True].count()["condition"]  # df[df.a > 1]
    #    print(population.to_string())
        population=reset_fields_for_next_cycle(population)
      #  print(population.sort_values("strength",ascending=False).to_string())
      #  print("len=",len(population))
      #  input("?")

     #   check_for_nans_in_rows(population)   

        #    print(population.to_string())
     #   input("?")
    #    print("\rgen count=",count,"/",params["epoch"],"conditions",population["condition"].count(),"no. matched cond",match_count,"median strength=",round(population["strength"].median(),1),"max strength=",round(population["strength"].max(),1),end="\r",flush=True)  

        if count%params["report_freq"]==0:
            msg="PID{"+str(pid)+"} gen="+str(count)+"/"+str(params["epoch"])+" conds="+str(population["condition"].count())+" mtched conds="+str(match_count)+" median s="+str(round(population["strength"].median(),1))+" Mut Rate:"+str(params["mutrate"])+" max s="+str(round(population["strength"].max(),1))
            msg_as_bytes = pickle.dumps(msg)
            add_to_queue(msg_as_bytes,q)
     #   print("PID:",pid,"gen count=",count,"/",params["epoch"],"conditions",population["condition"].count(),"no. matched cond",match_count,"median strength=",round(population["strength"].median(),1),"max strength=",round(population["strength"].max(),1),flush=True)  
        count+=1  

    msg="\n\nPID:"+str(pid)+" Final classifier solution\n"
    add_to_queue(pickle.dumps(msg),q)
  #  print("\n\nPID:",pid,"Final classifier solution")
 #   df.loc[:, ['foo','bar','dat']]
 #   print(population.sort_values("strength",ascending=False).to_string())
 #   print(population.loc[:,["condition","message",'strength']].sort_values("strength",ascending=False))   #.to_string())




 
    add_to_queue(pickle.dumps(population.loc[:,["condition","message",'strength']].sort_values("strength",ascending=False).to_string()),q)   #.to_string())





    #  write results to output csv 
    
    with open(params["outfile_csv"], 'a',newline="\n") as f:
        df=population.sort_values(by=["strength"],ascending=False)
        df.head(10).to_csv(f, header=True, index=False, columns=["condition","message"])
 #   population.to_csv(params["outfile_csv"],header=True, mode="a", columns=["condition","message"])
   # print("messages applied",params["messages_applied"])
  #  print(count,":all winners=",params["winners"])

  #  print("\n\nmplx_data=",params["mplx_data"])
  #  print(population.to_string())
    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("duration clock=",duration_clock,"secs.")
    

    return     


def main():
    freeze_support()
    if(len(sys.argv) < 2 ) :
        print("Usage : python simple_gen..._mp.py epoch_length")
        sys.exit()

#
    params = dict(
##        no_of_address_bits=2,
##        no_of_data_bits=2**2,   # binary,  in this case 6 input , 1 output
        min_pop=40,   #3**3,
        import_flag=True, #False,  #True,  imports the starting classifier population 
        use_big_env_file=False, #True,   #  use linecache for the env file instead of pandas
        create_unique_classifiers_flag=False,  #True,   # if true it creates all possible classifiers (conditions : messages) so that they are all unique.  It ignores the size number below, if false they are selected randomly to the size number below
        size=10000,  #(3**9)*(2**2),  # =  17,600ish,  This is for random selection which occurs when no import and not universe creation
        epoch=int(sys.argv[1]),  #1000,
       # input_file=sys.argv[1],
        condition_bits=10,
        message_bits=2,
        message_list=[],
        messages_applied=[],
        scaling=100000,      #   scale the breakdown % of each strength into a whole number to create the wheel
        starting_strength=5000.0,
        bidcoeff1=0.07,  # the % of the strength
        bidcoeff2=0.07,    #  added and adjusted by specificity of the genes ie the amountof '#'.  The total of the two should equal approx 0.12
        bidcoeff_mp=[(0.1,0),(0.095,0.01),(0.09,0.02),(0.085,0.03),(0.08,0.04),(0.075,0.05),(0.07,0.06),(0.065,0.7),(0.095,0.01),(0.09,0.02),(0.1,0.0),(0.1,0.0),(0.1,0.0),(0.1,0.0),(0.1,0.0),(0.1,0.0),(0.1,0.0),(0.1,0.0),(0.1,0.0)],
        bidsigma=0.02,
        bidtax=0,   #0.01
        lifetax=0,
        reward=1,    # 1.45 reward scaling factor from payment
        envrate=1,    # every 1 iterations, inject an environment signal
        feedrate=6000,   # rate that messages are fedback into the conditions (higher is less frequent)  Currently not enabled
        drop_duplicates_enabled=False,  # any duplicate conditions are deleted out each cycle if True
        mutation_enabled=False,
        mutrate=303,  #[0,1,2],  every 50 cycles
        mutstart=23000,  # mutate starting
        mutfinish=24000,
        birth_enabled=False,  #True,  #False,
        birthrate=4,  # 2 new conditions are born every ? cycles
        birthstart=3,  # the number of cycles before repdocution starts
        birthfinish=2000,
        death_enabled=True,  #False,
        deathrate=1,  # an old unfit condition or duplicate is killed every ? cycles
        deathstart=2000,  # the number of cycles before the deaths start
        deathfinish=5000,
        crowding_enabled=False,
        crowding_rate=201,  # purge a number of similar genes to prevent crowding
        crowdingstart=30000,   # the number of cycles befoes crowding kills start
        crowdingfinish=40000,
        crowding_depth=2,  #from the right going left, how many allelles have to be the same to form the crowd sets
        ideal_final_population_size=60,  # use a PID type control system to get the population size to be about this once the birth, death, crowding start 
        condition_match_specificity=1.0,   #  the condition needs only to match the signal 100% of the alleles in the gene to be marked a match
        condition_match_sensitivity_test_disabled=True,   #condition_match_specificity_test_disabled
        message_match_specificity=1.0,      # the message need to match 100%
        message_match_sensitivity_test_disabled=True,
        winner_list=[],   # the top winners over time
       # winner=0,       # current winner all winners over time
        all_winners=[],   # keeps a list of winners to when a kill a row it can check first if it is on the list
        old_winners=[],  # previous winners and before previous winners
        winner_spoils=1.0,   #[1.0,0.0,0.0,0.0],   # strongest winner gets first one and so on.  If there are less winners than elements in the list, the winner gets the balance
        minus1factor=1.0,   # 100% of the receipts go into the previous weeks winners
        minus2factor=0.0,   # 0% of the receipts go to the week before the previous week winners
        env_filename="shop_concat_SGC_encoded34-4.csv",  #   shop_concat_SGC_encoded31_5-1.csv",   #MLGA_environment.csv",
        starting_pop_filename="starting_population.csv",
        diagnostic_file="MLGA_diagnostic.txt",
        outfile_csv="results_csv.csv",    # shop csv
        report_freq=10,   # number of cycles between reporting progress
        results_file="MLGA_results.txt")

    if platform.system().strip().lower()[:7]=="windows":
        params["extra_eol_char"]="\n"
        params["cls"]="cls"
    else:
        params["extra_eol_char"]=""
        params["cls"]="clear"
#
    if params["epoch"]==0:
         params["epoch"]=1000

  #  params["no_of_data_bits"]=2**params["no_of_address_bits"]
    classifier_mp_GA(params)  # run multiproceesing

 

if __name__ == '__main__':
    main()

