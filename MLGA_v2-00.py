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
# ususually we use a roulette wheel based on each classifiers strength as used for fitness
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
import pandas as pd
import random
import time
import csv
import math
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


# multiplexer
# 6 signals come into the multiplexer
# first two lines are A are decoded as a unsigned binary integer
# this address values is then used to work out which of the four remaining signels
# on the Data or d lines is to be passed thwough to the mulriplexer output


def generate_data(no_of_data_bits):
    
    data=""
    for i in range(0,no_of_data_bits):
        data=data+str(random.randint(0,1))
        

    return(data)

def generate_signal(no_of_address_bits):
    
    addresses=""
    for i in range(0,no_of_address_bits):
        addresses=addresses+str(random.randint(0,1))
        
   
    return(addresses)



def multiplexer(no_of_address_bits,no_of_data_bits):
    
    data=generate_data(no_of_data_bits)
    print("data=",data)
    
    
    signal=generate_signal(no_of_address_bits)
    
    print("signal=",signal)
  #  print("data=",data)
    output_address=""
    for i in range(0,len(signal)):
        output_address=output_address+str(int(signal[i],2))    #"".join(map(str,signal)),2)
    output=data[int(output_address,2)]

  #  print("output address=",output_address)
    print("output=",output)
    print("")
        
    return(signal,output)



def create_classifiers(population,classifier_bits):
    population_len=len(population)
    
    for p in range(0,population_len):
        not_unique=True
        while not_unique:
            cl=""    
            for alleles in range(0,classifier_bits):
                bit=random.randint(-1,1)
                if bit==-1:
                    b="#"
                else:
                    b=str(bit)
                cl=cl+b

            not_unique= not (population[population["classifier"]==cl].empty)

        population.loc[p,"classifier"]=cl
        
    return(population)   
 

def create_messages(population,message_bits):
    population_len=len(population)
    
    for p in range(0,population_len):
        ml=""    
        for alleles in range(0,message_bits):
            ml=ml+str(random.randint(0,1))
           
        population.loc[p,"message"]=ml
        
    return(population)   
 


def find_matches(message,population):
    for p in range(0,len(population)):
        match=False
        for allele in range(0,len(message)):
            m=message[allele]
  #          t=population.loc[p].values.tolist()[0][allele]   # note classifer is on column 0
            t=population.iloc[p,0][allele]   # note classifer is on column 0

            if t=="#" or m==t:
            #   print(m,t,"match")
               match=True
            else:
             #  print(m,t,"no match")
               match=False
               break
        
        population.loc[p,"match_flag"]=match
    
            
    return(population[population["match_flag"]==True])









##    individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
##    poparray = np.zeros(params["pop_size"], dtype=individual) 
##    population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")


# set up numpy structure and pandas dataframe
# strength(t+1)=strength(t)-payments(t)-taxes(t)+receipts(t)

def setup_df(p):
    classifier_len="S"+str(p["classifier_bits"])
    message_len="S"+str(p["message_bits"])    
    
    individual = np.dtype([("classifier",classifier_len),("message",message_len),("bid",'f'),("ebid",'f'),("match_flag",'b'),("strength",'f'),("payments","f"),("taxes","f"),("receipts","f")])
    poparray = np.zeros(p["size"], dtype=individual) 
    population = pd.DataFrame({"classifier":poparray["classifier"],"message":poparray["message"],"bid":poparray["bid"],"ebid":poparray["ebid"],"match_flag":poparray["match_flag"],"strength":poparray["strength"],"payments":poparray["payments"],"taxes":poparray["taxes"],"receipts":poparray["receipts"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")
    population["strength"]=p["starting_strength"]
    population=create_classifiers(population,p["classifier_bits"])
    population=create_messages(population,p["message_bits"])
    
    return(population)



def main():
    freeze_support()

    params = dict(
        size=81,
       # input_file=sys.argv[1],
        classifier_bits=4,
        message_bits=4,
        starting_strength=200.0,
        bidcoeff=0.1,
        bidtax=0.1,
        lifetax=0.0,
        reward=1,
        noise=0.0,
        mutation_rate=[0,1,2],
        diagnostic_file="MLGA_diagnostic.txt",
        results_file="MLGA_results.txt")




    no_of_address_bits=4
    no_of_data_bits=2**no_of_address_bits
    total_inputs=no_of_address_bits+no_of_data_bits
    print("Address bits=",no_of_address_bits," Data bits=",no_of_data_bits," Total number of inputs=",total_inputs)

    startenv,output=multiplexer(no_of_address_bits,no_of_data_bits)  # returns signal, output





    
    df=setup_df(params)
    print(df)

   # startenv="0100"
    matches=find_matches(startenv,df)
    print("startenv=",startenv,"matches=\n",matches)

##    newmatches=matches
##    m=len(newmatches)
##    print("m=",m)
##    for p in range(0,m):
##        env=matches.iloc[p,1]
##        print("p=",p,"env2=",env)
##        newmatches.append(find_matches(env,matches))
##    print(newmatches)
      #  matches=newmatches  #.copy()


if __name__ == '__main__':
    main()

