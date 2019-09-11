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

 #   print("output address=",output_address)
    print("output=",output)
    print("")
        
    return([signal])



def read_env_file_in(env_filename):
    pass




def extract_messages_to_list(population):
    message_list=population["message"].values.tolist()
    return(message_list)

def get_message_off_message_list(message_list):
    if len(message_list)>0:
        message=message_list[0]
        del message_list[0]
        return(message)
    else:
        return("")


def add_message_to_message_list(messages,message_list):
 #   message_list.append(messages)
 #   return(list(itertools.chain(*message_list)))
     return(message_list+list(messages))



def create_classifiers(population,condition_bits):
    population_len=len(population)
    
    for p in range(0,population_len):
        not_unique=True
        while not_unique:
            cl=""    
            for alleles in range(0,condition_bits):
                bit=random.randint(-1,1)
                if bit==-1:
                    b="#"
                else:
                    b=str(bit)
                cl=cl+b

            not_unique= not (population[population["condition"]==cl].empty)

        population.loc[p,"condition"]=cl
        
    return(population)   
 

def create_messages(population,message_bits):
    population_len=len(population)
    
    for p in range(0,population_len):
        ml=""    
        for alleles in range(0,message_bits):
            ml=ml+str(random.randint(0,1))
           
        population.loc[p,"message"]=ml
        
    return(population)   
 

def update_strengths(population):
    # strength(t+1)=strength(t)-payments(t)-taxes(t)+receipts(t)
    population["strength"]=population["strength"]-population["payments"]-population["taxes"]+population["receipts"]
    return(population)

def bid_adjust(row,coeff):
    return(row["strength"]*coeff)   #params["bidcoeff"])

def bid(population,params):
    # each condition that matching the message makes a bid proportional to thier strength ( bidcoeff)
    mask=population.match_flag == True
  #  column_name="match_flag"
 #   population.loc[mask,"bid"]=population.apply(bid_adjust,axis=1,args=[3])
    population.loc[mask,"bid"]=population.apply(bid_adjust,axis=1,args=[params["bidcoeff"]])

    return(population)
    
def noise(row,mu,sigma):  # normal distribution
    return(row["bid"]+np.random.normal(mu,sigma))


def effective_bid(population,params):
    # there is a tax on each transaction which reduces the bid to an effective bid which is ebid
    
    mask=population.match_flag == True
 #   column_name="match_flag"
    population.loc[mask,"ebid"]=population.apply(noise,axis=1,args=[0.0,params["bidsigma"]])

##    
##    for p in range(0,len(population)):
##        if population.iloc[p][2]!=0:
##            population.iloc[p][3]=population.iloc[p][2]+noise(0.0,params["bidsigma"])

    return(population)



def tax_adjust_existance(row,taxrate):
    return(row["taxes"]+row["strength"]*taxrate)

def tax_adjust_bid(row,taxrate):
    return(row["taxes"]+row["bid"]*taxrate)


def tax(population,params):
  
  # apply exisitance tax to everyone
    population["taxes"]=population.apply(tax_adjust_existance,axis=1,args=[params["lifetax"]])
    
    # apply bidtax to bidders only   
    mask=population.match_flag == True
    population.loc[mask,"taxes"]=population.apply(tax_adjust_bid,axis=1,args=[params["bidtax"]])
    
    return(population)





def winner(population,winner_list):
    # find the highest bidder
    highest=population["ebid"].max()
    indexrow=population["ebid"].idxmax()
   
    return(highest,indexrow)


def clearing_house(population,params):
    # return highest bidder
    if params["winner_list"]:
        previous_winner=params["winner_list"][-1]
        print("previous winner",previous_winner)
    else:
        previous_winner=-1
        print("winners list empty")

    value,row=winner(population,params["winner_list"])
    params["winner_list"].append(row)
    print("best bid=",value,"row=",row)
    # add the payment value to winners row
    population.loc[row,"payments"]=value

    # divide the payment values among the previous winners according to their strength
    if previous_winner >=0:   # a previous winner exists
        population.loc[previous_winner,"receipts"]=value*params["reward"]    
    
    
    return(population)
    

def find_matches(message,population):
    for p in range(0,len(population)):
        match=False
        for allele in range(0,len(message)):
            m=message[allele]
            t=population.iloc[p,0][allele]   # note classifer is on column 0

            if t=="#" or m==t:
               match=True
            else:
               match=False
               break
        
        population.loc[p,"match_flag"]=match
    
            
    return(population[population["match_flag"]==True])


def reset_fields_for_next_cycle(population):
     population["match_flag"]=False
     population["bid"]=0.0
     population["ebid"]=0.0
     population["payments"]=0.0
     population["taxes"]=0.0
     population["receipts"]=0.0
     return(population)






##    individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
##    poparray = np.zeros(params["pop_size"], dtype=individual) 
##    population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")


# set up numpy structure and pandas dataframe
# strength(t+1)=strength(t)-payments(t)-taxes(t)+receipts(t)

def setup_df(p):
    condition_len="S"+str(p["condition_bits"])
    message_len="S"+str(p["message_bits"])    
    
    individual = np.dtype([("condition",condition_len),("message",message_len),("bid",'f'),("ebid",'f'),("match_flag",'b'),("strength",'f'),("payments","f"),("taxes","f"),("receipts","f")])
    poparray = np.zeros(p["size"], dtype=individual) 
    population = pd.DataFrame({"condition":poparray["condition"],"message":poparray["message"],"bid":poparray["bid"],"ebid":poparray["ebid"],"match_flag":poparray["match_flag"],"strength":poparray["strength"],"payments":poparray["payments"],"taxes":poparray["taxes"],"receipts":poparray["receipts"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")
    population["strength"]=p["starting_strength"]   #+np.random.normal(0.0,1)
    population["match_flag"]=False
    population=create_classifiers(population,p["condition_bits"])
    population=create_messages(population,p["message_bits"])
    
    return(population)



def startup(params):

    df=setup_df(params)   # setup dataframe
    
    message_temp=[]

    total_inputs=params["no_of_address_bits"]+params["no_of_data_bits"]
    print("Address bits=",params["no_of_address_bits"]," Data bits=",params["no_of_data_bits"]," Total number of inputs=",total_inputs)

    # create first environmental message (from multiplexer)
    message_temp=add_message_to_message_list(multiplexer(params["no_of_address_bits"],params["no_of_data_bits"]),message_temp)

    # OR read in from environment CSV file
 #   message_temp=add_message_to_message_list([read_env_file_in(params["env_filename"])],message_temp)

    return(message_temp,df)



def classifier_mp_GA(params):   # multiprocessing stuff will go here later
    classifier_GA(params)



def classifier_GA(params):
    
    startenv_message,population=startup(params)
    
    startenv=str(startenv_message[0])  # value is one element in a list inside a list
    print("se=",startenv)
    params["message_list"]=add_message_to_message_list([startenv],[])
    print("message list=",params["message_list"])        
 #   matches=find_matches(startenv,population)
 #   print("startenv=",startenv,"matches=\n",matches)


  #  print("se=",startenv)
    # seed the classifers with the first message

    
 #   print("message list before",params["message_list"])
##    m="m"
##    while m:
##        m=get_message_off_message_list(params["message_list"])
##        if m:
##            print("m=",m)


    while len(params["winner_list"])<50 and len(params["message_list"])>0:   # loop while the message list is not empty

        message=get_message_off_message_list(params["message_list"])
        matches=find_matches(message,population)
        print("message=",message,"matches=\n",matches)

        params["message_list"]=add_message_to_message_list(extract_messages_to_list(matches),params["message_list"])

        print("message list=",params["message_list"])
        

     #   print(population)
        population=bid(population,params)
     #   print(population)
        population=effective_bid(population,params)
     #   print(population)
        population=tax(population,params)
     #   print(population)
        population=clearing_house(population,params)
      #  print(population.to_string())
        print("winners=",params["winner_list"])
      #  input("?")

        population=update_strengths(population)  
        print(population.to_string())
      #  input("?")
        population=reset_fields_for_next_cycle(population)
      #  print(population.to_string())
      #  input("?")

    print("\n\nFinal classifier solution")
    print(population.sort_values("strength",ascending=False))
     


def main():
    freeze_support()

    params = dict(
        no_of_address_bits=4,
        no_of_data_bits=0,
        size=11,
       # input_file=sys.argv[1],
        condition_bits=4,
        message_bits=4,
        message_list=[],
        starting_strength=200.0,
        bidcoeff=0.1,
        bidsigma=0.075,
        bidtax=0.01,
        lifetax=0.005,
        reward=1.4,
        noise=0.01,
        mutation_rate=[0,1,2],
        winner_list=[],
        env_filename="MLGA_environment.csv",
        diagnostic_file="MLGA_diagnostic.txt",
        results_file="MLGA_results.txt")
    

    params["no_of_data_bits"]=2**params["no_of_address_bits"]


    classifier_mp_GA(params)  # run multiproceesing

 

if __name__ == '__main__':
    main()

