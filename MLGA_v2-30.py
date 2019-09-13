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
        
    return(signal)


###############################################################################



# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the probability_table m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

  

def calc_mating_probabilities(newpopulation,r,env):
        count=0
        total_payoff=0.00001

        
        
        #size=len(newpopulation)
       # print("new population size=",size)
        for gene in range(0,r.pop_size):
           # print("gene=",gene,"/",pop_size-1)
            fittest=newpopulation.loc[gene,"expressed"]
          #  val=int(fittest,2)   # binary base turned into integer
            if r.row_find_method=="c":
                total_payoff+=abs(return_a_row_from_envir_using_concatno(fittest,env)[8])
            elif r.row_find_method=="r":
                total_payoff+=abs(return_a_row_from_envir_using_rowno(int(fittest,2),env)[8])
            elif r.row_find_method=="s":
                total_payoff+=abs(float(return_a_row_from_file(int(fittest,2),r.payoff_filename)[8].rstrip()))
            elif r.row_find_method=="l":
                total_payoff+=abs(float(return_a_row_from_linecache(int(fittest,2),r.payoff_filename)[8].rstrip()))
            else:
                pass
           # count+=1


        count=0
        probability_table=[]
        if len(newpopulation)<=1:
            print("\nlen(dna)<=1!")
        for gene in range(0,r.pop_size):
            #val=int(dna[count],2)(newpopulation.loc[gene,"expressed"]
            fittest=newpopulation.loc[gene,"expressed"]
#            val=int(fittest,2)   # binary base turned into integer
            if r.row_find_method=="c":
                p=abs(return_a_row_from_envir_using_concatno(fittest,env)[8])
            elif r.row_find_method=="r":
                p=abs(return_a_row_from_envir_using_rowno(int(fittest,2),env)[8])
            elif r.row_find_method=="s":
                p=abs(float(return_a_row_from_file(int(fittest,2),r.payoff_filename)[8].rstrip()))
            elif r.row_find_method=="l":
                p=abs(float(return_a_row_from_linecache(int(fittest,2),r.payoff_filename)[8].rstrip()))
            else:
                pass

       
            probability_table.append(int(round((p/total_payoff)*r.actual_scaling))) # scaling usually > pop_size*20
            count+=1
        return(probability_table)

    
def spin_the_mating_wheel(probability_table,newpopulation,iterations,direction):
        wheel=[]
        mates=[]
        n=0

   # clock_start=time.clock()

        probability_table_len=len(probability_table)
        if probability_table_len<=1:
            print("\nprobability_table length<=1",probability_table_len)

        mpt=round(mean(probability_table))
   #     print("mean ptable=",mpt)
        #input("?") 
        while n<=probability_table_len-1:
            piesize=probability_table[n]
            if piesize<0:
                piesize=0
                
            if direction=="x":  # maximise
        #    sel=sel+([n+1] * abs(probability_table[n]))
                wheel=wheel+([n+1] * piesize)
            elif direction=="n":   # minimise
                # invert probabilities
                wheel=wheel+([n+1] * abs((2*mpt)-piesize))   # invert across mean
            else:
                print("direction error in spin the probability_table")
                sys.exit()
            n=n+1

  #      print("\nWHEEL\n")
   #     print(wheel)
   #     input("?")
        len_wheel=len(wheel)
   #     print("\nlen(wheel)=",len_wheel,"wheel=",wheel,"\n\nprobability_table=",probability_table)
   #     input("?")
       
        if len_wheel<=20:
            print("\n Warning! increase your total_payoff scaling. len wheel <=20",len_wheel," probability_table len=",probability_table_len)
        for i in range(0,iterations):
            go_back=True
            while go_back:
                # pick a random string for mating
                first_string_no=random.randint(1,probability_table_len)
                # choose its mate from the probability_table
                second_string_no=first_string_no
                while second_string_no==first_string_no:
                    second_string_no=wheel[random.randint(0,len_wheel-1)]
                   # print("mate ",first_string_no,dna[first_string_no-1]," with ",second_string_no,dna[second_string_no-1])

                    # if the string to mate with is the same, try again
                go_back=False
                if newpopulation.loc[first_string_no-1,"chromo1"]==newpopulation.loc[second_string_no-1,"chromo2"]:
                    go_back=True

            mates=mates+[[0.0,first_string_no-1,0,second_string_no-1,0,newpopulation.loc[first_string_no-1,"chromo1"],newpopulation.loc[second_string_no-1,"chromo2"],"","",""]]      # mates is a list of tuples to be mated               


        return(mates,len_wheel)   # if len_wheel gets small, there is a lack of genetic diversity





def crossover(mates,no_of_alleles,individual):
    mate1col=5
    mate2col=6

    xpoint1col=2
    xpoint2col=4

    newchromo1col=7
    newchromo2col=8

    row=0
   # print(mates)

    for mate in mates:
          #  print(mate[i])
           # input("?")
            splitpoint=random.randint(1,no_of_alleles-1)

            child1=""
            child2=""
            remain1=mate[mate1col][:splitpoint]
            swap1=mate[mate1col][splitpoint-no_of_alleles:]
            remain2=mate[mate2col][:splitpoint]
            swap2=mate[mate2col][splitpoint-no_of_alleles:]

            child1=remain1+swap2
            child2=remain2+swap1

            mates[row][newchromo1col]=child1   #+[child1,child2)]
            mates[row][newchromo2col]=child2
            mates[row][xpoint1col]=splitpoint
            mates[row][xpoint2col]=splitpoint
            row+=1    



 #   print("mates dataframe")     

  #  mates_df = pd.DataFrame(mates, columns=list('ABCD'))
  #  mates_df = pd.DataFrame(mates, dtype=individual)   #columns=list('ABCD'))
    mates_df = pd.DataFrame(mates, columns=["fitness","parentid1","xpoint1","parentid2","xpoint2","chromo1","chromo2","newchromo1","newchromo2","expressed"])


    
   # print("mates df")
   # print(mates_df)


    
   # pd.concat([pd.DataFrame(mates[i][0], columns=['chromo1']) for i in range(0,5)], ignore_index=True)
   # pd.concat([newpopulation([i], columns=['chromo1']) for i in range(0,5)], ignore_index=True)
  #  crossed_population.append(mates_df, ignore_index=True,sort=False)


    #mates_df.loc["xpoint1"]=2
    

 #   input("?")

    #delete columns "newchromo1" and "newchromo2"

    mates_df=mates_df.drop(columns=["chromo1","chromo2"])   # delete old chromosomes columns in population
    mates_df.columns=["fitness","parentid1","xpoint1","parentid2","xpoint2","chromo1","chromo2","expressed"]  # rename newchromos to chromos

    mates_df.index
  #  print("mates df drop columns and rename")
   # print(mates_df)
   # input("?")

    
    return(mates_df)







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


    #    print("chromo before mutate=",chromo," at",c1_choice,c2_choice)

        newc=chromo[:c3_choice]+mutated_bit+chromo[c3_choice+1:]
   #     new_chromo_list.append(newc)
   #     print("new chromo after mutate=",newchromo," at",c3_choice,c4_choice)


        if c1_choice==1:
            # chromo1
            newpopulation.loc[c2_choice,"chromo1"]=newc
            
        else:
            # chromo2
            newpopulation.loc[c2_choice,"chromo2"]=newc


       # print(newpopulation.to_string())
        #input("?")

        mutation_count+=1  
        
    return(newpopulation,mutation_count, c1_choice_list, c2_choice_list)









################################################################################

def read_env_file_in(env_filename):
    pass




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
    message=population.loc[indexrow,"message"]   
    return(highest,indexrow,message)


def clearing_house(population,params):
    # return highest bidder
    if len(params["winner_list"])>1:
        previous_winner=params["winner_list"][-2]
        print("previous winner",previous_winner)
    else:
        previous_winner=-1
        print("winners list empty")


##    nan_rows = population[population.isnull().any(1)]
##    print("nan rows=",nan_rows)
##    if not nan_rows.empty:
##        print("pw=",nan_rows.loc[previous_winner])



    value,rowindex,message=winner(population,params["winner_list"])
    params["winner_list"].append(rowindex)
    
    print("best bid=",value,"rowindex=",rowindex,"message=",message)
    # add the payment value to winners row
    population.loc[rowindex,"payments"]=value

    if previous_winner==rowindex:  # if the same winner as made the payment, try the previous winner
        if len(params["winner_list"])>1:
            previous_winner=params["winner_list"][-2]
        else:
            previous_winner=-1
            print("winners list only has one entry")
        

    # divide the payment values among the previous winners according to their strength
    if previous_winner >=0 and (previous_winner in population.index): # and not nan_rows.loc[previous_winner,"condition"]:   # a previous winner exists
        population.loc[previous_winner,"receipts"]=value*params["reward"]    
    
    
    return(population)




def check_allele(row,message,rowindex):
    allele=row["condition"]
   # print("check allele message=",message,"allele=",allele)
    match=False  
    for message_elem in range(0,len(message)):
        m=message[message_elem]
        a=allele[message_elem]   # note classifer is on column 0
     #   print("check allele e=",message_elem,"m=",m,"a=",a)
        if a=="#" or m==a:
           match=True
        else:
           match=False
           break
 #   print("check allele message=",message,"allele=",allele,"match=",match,"rowno=",row.name,"row index=",rowindex[row.name])
    return(match)


def find_matches(message,temppop):
  
  # check each row that the allele in condition matches the message
    temppop["match_flag"]=temppop.apply(check_allele,axis=1,args=[message,temppop.index])

    return(temppop.query("match_flag==True"))

                                        



def kill_the_most_unfit_condition(population):
    print("killing the least fit")
    return(population.drop(population['strength'].idxmin()))


def reset_fields_for_next_cycle(population):
     population["match_flag"]=False
     population["bid"]=0.0
     population["ebid"]=0.0
     population["payments"]=0.0
     population["taxes"]=0.0
     population["receipts"]=0.0
     return(population)



def timer(count,params):
    # this times the mutations, environmental messages, deaths and births

    mutate_flag=False
    env_flag=False
    death_flag=False
    birth_flag=False

    if count%params["mutrate"]==0:
        mutate_flag=True

    if count%params["envrate"]==0:
        env_flag=True

    if count%params["deathrate"]==0:
        death_flag=True
        
    if count%params["birthrate"]==0:
        birth_flag=True



    return(mutate_flag,env_flag,death_flag,birth_flag)


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
   # message_temp=add_message_to_message_list(multiplexer(params["no_of_address_bits"],params["no_of_data_bits"]),message_temp)

    # OR read in from environment CSV file
 #   message_temp=add_message_to_message_list([read_env_file_in(params["env_filename"])],message_temp)

    return(df)



def classifier_mp_GA(params):   # multiprocessing stuff will go here later
    classifier_GA(params)



def classifier_GA(params):
    count=0
    population=startup(params)
##    
##    startenv=str(startenv_message[0])  # value is one element in a list inside a list
##    print("se=",startenv)
##    params["message_list"]=add_message_to_message_list([startenv],[])
##    print("message list=",params["message_list"])        
## #   matches=find_matches(startenv,population)
 #   print("startenv=",startenv,"matches=\n",matches)


  #  print("se=",startenv)
    # seed the classifers with the first message

    
 #   print("message list before",params["message_list"])
##    m="m"
##    while m:
##        m=get_message_off_message_list(params["message_list"])
##        if m:
##            print("m=",m)


    while count<params["epoch"]:   # loop while the message list is not empty

        env_flag,mutate_flag,death_flag,birth_flag=timer(count,params)

        print("len5=",len(population))
        message=get_message_off_message_list(params["message_list"],params)
        print("len7=",len(population))
        matches=find_matches(message,population)
        print("len8=",len(population))
    #    print("message=",message)
        #print("matches=\n",matches)

        if count%params["envrate"]==0:
            print("message from env")   # insert the env message first on the list
            params["message_list"].insert(0,multiplexer(params["no_of_address_bits"],params["no_of_data_bits"]))   #        else:
        else:
         #   print("message from conditions")   # add message from conditions to the end of the list
   #        params["message_list"]=add_message_to_message_list(extract_messages_to_list(matches),params["message_list"])
            params["message_list"]=add_message_to_message_list(extract_messages_to_list(matches),params["message_list"])

        print("len6=",len(population))
        print("message list=",params["message_list"])
        if len(params["message_list"])==0:
            params["message_list"].insert(0,multiplexer(params["no_of_address_bits"],params["no_of_data_bits"]))   #        else:
 
            #print("message list empty")
            #break

        print("len=",len(population))
        population=bid(population,params)
    #    print(population)
        population=effective_bid(population,params)
        print("len2=",len(population))
        population=tax(population,params)
     #   print(population)
        population=clearing_house(population,params)
      #  print(population.to_string())
        print(count,":winners=",params["winner_list"])
     #   print("messages applied=",params["messages_applied"])
      #  input("?")

        population=update_strengths(population)  
    #    print(population.to_string())
     #   input("?")
        population=reset_fields_for_next_cycle(population)
        print(population.to_string())
      #  input("?")
      #  print("len before=",len(population),"pop before=",population)
        if death_flag:  
            population=kill_the_most_unfit_condition(population)
    #    print("len after=",len(population),"pop after=",population)

    #    print(population.to_string())
   #     nan_rows = population[population.isnull().any(1)]
   #     print("nan rows=",nan_rows)
        input("?")
      
        count+=1  

    print("\n\nFinal classifier solution")
    print(population.sort_values("strength",ascending=False))
     


def main():
    freeze_support()

    params = dict(
        no_of_address_bits=4,
        no_of_data_bits=0,
        size=16,
        epoch=40,
       # input_file=sys.argv[1],
        condition_bits=4,
        message_bits=4,
        message_list=[],
        messages_applied=[],
        starting_strength=200.0,
        bidcoeff=0.1,
        bidsigma=0.15,
        bidtax=0.03,
        lifetax=0.01,
        reward=1.3,
        envrate=4,    # every 4 iterations, inject an environment signal
        mutrate=50,  #[0,1,2],  every 50 cycles
        birthrate=10,  # new condition is born every 10 cycles
        deathrate=3,  # an old unfit condition is killed every 10 cycles
        winner_list=[],
        env_filename="MLGA_environment.csv",
        diagnostic_file="MLGA_diagnostic.txt",
        results_file="MLGA_results.txt")
    

    params["no_of_data_bits"]=2**params["no_of_address_bits"]


    classifier_mp_GA(params)  # run multiproceesing

 

if __name__ == '__main__':
    main()

