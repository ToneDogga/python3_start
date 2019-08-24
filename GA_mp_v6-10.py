# Genetic algorithms trialletic version 1 started 10/8/19
#
# Basic structure for a simple algorthim


# improvements over simple algorithm
#  instead of binary, use three alleles per locus
#  -1,0 and 1
#  this allows a dominate gene and a recessive gene
# this allows a long term memory of adaptation that can be called on when
# the environment changes and fitness levels adjust
#
# with this range of 3, we can intend -1 to map to a recessive 1  (use % to report)
# 0 to map to a 0
# and 1 to map to a dominant 1
# then the dominance expression is a simple >= compare
#
# we now have a pair of choromomes
# dominance expression is simply moving through each together at each locus and
# comparing the value and taking the largest.
#
# in reproduction
# the pair of chromosomes creates a pair of gametes which in turn is fertilised by a second pair of gametes
# 
# with creation and mutation
# -1 (recessive 1) is chosen 25% of the time
# 0 is chosen 50% of the time
# and dominant 1 is chosen 25% of the time
#

#



# 5 switches either on or off
# 1 payoff value for each setting
# the switches are represented by a 5 bits 0 or 1
#
# the payoff value could be a simple function.  say f(x)=x**2
#
# initialise
# generate search space
#
# generation 1
# randomly generate a population of n encoded strings    eg '01000'
#
# start
# test each member of the population against the payoff value
#
# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the wheel m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.
#
# crossover
# the new generation of strings is mated at random
# this means each pair of strings is crossed over at a uniform point in the string
# find a random split point in the string a swap the remaining information over between the mating pairs
#
# mutation
# this is the occasional and small chance (1/1000) that an element of a string changes randomly.
#
# go back to start with the new generation

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

#global TOTAL_TOTAL_GENERATIONS
#TOTAL_TOTAL_GENERATIONS=0

#def writetofile(bytestr,f):   
#    f.write(str(bytestr)+"\n")
#    f.flush()
#    return



def listener(q,results_file):    #,l_lock):
    '''listens for messages on the q, writes to file. '''
# empty out file
   # f=open(results_file,"w")
   # f.close()
  #  g=open(diag_file,"w")
  #  g.close()
    
    print("Multicore processing starting. Queue listener started on PID:",os.getpid())
   # l_lock.acquire()
    with open(results_file,"a") as f:   #,buffering=8096)
        while True:
           # if q.full():
           #     print("q full")
           #     break
            message = q.get()    #timeout=20)  get_nowait
            if message == "kill":
             #   print("trying to kill listener process.  Flushing q.  qsize=",q.qsize())
                #f.write("killed\n")
                while not q.empty():
                    message=q.get()
                    f.write(str(message)+"\n")
                    f.flush()
                    #writetofile(m,f)
       #             writetofile(m,g)
                #   f.flush()
                break
            else: 
                #writetofile(m,f)
                f.write(str(message)+"\n")
                f.flush()
            #    writetofile(m,g)
               #+":"+q.qsize()+"\n")
              #  f.write(str(q.qsize())+'\n')
               # f.flush()
           
      #  q.close()
    f.close()
 #   g.close()       
    print("Multicore processing finishing. Queue listener closed on:",os.getpid())
   # l_lock.release()
    return(True)

def add_to_queue(msg,q):
    #if rfileout is True, send messages arriving in the q to the results file
    # if dfileout is true, do the same for the diagnostics file

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

##def generate_payoff_environment_1d_mp(astart_val,asize_of_env,q,linewrite,lwremainder):    #l_lock  
##    rowno=0
##    print("payoff calc function started:",os.getpid())
##
##    clock_start=timer()  #.process_time()
##
##    pline=""
##    for a in range(astart_val,astart_val+asize_of_env):
##        payoff=-10*math.sin(a/44)*12*math.cos(a/33)*1000/(a+1)
##        pline=pline+(str(rowno)+","+str(a)+","+str(payoff)+"\n")
##        if rowno%linewrite==0:
##            add_to_queue(pline,q)
##            pline=""     
##        rowno+=1
##        
##    if pline:
##        add_to_queue(pline,q)
##
## 
##    clock_end=timer()  #.process_time()
##    duration_clock=clock_end-clock_start
##
##    print("payoff pid:[",os.getpid(),"] finished chunk",rowno,"rows in:",duration_clock,"secs.")
##    return   #(payoff)    #(os.getpid(),payoff)



        
# Python program to find SHA256 hash string of a file
def hash_a_file(FILENAME):
    #filename = input("Enter the input file name: ")
    sha256_hash = hashlib.sha256()
    with open(FILENAME,"rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    return(sha256_hash.hexdigest())



def count_file_rows(FILENAME):
    with open(FILENAME,'r') as f:
        return sum(1 for row in f)









  

def generate_payoff_environment_7d_file(astart_val,asize_of_env,bstart_val,bsize_of_env,cstart_val,csize_of_env,dstart_val,dsize_of_env,estart_val,esize_of_env,fstart_val,fsize_of_env,gstart_val,gsize_of_env,filename):   
    rowno=0    
    total_rows=asize_of_env*bsize_of_env*csize_of_env*dsize_of_env*esize_of_env*fsize_of_env*gsize_of_env    
    with open(filename,"w") as filein:
        for a in range(astart_val,astart_val+asize_of_env):
            print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
            for b in range(bstart_val,bstart_val+bsize_of_env):
                for c in range(cstart_val,cstart_val+csize_of_env):
                    for d in range(dstart_val,dstart_val+dsize_of_env):
                        for e in range(estart_val,estart_val+esize_of_env):
                            for f in range(fstart_val,fstart_val+fsize_of_env):
                                for g in range(gstart_val,gstart_val+gsize_of_env):
                                    payoff=100*math.sin(a/44)*120*math.cos(b/33)-193*math.tan(c/55)+78*math.sin(d/11)-98*math.cos(e/17)+f+g
#                                  filein.write(str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(f)+","+str(g)+","+str(payoff)+"\n")
                                    # concat number is the concatentation of the binary values of the input variable.  it is a string
                #                    concatno=f"{a:02b}"+f"{b:02b}"+f"{c:02b}"+f"{d:02b}"+f"{e:02b}"+f"{f:02b}"+f"{g:02b}"
                               #     w=str(rowno)+","+str(int(concatno,2))+","+concatno+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(f)+","+str(g)+","+str(payoff)
                                #    w=str(int(concatno,2))+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(f)+","+str(g)+","+str(payoff)
                                    w=str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(f)+","+str(g)+","+str(payoff)
                                    padding=76-len(w)-1
                                    w=w+" "*padding
                                    rowno+=1
                                    filein.write(w+"\n")
                                    
    print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
    filein.close()
    print("")
    return(rowno)


##
##def format_csv(r):   
##    rowno=0
##    g=open("new"+r.payoff_filename,"w")  # clear file out
##    g.close()
##    g=open("new"+r.payoff_filename,"a")  # create a new file to write the padded lines out to
##    f=open(r.payoff_filename,"r")
##    while True:
##        w=f.readline().strip()
##        if not w: break
##        # w=w[:-(r.extra_eol_char)]   # take \r off if windows
##        padding=r.linewidth+r.extra_eol_char-2-len(w)   #-2
##        if rowno==0 and r.extra_eol_char==1:
##            w=w+"   "
##        w=w+" "*padding
##      #  while len(w)<r.linewidth:
##      #      w=w+" "
##      #  if len(w)!=r.linewidth:    
##      #      print("lenw=",len(w))
##      #      input("?")
##        g.write(w+"\n")
##        rowno+=1                           
##
##   # print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
##    g.close()
##    f.close()
##  #  print("")
##    return
##




def count_file_rows(filename):
    with open(filename,'r') as f:
        return sum(1 for row in f)


def build_population(size,no_of_alleles,genepool):
    for p in range(0,size):
        genepool.loc[p,"chromo1"]=build_chromosome(no_of_alleles)
        genepool.loc[p,"chromo2"]=build_chromosome(no_of_alleles)
        genepool.loc[p,"expressed"]=express(genepool.loc[p,"chromo1"],genepool.loc[p,"chromo2"])

    return(genepool)

def build_chromosome(no_of_alleles):
        dna=[]
        dna_string=""
        for locus in range(0,no_of_alleles):
            if random.randint(0,1)==0:
                dna.append("0")
                dna_string+="0"
            else:
                if random.randint(0,1)==1:
                        # dominant 1
                    dna.append("1")
                    dna_string+="1"
                else:
                        # recessive 1
                    dna.append("-1")
                    dna_string+="%"
        return(dna_string)


def express(c1,c2):
    # expressed dominance - the homologus pair = single chromosome
        single=""
        l=len(c1)
        for locus in range(0,l):
            single+=convert_allele_to_string(mapdominance(return_allele_as_number(c1[locus:locus+1]),return_allele_as_number(c2[locus:locus+1])))
      #  print("single=",single)
        return(single)


def mapdominance(allele1,allele2):
        # simple dominance map based on -1,0 or 1 where 1 is represented by % in the string and is an recessive 1
        if allele1>=allele2:
            return(abs(allele1))
        else:
            return(abs(allele2))
    
        

def return_allele_as_number(a):
        if a=="%":
            return(-1)
        else:
            return(int(a))

def convert_allele_to_string(allele):
        if allele==-1:
            return("%")
        elif allele==0:
            return("0")
        elif allele==1:
            return("1")
        else:
            print("convert allele to string function error. allele=",allele)
            return("X")


def return_a_row_from_envir_using_rowno(val,env):
        # use the preloaded pandas dataframe of the payoff file
        try:   
            ret=env.iloc[val].values.tolist()
            if ret:
         #       print(ret)
                return(ret,True)
            else:
         #       print("val not found in environment")
                return(["",0,0,0,0,0,0,0,0],False)
            
        except IndexError:
         #   print("\nindex error val=",val)
            return(["",0,0,0,0,0,0,0,0],False)
        except ValueError:
            print("\nvalue error val=",val)
            return(["",0,0,0,0,0,0,0,0],False)
        except IOError:
            print("\nIO error")
            return(["",0,0,0,0,0,0,0,0],False)





def return_a_row_from_envir_using_concatno(val,env):
        # use the preloaded pandas dataframe of the payoff file
        try:
            ret=env.loc[env["concatno"]==val].values.tolist()[0]   
            if ret:
                return(ret,True)
            else:
         #       print("val not found in environment")
                return(["",0,0,0,0,0,0,0,0],False)
            
        except IndexError:
        #   print("\nindex error val=",val)
            return(["",0,0,0,0,0,0,0,0],False)
        except ValueError:
            print("\nvalue error val=",val)
            return(["",0,0,0,0,0,0,0,0],False)
        except IOError:
            print("\nIO error")
            return(["",0,0,0,0,0,0,0,0],False)



def return_a_row_from_file(val,r):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:
            with open(r.payoff_filename,"r") as f:
                f.seek((r.linewidth+r.extra_eol_char)*val,0)   # if a windows machine add an extra char for the '\r' EOL char
                d=f.readline().rstrip()

          #     return(list(itertools.chain(*p)))
                
        except IndexError:
          #  print("\nindex error at val+1, val=",val)
            return([0,0,0,0,0,0,0,0,0],False)
        except ValueError:
            print("\nvalue error val+1=",val+1)
            return([0,0,0,0,0,0,0,0,0],False)
        except IOError:
            print("\nIO error")
            return([0,0,0,0,0,0,0,0,0],False)

               
        line=d.split(',')  

           # i=[]    
         #   d.insert(0,0)  # add an extra element to the row because the concatno field is not in the payoff file
        print("line=",line,"len line=",len(line),"d=",d,"val=",val)
          #  input("?")
        if len(line)>2:
               # i=list(itertools.chain(*d))
            line.insert(0,0)   # add an extra element to the row because the concatno field is not in the payoff file
            print("line=",line)
            input("?")
            return(line,True)

        else:
            return([0,0,0,0,0,0,0,0,0],False)




def return_a_row_from_linecache(val,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            l=linecache.getline(filename,val+1).rstrip()
            if l:
                p=l.split(',')  # add an extra element to the row because the concatno field is not in the payoff file
             #   print("linecache p=",p)
                if p:
                    return(p,True)
                #    return(list(itertools.chain(*p)),True)
                else:
                    return([0,0,0,0,0,0,0,0],False)
            else:
                return([0,0,0,0,0,0,0,0],False)
               
 
            
        except IndexError:
         #   print("\nindex error at val+1",val+1)
            return([0,0,0,0,0,0,0,0],False)
        except ValueError:
       #     print("\nvalue error row+1=",val+1) 
            return([0,0,0,0,0,0,0,0],False)
        except IOError:
            print("\nIO error")
            return([0,0,0,0,0,0,0,0],False)




# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the probability_table m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

  

def calc_mating_probabilities(newpopulation,params):

        clock_start=time.process_time()

        count=0
        total_payoff=0.00001
        found=False
        payoff=[]
        
        #size=len(newpopulation)
       # print("new population size=",size)
        for gene in range(0,params["pop_size"]-1):
           # print("gene=",gene,"/",pop_size-1)
            fittest=newpopulation.loc[gene,"expressed"]
          #  val=int(fittest,2)   # binary base turned into integer
        #    if r.row_find_method=="c":
        #        payoff,found=return_a_row_from_envir_using_concatno(fittest,env)
        #        if found:
        #            total_payoff+=abs(payoff[8])
                    
       #     elif r.row_find_method=="r":
       #         payoff,found=return_a_row_from_envir_using_rowno(int(fittest,2),env)
       #         if found:
       #             total_payoff+=abs(payoff[8])
                    
        #    elif r.row_find_method=="s":
        #        payoff,found=return_a_row_from_file(int(fittest,2),r)
        #        if found:
        #            total_payoff+=abs(float(payoff[8].rstrip()))
                
         #   elif r.row_find_method=="l":
            payoff,found=return_a_row_from_linecache(int(fittest,2),params["payoff_filename"])
            if found:
                total_payoff+=abs(float(payoff[7].rstrip()))
           # else:
            #    print("row find method error")
             #   sys.exit()
 
                
           


        count=0
        probability_table=[]
        if len(newpopulation)<=1:
            print("\nlen(dna)<=1!")
        for gene in range(0,params["pop_size"]-1):
            #val=int(dna[count],2)(newpopulation.loc[gene,"expressed"]
            fittest=newpopulation.loc[gene,"expressed"]
#            val=int(fittest,2)   # binary base turned into integer
        #    if params["row_find_method"]=="c":
        #        payoff,found=return_a_row_from_envir_using_concatno(fittest,env)
        #        if found:
        #            p=abs(payoff[8])
        #        else:
        #            p=0

        #    elif r.row_find_method=="r":
        #        payoff,found=return_a_row_from_envir_using_rowno(int(fittest,2),env)
        #        if found:
        #            p=abs(payoff[8])
        #        else:
        #            p=0
                    
        #    elif r.row_find_method=="s":
        #        payoff,found=return_a_row_from_file(int(fittest,2),r)
        #        if found:
        #            p=abs(float(payoff[8].rstrip()))
        #        else:
        #            p=0
                    
       #     elif r.row_find_method=="l":
            payoff,found=return_a_row_from_linecache(int(fittest,2),params["payoff_filename"])
            if found:
                p=abs(float(payoff[7].rstrip()))
            else:
                p=0
          #  else:
          #      print("row find method error")
          #      sys.exit()
                

       
            probability_table.append(int(round((p/total_payoff)*params["actual_scaling"]))) # scaling usually > pop_size*20
            count+=1
        clock_end=time.process_time()
        duration_clock=clock_end-clock_start
        #outfile.write("probability table - Clock: duration_clock ="+str(duration_clock)+"\n")


        return(probability_table)

    
def spin_the_mating_wheel(probability_table,newpopulation,iterations,direction):
        clock_start=time.process_time()

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
        clock_end=time.process_time()
        duration_clock=clock_end-clock_start
        #outfile.write("spin the mating wheel - Clock: duration_clock ="+str(duration_clock)+"\n")


        return(mates,len_wheel)   # if len_wheel gets small, there is a lack of genetic diversity





def crossover(mates,no_of_alleles,individual):
    clock_start=time.process_time()

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
    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
 #   r.outfile.write("crossover - Clock: duration_clock ="+str(duration_clock)+"\n")
 
    
    return(mates_df)








def mutate(newpopulation,no_of_alleles,ploidy,pop_size,mutation_rate):
    # we need to change random bits on chromo1 or chromo2 column to a random selection - 25%="%" (recessive 1), 25%="1" , 50% = "0"
    #the mutation rate should ideally be about 1 in a 1000 bits.  a mutation rate of 1000 means 1 in a 1000 bits
    # number of bits going through per mutation cycle= no_of_alleles*2 ploidy * pop_size
    # 7*2*128=1792
    clock_start=time.process_time()
    
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
        
    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
 #   r.outfile.write("mutate - Clock: duration_clock ="+str(duration_clock)+"\n")
  
    return(newpopulation,mutation_count, c1_choice_list, c2_choice_list)


######################################################

        
def GA_mp(bits,constraints,params):  # multiprocessing! run the same epoch length, pool size and environment data in parallel at different mutation rates

    
    manager = multiprocessing.Manager()
    q = manager.Queue()    
   
    cpus = multiprocessing.cpu_count()
   
##        windows=platform.system().lower()[:7]
##        print("platform=",windows)
##        if windows=="windows":
##            EOL="\r\n"
##        else:
##            EOL='\n'


    print("\nCPUs=",cpus,". (1 queue listener, and",cpus-1,"cores optimising in parallel).")
    multiple_results=[]

    with Pool(processes=cpus) as pool:  # processes=cpus-1
         #put listener to work first
        watcher = pool.apply_async(listener, args=(q,params["results_file"] ))


   #     if remainder!=0:
   #         multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,args=(0,remainder,q,linewrite,lwremainder ))]
   #     else:
   #         multiple_results=[]
        for i in range(0,cpus-1):
            multiple_results.append(pool.apply_async(genetic_algorithm,args=(bits,constraints,params,params["mutation_rate"][i],q )))

        for res in multiple_results:
            result=res.get(timeout=None)
            params["total_total_generations"]=result
            res.wait()

        print("Generate payoff results finished")

   #     print("killing listener")
        q.put("kill")
        result=watcher.get(timeout=None) 
        watcher.wait()

        
  #  print("try to close pool")
    pool.close()
  #  print("pool closed.  trying to join() pool")
    pool.join()
  #  print("pool join() complete")
    return





###################################################

def genetic_algorithm(bits,constraints,params,mut_rate,q):   # dictionary "params" contains all the settings for the algorithm run, env is the pandas loaded dataframe from the payoff file csv

    pid=str(os.getpid())
    starttime=str(datetime.datetime.now())
    n="PID:"+pid+" Genetic algorithm. Started at: "+starttime+"\n"
    n=n+"PID:"+pid+" Environment file="+params["payoff_filename"]+" has "+str(params["total_rows"])+" rows.\n\n"
    n=n+"PID:"+pid+" Mutation rate 1/"+str(mut_rate)+".   \n"
  #  r.outfile.write("Genetic algorithm. Started at: "+starttime+"\n\n\n")
   # r.results.write("Genetic algorithm. Started at: "+starttime+"\n\n\n")

  #  outfile=open("outfile.txt","a")
  #  results=open("results.txt","a")


        
####################################


   # constraints=[False,1000000,-100000,False,1000000,-100000,False,1000000,-100000,False,1000000,-100000,False,1000000,-100000,False,1000000,-100000,False,1000000,-100000,False,1000000,-100000,False,1000000,-100000]
#constraints = dict([
#    (<key>, <value>),
#    (<key>, <value),
#      .
#      .
#      .
#    (<key>, <value>)
#]) 
##
##    pconstrain=constraints[0]
##    minp=constraints[1]
##    maxp=constraints[2]
##    aconstrain=constraints[3]
##    mina=constraints[4]
##    maxa=constraints[5]
##    bconstrain=constraints[6]
##    minb=constraints[7]
##    maxb=constraints[8]
##    cconstrain=constraints[9]
##    minc=constraints[10]
##    maxc=constraints[11]
##    dconstrain=constraints[12]
##    mind=constraints[13]
##    maxd=constraints[14]
##    econstrain=constraints[15]
##    mine=constraints[16]
##    maxe=constraints[17]
##    fconstrain=constraints[18]
##    minf=constraints[19]
##    maxf=constraints[20]
##    gconstrain=constraints[21]
##    ming=constraints[22]
##    maxg=constraints[23]
##
##
##   # bits=[alen+blen+clen+dlen+elen+flen+glen, 2**alen,2**blen,2**clen,2**dlen,2**elen,2**flen,2**glen]
##
##    no_of_alleles=bits[0]
##    arange=bits[1]
##    brange=bits[2]
##    crange=bits[3]
##    drange=bits[4]
##    erange=bits[5]
##    frange=bits[6]
##    grange=bits[7]
##    








#####################################

    #print("\n")
    if constraints["pconstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["minp"])+" <= payoff/cost <= "+str(constraints["maxp"])+"\n"
    #        r.outfile.write(str(r.minp)+" <= payoff/cost <= "+str(r.maxp)+"\n")
     #       r.results.write(str(r.minp)+" <= payoff/cost <= "+str(r.maxp)+"\n")
     #       if r.advanced_diag!="s": 
     #           print(str(r.minp)+" <= payoff/cost <= "+str(r.maxp)+"\n")

    if constraints["aconstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["mina"])+" <= a <= "+str(constraints["maxa"])+"\n"
      #      r.outfile.write(str(r.mina)+" <= a <= "+str(r.maxa)+"\n")
       #     r.results.write(str(r.mina)+" <= a <= "+str(r.maxa)+"\n")
       #     if r.advanced_diag!="s": 
       #         print(str(r.mina)+" <= a <= "+str(r.maxa)+"\n")

    if constraints["bconstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["minb"])+" <= b <= "+str(constraints["maxb"])+"\n"
       #     r.outfile.write(str(r.minb)+" <= b <= "+str(r.maxb)+"\n")
       #     r.results.write(str(r.minb)+" <= b <= "+str(r.maxb)+"\n")
       #     if r.advanced_diag!="s": 
       #         print(str(r.minb)+" <= b <= "+str(r.maxb)+"\n")
            
    if constraints["cconstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["minc"])+" <= c <= "+str(constraints["maxc"])+"\n"
         #   r.outfile.write(str(r.minc)+" <= c <= "+str(r.maxc)+"\n")
        #    r.results.write(str(r.minc)+" <= c <= "+str(r.maxc)+"\n")
        #    if r.advanced_diag!="s": 
        #        print(str(r.minc)+" <= c <= "+str(r.maxc)+"\n")
            
    if constraints["dconstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["mind"])+" <= d <= "+str(constraints["maxd"])+"\n"
       #     r.outfile.write(str(r.mind)+" <= d <= "+str(r.maxd)+"\n")
        #    r.results.write(str(r.mind)+" <= d <= "+str(r.maxd)+"\n")
        #    if r.advanced_diag!="s": 
        #        print(str(r.mind)+" <= d <= "+str(r.maxd)+"\n")

    if constraints["econstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["mine"])+" <= e <= "+str(constraints["maxe"])+"\n"
      #      r.outfile.write(str(r.mine)+" <= e <= "+str(r.maxe)+"\n")
     #       r.results.write(str(r.mine)+" <= e <= "+str(r.maxe)+"\n")
      #      if r.advanced_diag!="s": 
       #         print(str(r.mine)+" <= e <= "+str(r.maxe)+"\n")

    if constraints["fconstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["minf"])+" <= f <= "+str(constraints["maxf"])+"\n"
       #     r.outfile.write(str(r.minf)+" <= f <= "+str(r.maxf)+"\n")
        #    r.results.write(str(r.minf)+" <= f <= "+str(r.maxf)+"\n")
      #      if r.advanced_diag!="s": 
      #          print(str(r.minf)+" <= f <= "+str(r.maxf)+"\n")

    if constraints["gconstrain"]:
            n=n+"PID:"+pid+" "+str(constraints["ming"])+" <= g <= "+str(constraints["maxg"])+"\n\n\n"
      #      r.outfile.write(str(r.ming)+" <= g <= "+str(r.maxg)+"\n\n\n")
      #      r.results.write(str(r.ming)+" <= g <= "+str(r.maxg)+"\n\n\n")
      #      if r.advanced_diag!="s": 
       #         print(str(r.ming)+" <= g <= "+str(r.maxg)+"\n\n\n")

#################################################


           
    if params["direction"]=="x":
            n=n+"PID:"+pid+" "+"MAXIMISING....\n\n"
       #     r.outfile.write("MAXIMISING....\n\n")
       #     r.results.write("MAXIMISING....\n\n")
       #     if r.advanced_diag!="s": 
       #         print("MAXIMISING.....")
    elif params["direction"]=="n":
            n=n+"PID:"+pid+" "+"MINIMISING....\n\n"
     #       r.outfile.write("MINIMISING....\n\n")
      #      r.results.write("MINIMISING....\n\n")
      #      if r.advanced_diag!="s": 
       #         print("MINIMISING.....")
    else:
            print("direction error.")
            outfile.close()
            sys.exit()
            
    n=n+"===================================================\n\n\n"

    m=n    # m is the diagnostic narrative
       
    #r.outfile.write("===================================================\n\n\n")
    #r.outfile.flush()
    #r.results.write("===================================================\n\n\n")
    #r.results.flush()

#######################################################


    allele_len="S"+str(bits["no_of_alleles"])
    total_generations=0



    individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
    poparray = np.zeros(params["pop_size"], dtype=individual) 
    population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")



##    ploidy=2  # number of chromosomes per individual.  To increase this you will need to change the dataframes also!
##  #  pop_size=400  # population size
##  #  payoff_filename=sys.argv[1]  #  "shopsales4.csv"
##   # total_rows=count_file_rows(payoff_filename)
##  #  number_of_cols=8   # 7 input vars and 1 out  = 8
##   # row_find_method="l"
##    extra_eol_char=0
##    linewidth=76
##
##   
##    found=False
##    rowno=0
##    best_rowno=0
##
##    besta=0
##    bestb=0
##    bestc=0
##    bestd=0
##    beste=0
##    bestf=0
##    bestg=0
## 
##    epoch_length=80
##    no_of_epochs=4
##    total_generations=0
  #  total_total_generations=0
##    
##    
##    scaling_factor=25  # 25 scaling figure is the last. this is related to the size of the genepool. this multiplies the payoff up so that diversity is not lost on the probability_table when probs are rounded
##    actual_scaling=scaling_factor*pop_size
##  #  min_scaling=100  # when minimising, an extra factor is needed to prevent rounding cutting the diverity of the probability_table
##    
##    mutation_count=0
##    mutations=0
##    mutation_rate=400   # mutate 1 bit in every 400.  but the mutation is random 0 or 1 so we need to double the try to mutate rate. but there are also 2 chromos
##
##    advanced_diag="a"
##    
##
##    outfile=""
##    results=""
##    cls=""



    max_payoff=-10000000.0
    min_payoff=10000000.0
    max_fittest=""
    min_fittest=""
    best_epoch=1
    best_rowno=0
    best_gen=1
    best_fittest=""
    update_fittest=True
    
 #   no_of_epochs=0
 #   epoch_length=100
    
    besta=0
    bestb=0
    bestc=0
    bestd=0
    beste=0
    bestf=0
    bestg=0

    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    g=0


    for epoch in range(1,params["no_of_epochs"]+1):
 
        population=build_population(params["pop_size"],bits["no_of_alleles"],population)
      #  print("len population=",len(population))

    #    print("Epoch=",epoch," - generations in epoch=",epoch_length)
      #  print("\nStarting population len=",len(population),"\n")
      #  input("?")
        # print(population)



        bestpopulation=population.copy()


        fittest=""
       # best_rowno=0
        mutations=0
        totalp=0.0
        averagep=0.0
        elements_count=0
       # best_gen=1
        best_copy=False
        len_wheel=0

        if params["direction"]=="x":
            p=max_payoff
        elif params["direction"]=="n":
            p=min_payoff
        else:
            p=0.0
            print("direction error")
            sys.exit()
            
        plist=[]

      #  best_flag=True
        best=""
        fittest=""
            # element_list=[]
        len_probability_table=0    
        
        for generation in range(1,params["epoch_length"]+1):
            mutation_count=0


            #elements_count=0
            #averagep=0
            #totalp=0
            count=0
            returned_payoff=0
           # best=""
            #best_rowno=0
            #fittest=""
            #row_find_method="l"
            found=False   # found flag is a True if dna is found and a bestrow returned
           # best_flag=True
            
            
         #   print("generation=",generation)
         #   print("Epochs:",epoch+1,"Generation progress: [%d%%]" % (generation/epoch_length*100),"diversity (len_sel)=",len_sel,"Tot gene bits:%d" % total_genes,"Tot Mutations:%d      "  % (mutations), end='\r', flush=True)


            # update all the fitness values in the population based on the payoff file
           # population,found,rowno,fittest,returned_payoff,best_gen,best_rowno,max_fittest,max_payoff=calc_fitness(population,generation,direction,max_fittest,max_payoff,min_fittest,min_payoff,best_rowno,payoff_filename)  # use linecache method for fin row no in payoff file.csv



            #def calc_fitness(newpopulation,gen,direction,max_fittest,max_payoff,min_fittest,min_payoff,best_rowno,payoff_filename):   # the whole population 
                #  loop through the whole population dna and update the fitness based on the payoff file
     

            #newpopulation = population.copy(deep=True)

           # best_flag=False
            size=len(population)
          #  bestpopulation=population
           # print("len genepool=",size)
           # best_copy=False   # copy the population flag is set to true if the generation contains the highest fitness found so far

 

            
            for gene in range(0,size-1):  #newpopulation.iterrows():

                plist=[]
                pfound=False
                fittest=population.loc[gene,"expressed"]   # binary base turned into integer
              #  val=int(fittest,2)   # binary base turned into integer
                #rowno=val
                population.loc[gene, "fitness"] = p



       
###############################################################

                
                    #  print("val=",val)
              #  if val <= r.total_rows:   #total_rows:
  
                
                try:

                 #   if params["row_find_method"]=="c":
                 #       plist,pfound=return_a_row_from_envir_using_concatno(fittest,env)
                 #   elif params["row_find_method"]=="r":
                 #       plist,pfound=return_a_row_from_envir_using_rowno(int(fittest,2),env)
                 #   elif params["row_find_method"]=="s":
                 #       plist,pfound=return_a_row_from_file(int(fittest,2),r)
                 #   elif params["row_find_method"]=="l":
                     plist,pfound=return_a_row_from_linecache(int(fittest,2),params["payoff_filename"])                  
                  #  else:
                  #      print("find method code error.")
                  #      pass


                # indexerror needed here? 
                except IndexError:
                    print("index error finding p in calc fitness. fittest=",fittest)
                    sys.exit()
                except ValueError:
                    print("value error finding p in calc fitness. fittest=",fittest)
                    sys.exit()
                except IOError:
                    print("File IO error on ",params["payoff_filename"],"finding fittest=",fittest)
                    sys.exit()


              #  print("plist=",plist)
              #  input("?")
                if pfound and len(plist)>=params["no_of_cols"]:
                    found=True
             #       print("found plist=",plist)
                    
                    a=int(plist[0])
                    b=int(plist[1])
                    c=int(plist[2])
                    d=int(plist[3])
                    e=int(plist[4])
                    f=int(plist[5])
                    g=int(plist[6])
                    p=float(plist[7])

                    if constraints["pconstrain"] and (p<constraints["minp"] or p>constraints["maxp"]): 
                        pass   # p is outside of constraints, move on to next gene
                    else:
                        if constraints["aconstrain"] and (a<constraints["mina"] or a>constraints["maxa"]): 
                            pass   # a is outside of constraints, move on to next gene
                        else:
                                if constraints["bconstrain"] and (b<constraints["minb"] or b>constraints["maxb"]): 
                                    pass   # b is outside of constraints, move on to next gene
                                else:
                                    if constraints["cconstrain"] and (c<constraints["minc"] or c>constraints["maxc"]): 
                                        pass   # a is outside of constraints, move on to next gene
                                    else:
                                        if constraints["dconstrain"] and (d<constraints["mind"] or d>constraints["maxd"]): 
                                            pass   # d is outside of constraints, move on to next gene
                                        else:
                                            if constraints["econstrain"] and (e<constraints["mine"] or e>constraints["maxe"]): 
                                                pass   # e is outside of constraints, move on to next gene
                                            else:
                                                if constraints["fconstrain"] and (f<constraints["minf"] or f>constraints["maxf"]): 
                                                    pass   # f is outside of constraints, move on to next gene
                                                else:
                                                    if constraints["gconstrain"] and (g<constraints["ming"] or g>constraints["maxg"]): 
                                                        pass   # g is outside of constraints, move on to next gene
                                                    else:
                                                        
###################################################

                                                        #totalp+=p

                                                        #if p>returned_payoff:
                                                        returned_payoff=p
                                                       # try:
                                                        #    fittest=population.loc[val,"expressed"]   # if not constrained at all
                                                           # except IndexError:
                                                            #    print("index error. val=",val)

                        
                                              #          rowno=val
                                                       # best_rowno=val

                                                        elements_count+=1

                                                        #newpopulation.loc[row_number,["fitness"]]=p
                                                       # newpopulation.loc[int(newpopulation.loc[gene,"expressed"],2), "fitness"] = p
                                                #        population.loc[val, "fitness"] = p
                                                        population.loc[gene, "fitness"] = p

                                                       # best_copy=False

                                                        if params["direction"]=="x":  # maximising payoff
                                                            if p>max_payoff:
                            
                                                                      #  fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                                                max_payoff=p
                                                                max_fittest=fittest
                                                                best_rowno=gene   #rowno
                                                                best_gen=generation
                                                                best_epoch=epoch
                                                                best_fittest=fittest
                                                               # bestflag=True
                                                                #bestpopulation=population.copy()
                                                                best_copy=True
                                                                besta=a
                                                                bestb=b
                                                                bestc=c
                                                                bestd=d
                                                                beste=e
                                                                bestf=f
                                                                bestg=g
                                                            else:
                                                               #  best_flag=False
                                                                 pass
                                                        elif params["direction"]=="n":
                                                            if p<min_payoff:
                                
                                                               # fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                                                min_payoff=p
                                                                min_fittest=fittest
                                                                best_rowno=gene  #rowno
                                                                best_gen=generation
                                                                best_epoch=epoch
                                                                best_fittest=fittest
                                                             #   best_flag=True
                                                              #  bestpopulation=population.copy()
                                                                best_copy=True
                                                                besta=a
                                                                bestb=b
                                                                bestc=c
                                                                bestd=d
                                                                beste=e
                                                                bestf=f
                                                                bestg=g

                                                            else:
                                                                 #best_flag=False
                                                                 pass
                                                        else:
                                                            print("direction error")
                                                            sys.exit()




                                                            
                else:
                    pass
              #      n=n+"PID:"+pid+" "+"Row "+fittest+" not found in environment.\n"
           #         r.outfile.write("Row "+fittest+" not found in environment.\n")

                    
  #              else:
   #                 pass
    #                print("\nval ",val," is greater than total environment (",r.total_rows,")")
                  #  print("#",count+1,":",elem," val=",val,"payoff=",p," fitness=",fitness[count])
                 #   print("bestrow=",bestrow,"best=",best," max",max_payoff)


###################################################################  


                count+=1

               # if elements_count>0:
               #     averagep=totalp/elements_count

            if best_copy:
                best_copy=False
                update_fittest=True
                bestpopulation=population.copy()
 



            
            allcount=(population['expressed']).count()
            avefitness=((population['fitness']).sum())/allcount
         #   if params["advanced_diag"]=="a":   # extra advance detail
            dupcount=allcount-(((population['expressed']).drop_duplicates()).count())
        # count unique parents from parentid1 and parentid2
            nondup_par1=(population['parentid1'].drop_duplicates()).count()   #+((population['parentid1']).drop_duplicates()).count()
            nondup_par2=(population['parentid2'].drop_duplicates()).count()   #+((population['parentid1']).drop_duplicates()).count()


     #       Counter(probability_table).keys() # equals to list(set(words))
      #      Counter(probability_table).values() # counts the elements' frequency




           # if params["advanced_diag"]=="s": # silent mode
           #     pass
           # else:


           
            tmp=sp.call(params["cls"],shell=True)  # clear screen 'use 'clear for unix, cls for windows

            print("Genetic algorithm. By Anthony Paech.")
            print("Started at:",datetime.datetime.now())
            if params["direction"]=="x":
                print("\nMAXIMISING.....")
            elif params["direction"]=="n":
                print("\nMINIMISING.....")
            if constraints["pconstrain"]:
           # outfile.write(str(minp)+" <= payoff/cost <= "+str(maxp)+"\n")
                print(str(constraints["minp"])," <= payoff/cost <= ",str(constraints["maxp"]))

            if constraints["aconstrain"]:
           #     outfile.write(str(mina)+" <= a <= "+str(maxa)+"\n")
                print(str(constraints["mina"])," <= a <= ",str(constraints["maxa"]))

            if constraints["bconstrain"]:
                #outfile.write(str(minb)+" <= b <= "+str(maxb)+"\n")
                print(str(constraints["minb"])," <= b <= ",str(constraints["maxb"]))
        
            if constraints["cconstrain"]:
                #outfile.write(str(minc)+" <= c <= "+str(maxc)+"\n")
                print(str(constraints["minc"])," <= c <= ",str(constraints["maxc"]))
        
            if constraints["dconstrain"]:
                #outfile.write(str(mind)+" <= d <= "+str(maxd)+"\n")
                print(str(constraints["mind"])," <= d <= ",str(constraints["maxd"]))

            if constraints["econstrain"]:
                #outfile.write(str(mine)+" <= e <= "+str(maxe)+"\n")
                print(str(constraints["mine"])," <= e <= ",str(constraints["maxe"]))

            if constraints["fconstrain"]:
               # outfile.write(str(minf)+" <= f <= "+str(maxf)+"\n")
                print(str(constraints["minf"])," <= f <= ",str(constraints["maxf"]))

            if constraints["gconstrain"]:
                #outfile.write(str(ming)+" <= g <= "+str(maxg)+"\n\n\n")
                print(str(constraints["ming"])," <= g <= ",str(constraints["maxg"]))
     
            print("======================================")




            if params["direction"]=="x":    # maximise
               # if params["advanced_diag"]=="s":  # silent mode
               #     pass
               # else:
                print("Epoch",epoch,"of",params["no_of_epochs"])
                print("Epoch progress:[%d%%]" % (generation/params["epoch_length"]*100)," Generation no:",generation,"/",params["epoch_length"],"  Average fitness of this gen:",avefitness," max payoff->",max_payoff)

     #          if params["advanced_diag"]=="a":
                m=m+"\nGenepool. Ave Fitness="+str(avefitness)+" #Duplicates expressed="+str(dupcount)+" of "+str(allcount)+" . Diversity of probability_table weighting=["+str(len_wheel)+"]. probability_table[] size="+str(len_probability_table)+" \n"
                m=m+"\n"+str(nondup_par1)+" unique first parents, and "+str(nondup_par2)+" unique second parents of "+str(allcount)+" chomosomes.\n"
                m=m+"\nScaling factor="+str(params["scaling_factor"])+" * genepool size "+str(params["pop_size"])+" = actual scaling:"+str(params["actual_scaling"])+" \n"

          #      if params["advanced_diag"]=="s":  # silent mode
#                    print("\rTotal Progress:[%d%%]" % (round(((total_generations+1)/(r.epoch_length*r.no_of_epochs))*100)),end="\r",flush=True)

                if params["batch_run"]=="y":
                    print("\rBATCH Total Progress:[%d%%]" % (round(((params["total_total_generations"]+1)/(params["epoch_length"]*params["no_of_epochs"]*2*(bits["arange"]+bits["brange"]+bits["crange"]+bits["drange"]+bits["erange"]+bits["frange"]+bits["grange"])))*100)),end="\r",flush=True)
                else:    
                    print("\rTotal Progress:[%d%%]" % (round(((total_generations+1)/(params["epoch_length"]*params["no_of_epochs"]))*100)),end="\r",flush=True)
                        
##                else:
##                    #print("\n",returned_payoff," payoff.")
##                    print("\nFittest inputs: a=",a," b=",b," c=",c," d=",d," e=",e," f=",f," g=",g,"of this generation.")
##                    print("\n======================================")        
##                    print("\nFittest so far:",best_fittest," best rowno [",best_rowno,"] in best generation [",best_gen,"] in best epoch [",best_epoch,"] max payoff",max_payoff)
##                    print("\nBest overall so far: a=",besta," b=",bestb," c=",bestc," d=",bestd," e=",beste," f=",bestf," g=",bestg)
##                    print("\nTotal Progress:[%d%%]" % (round(((total_generations+1)/(params["epoch_length"]*params["no_of_epochs"]))*100)),"\n",flush=True)

                
                m=m+"Epoch "+str(epoch)+"/"+str(params["no_of_epochs"])+" Gen:[%d%%] " % (generation/params["epoch_length"]*100)+" generation # "+str(generation)+" fittest of this generation "+fittest+"="+str(returned_payoff)+" best="+str(max_fittest)+"\n"
                m=m+"Best epoch "+str(best_epoch)+" best gen "+str(best_gen)+" best row no "+str(best_rowno)+" max pay off "+str(max_payoff)+"\n"
                n=n+"PID:"+pid+" "+"Best epoch "+str(best_epoch)+" best gen "+str(best_gen)+" best row no "+str(best_rowno)+" max pay off "+str(max_payoff)+"\n"

          #      if params["advanced_diag"]=="a":
                m=m+"Diversity of probability_table weighting=["+str(len_wheel)+"]   len probability_table[]="+str(len_probability_table)+"\n"
      #          outfile.write(" keys in probability_table "+Counter(probability_table).keys()+"\n") # equals to list(set(words))
       #        outfile.write(" frequency of values in probability_table "+Counter(probability_table).values()+"\n") # counts the elements' frequency
                m=m+"Genepool. Ave fitness= "+str(avefitness)+" #duplicates expressed="+str(dupcount)+" of "+str(allcount)+"\n"
                m=m+str(nondup_par1)+" unique first parents, and "+str(nondup_par2)+" unique second parents of "+str(allcount)+" chomosomes.\n"
                m=m+"Scaling factor="+str(params["scaling_factor"])+" * genepool size "+str(params["pop_size"])+" = actual scaling: "+str(params["actual_scaling"])+"\n"

                m=m+"PID:"+pid+" "+"Current a="+str(a)+" b="+str(b)+" c="+str(c)+" d="+str(d)+" e="+str(e)+" f="+str(f)+" g="+str(g)+"\n"
                m=m+"PID:"+pid+" "+"Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"] max payoff "+str(max_payoff)+"\n"
                m=m+"PID:"+pid+" "+"Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n"



           #     m=m+"Current a="+str(a)+" b="+str(b)+" c="+str(c)+" d="+str(d)+" e="+str(e)+" f="+str(f)+" g="+str(g)+"\n"
      #          outfile.write("Fittest so far "+str(best_fittest)+" best epoch "+str(best_epoch)+" best generation in best epoch "+str(best_gen)+" best row no "+str(best_rowno)+" max pay off "+str(max_payoff)+"\n")
          #      m=m+"Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"] max payoff "+str(max_payoff)+"\n"
          #      m=m+"Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n"
                if update_fittest:
                    update_fittest=False
                    n=n+"Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"]\nMax payoff "+str(max_payoff)+"\n"
                    n=n+"Genepool average fitness= "+str(avefitness)+"\n From pop_size:{"+str(params["pop_size"])+"}    \n"
                    n=n+"Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n"

            elif params["direction"]=="n":  # minimise
             #   if params["advanced_diag"]=="s":  # silent mode
             #       pass
             #   else:
             
                print("Epoch",epoch,"of",params["no_of_epochs"])
                print("Epoch progress:[%d%%]" % (generation/params["epoch_length"]*100),"Generation no:",generation,"/",params["epoch_length"],"  Average fitness of this gen:",avefitness," min cost->",min_payoff)

#                if params["advanced_diag"]=="a":
                m=m+"\nGenepool. Ave fitness="+str(avefitness)+" #Duplicates expressed="+str(dupcount)+" of "+str(allcount)+" . Diversity of probability_table weighting=["+str(len_wheel)+"].  probability_table[] size="+str(len_probability_table)+" \n"
                m=m+"\n"+str(nondup_par1)+" unique first parents, and "+str(nondup_par2)+" unique second parents of "+str(allcount)+" chomosomes.\n"
         #       print("\n Genepool size",pop_size," * min_scaling factor",min_scaling," / Scaling factor=",scaling_factor," = actual scaling:",(pop_size*min_scaling)/scaling_factor)
                m=m+"\nScaling factor="+str(params["scaling_factor"])+" * genepool size "+str(params["pop_size"])+" = actual scaling:"+str(params["actual_scaling"])+"\n"

         #   if params["advanced_diag"]=="s":  # silent mode
                if params["batch_run"]=="y":
                    print("\rBATCH Total Progress:[%d%%]" % (round(((params["total_total_generations"]+1)/(params["epoch_length"]*params["no_of_epochs"]*2*(bits["arange"]+bits["brange"]+bits["crange"]+bits["drange"]+bits["erange"]+bits["frange"]+bits["grange"])))*100)),end="\r",flush=True)
                else:    
                    print("\rTotal Progress:[%d%%]" % (round(((total_generations+1)/(params["epoch_length"]*params["no_of_epochs"]))*100)),end="\r",flush=True)
##                else:
##                  #  print("\nRow number",rowno,"=",returned_payoff," cost.")
##                    print("\nFittest inputs: a=",a," b=",b," c=",c," d=",d," e=",e," f=",f," g=",g,"of this generation.")
##                    print("\n======================================")        
###                   print("\n\nFittest so far:",min_fittest,"best epoch:[",best_epoch,"] best generation in best epoch [",best_gen,"] best rowno [",best_rowno,"] min cost",min_payoff)
##                    print("\nFittest so far:",best_fittest," best rowno [",best_rowno,"] in best generation [",best_gen,"] in best epoch [",best_epoch,"] min cost",min_payoff)
##                    print("\nBest overall so far: a=",besta," b=",bestb," c=",bestc," d=",bestd," e=",beste," f=",bestf," g=",bestg)
##                    print("\nTotal Progress:[%d%%]" % (round(((total_generations+1)/(params["epoch_length"]*params["no_of_epochs"]))*100)),"\n",flush=True)

                m=m+"Epoch "+str(epoch)+"/"+str(params["epoch_length"])+" Gen:[%d%%] " % (generation/params["epoch_length"]*100)+" generation # "+str(generation)+" fittest of this generation "+fittest+"="+str(returned_payoff)+" best="+str(min_fittest)+"\n"
                m=m+"Best epoch "+str(best_epoch)+" best gen "+str(best_gen)+" best row no "+str(best_rowno)+" min cost "+str(min_payoff)+"\n"
                n=n+"PID:"+pid+" "+"Best epoch "+str(best_epoch)+" best gen "+str(best_gen)+" best row no "+str(best_rowno)+" min cost "+str(min_payoff)+"\n"

         #   if params["advanced_diag"]=="a":
                m=m+"Diversity of probability_table weighting=["+str(len_wheel)+"]   len probability_table[]="+str(len_probability_table)+"\n"
    #           outfile.write(" keys in probability_table "+Counter(probability_table).keys()+"\n") # equals to list(set(words))
    #            outfile.write(" frequency of values in probability_table "+Counter(probability_table).values()+"\n") # counts the elements' frequency
                m=m+"Genepool. Ave fitness="+str(avefitness)+" #duplicates expressed="+str(dupcount)+" of "+str(allcount)+"\n"
                m=m+str(nondup_par1)+" unique first parents, and "+str(nondup_par2)+" unique second parents of "+str(allcount)+" chomosomes.\n"
   #             outfile.write("Genepool size "+str(pop_size)+" * min_scaling factor "+str(min_scaling)+" / scaling factor "+str(scaling_factor)+" = actual scaling: "+str((pop_size*min_scaling)/scaling_factor)+"\n")
                m=m+"Scaling factor="+str(params["scaling_factor"])+" * genepool size "+str(params["pop_size"])+" = actual scaling: "+str(params["actual_scaling"])+"\n"

                m=m+"PID:"+pid+" "+"Current a="+str(a)+" b="+str(b)+" c="+str(c)+" d="+str(d)+" e="+str(e)+" f="+str(f)+" g="+str(g)+"\n"
                m=m+"PID:"+pid+" "+"Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"] min payoff "+str(min_payoff)+"\n"
                m=m+"PID:"+pid+" "+"Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n"


           #     m=m+"Current a="+str(a)+" b="+str(b)+" c="+str(c)+" d="+str(d)+" e="+str(e)+" f="+str(f)+" g="+str(g)+"\n"
         #       outfile.write("Fittest so far "+str(best_fittest)+" best epoch "+str(best_epoch)+" best generation in best epoch "+str(best_gen)+" best row no "+str(best_rowno)+" min cost "+str(min_payoff)+"\n")
            #    m=m+"Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"] min cost "+str(min_payoff)+"\n"
            #    m=m+"Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n"
                if update_fittest:
                    update_fittest=False
                    n=n+"Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"]\nMin cost "+str(min_payoff)+"\n"
                    n=n+"Genepool average fitness= "+str(avefitness)+"\n From pop_size:{"+str(params["pop_size"])+"}    \n"
                    n=n+"Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n"


            else:
                print("direction error")
                sys.exit()
 
    #        outfile.flush()
    #        results.flush()
            
            probability_table=calc_mating_probabilities(population,params)
            # scaling figure is the last.  this multiplies the payoff up so that divsity is not lost on the probability_table when probs are rounded

            len_probability_table=len(probability_table)
            if len_probability_table==0:
                    print("probability_table empty")
                    sys.exit
            #print(probability_table)


            #if params["advanced_diag"]=="a":
            #print(" Keys in probability_table",Counter(probability_table).keys()) # equals to list(set(words))
            #print(" Frequency of values in probability_table",Counter(probability_table).values()) # counts the elements' frequency
            m=m+" Keys in probability_table "+str(Counter(probability_table).keys())+"\n" # equals to list(set(words))
            m=m+" Frequency of values in probability_table "+str(Counter(probability_table).values())+"\n\n" # counts the elements' frequen


            mates,len_wheel=spin_the_mating_wheel(probability_table,population,params["pop_size"],params["direction"])  # wheel_len is the size of the unique gene pool to select from in the probability_table

               
            population=crossover(mates,bits["no_of_alleles"],individual)   # simple crossover
              

            population, mutation_count, whichchromo,whichmember=mutate(population,bits["no_of_alleles"],params["ploidy"],params["pop_size"],mut_rate)   # 1000 means mutation 1 in a 1000 bits processed


      #      if params["advanced_diag"]=="a":
       #         print("Mutation rate: 1/",mut_rate,"alleles. Mutation in individuals:",whichmember)
             #   print(" member id=",whichmember)
            m=m+"Mutation rate: 1/"+str(mut_rate)+" alleles. Mutation in individuals:"+str(whichmember)+"\n\n\n"


            if params["batch_run"]=="y":
                print("\rBATCH Total Progress:[%d%%]" % (round(((params["total_total_generations"]+1)/(params["epoch_length"]*params["no_of_epochs"]*2*(bits["arange"]+bits["brange"]+bits["crange"]+bits["drange"]+bits["erange"]+bits["frange"]+bits["grange"])))*100)),end="\r",flush=True)



        #    print("after mutation pop size=",len(population))
            for p in range(0,params["pop_size"]-1):    #fill out the express column for the population:
                population.loc[p,"expressed"]=express(population.loc[p,"chromo1"],population.loc[p,"chromo2"])

        
              #  total_genes=total_genes+gene_pool_size
            mutations=mutations+mutation_count
             #   generation_number+=1

     #   else:
       #     pass   # nothing found
            


            total_generations+=1
            params["total_total_generations"]+=1

       # if params["advanced_diag"]!="s":    
       #     print("")


    print("\nPID:"+pid+" Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"]")
    if params["direction"]=="x":
        print("Max payoff:"+str(max_payoff))
    else:          
        print("Min cost:"+str(min_payoff))
    print("PID:"+pid+" Genepool average fitness= "+str(avefitness)+"\n From pop_size:{"+str(params["pop_size"])+"}"+" Mutation rate 1/"+str(mut_rate)+"\n")
    print("PID:"+pid+" Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg))


    m=m+"\nOptimisation GA finished.\n\n"
    m=m+"==========================================\n\n"
    m=m+"\n\nBest Population:"
    m=m+"\n\n\n"+bestpopulation.to_string()
    m=m+"\n\n\n"


    n=n+"\nPID:"+pid+" Optimisation GA finished.\n\n"
    n=n+ "==========================================\n\n"
 #   n=n+"\n\nPID:"+pid+" "+"Best Population:"


 #   n=n+"\n\n\n"+bestpopulation.to_string()
    n=n+"\n\n\n"
    if params["advanced_diag"]=="y":
        add_to_queue(m,q)   # diagnotistic string to file
        time.sleep(1)
    
    add_to_queue(n,q)   # results string to file

    time.sleep(3)
    
  #  outfile.close()
 #   results.close()


    return(params["total_total_generations"])


###########################################






def main():
    freeze_support()
    
  #  filename="GAresults.txt"

    if(len(sys.argv) !=9 ) :
        print("Usage : python GA_xxxxx.py filename bitlen1 bitlen2 bitlen3 bitlen4 bitlen5 bitlen6 bitlen7")
        sys.exit()


    #class ga(object):
     #   pass

##    ploidy=2  # number of chromosomes per individual.  To increase this you will need to change the dataframes also!
 #   pop_size=400  # population size
 #   payoff_filename=sys.argv[1]  #  "shopsales4.csv"
##    total_rows=count_file_rows(ga.payoff_filename)
 #   no_of_cols=8   # 7 input vars and 1 out  = 8
##    row_find_method="l"
##    extra_eol_char=0
##    linewidth=76


    params = dict(
        pop_size=400,
        payoff_filename=sys.argv[1],
        no_of_cols=8,
        ploidy=2,
        row_find_method="l",
        advanced_diag="n",
        extra_eol_char=0,
        linewidth=76,
        total_rows=0,
        epoch_length=20,
        no_of_epochs=1,
        total_generations=0,
        total_total_generations=0,
        direction="x",
        batch_run="n",
        cls="",
        scaling_factor=25,  # 25 scaling figure is the last. this is related to the size of the genepool. this multiplies the payoff up so that diversity is not lost on the probability_table when probs are rounded
        actual_scaling=0,  
        mutation_count=0,
        mutations=0,
        mutation_rate=[800,700,600,500,400,300,200],   # different mutation rates mutate 1 bit in every 400.  but the mutation is random 0 or 1 so we need to double the try to mutate rate. but there are also 2 chromos
        diagnostic_file="GA_diagnostic.txt",
        results_file="GA_results.txt")

     #   ga.c_over=""
     
     
    params["total_rows"]=count_file_rows(params["payoff_filename"])       
    params["actual_scaling"]=params["scaling_factor"]*params["pop_size"]        

#    constraints=dict[False,1000000,100000,False,1000000,100000,False,1000000,100000,False,1000000,100000,False,1000000,100000,False,1000000,100000,False,1000000,100000,False,1000000,100000,False,1000000,100000]

    constraints = dict(
        pconstrain=False,
        minp=1000000,
        maxp=-1000000,
        aconstrain=False,
        mina=1000000,
        maxa=-1000000,
        bconstrain=False,
        minb=1000000,
        maxb=-1000000,
        cconstrain=False,
        minc=1000000,
        maxc=-1000000,
        dconstrain=False,
        mind=1000000,
        maxd=-1000000,
        econstrain=False,
        mine=1000000,
        maxe=-1000000,
        fconstrain=False,
        minf=1000000,
        maxf=-1000000,
        gconstrain=False,
        ming=1000000,
        maxg=-1000000)


    
##    found=False
##    rowno=0
##    best_rowno=0
##
##    besta=0
##    bestb=0
##    bestc=0
##    bestd=0
##    beste=0
##    bestf=0
##    bestg=0
## 
##    epoch_length=80
##    no_of_epochs=4
##    total_generations=0
##    total_total_generations=0
##    pop_size=200
##    
##    direction="x"
##    scaling_factor=25  # 25 scaling figure is the last. this is related to the size of the genepool. this multiplies the payoff up so that diversity is not lost on the probability_table when probs are rounded
##    actual_scaling=scaling_factor*pop_size
##  #  min_scaling=100  # when minimising, an extra factor is needed to prevent rounding cutting the diverity of the probability_table
##    
##    mutation_count=0
##    mutations=0
##    mutation_rate=400   # mutate 1 bit in every 400.  but the mutation is random 0 or 1 so we need to double the try to mutate rate. but there are also 2 chromos
##
##    advanced_diag="a"
## #   ga.c_over=""
##    batch_run="n"
##
##    outfile=""
##    results=""
##    cls=""




################################################
# length in bits of each input variable

    alen=int(sys.argv[2])   # length in binary bits of each integer input
    blen=int(sys.argv[3])
    clen=int(sys.argv[4])
    dlen=int(sys.argv[5])
    elen=int(sys.argv[6])
    flen=int(sys.argv[7])
    glen=int(sys.argv[8])



   # no_of_alleles=alen+blen+clen+dlen+elen+flen+glen  # length of each chromosome
    
##    
##    ga.arange=2**ga.alen    
##    ga.brange=2**ga.blen
##    ga.crange=2**ga.clen
##    ga.drange=2**ga.dlen
##    ga.erange=2**ga.elen
##    ga.frange=2**ga.flen
##    ga.grange=2**ga.glen


    bits=dict(no_of_alleles=alen+blen+clen+dlen+elen+flen+glen,arange=2**alen,brange=2**blen,crange=2**clen,drange=2**dlen,erange=2**elen,frange=2**flen,grange=2**glen)


#################################################3
    
    if platform.system().strip().lower()[:7]=="windows":
        params["extra_eol_char"]=1
        params["cls"]="cls"
    else:
        params["extra_eol_char"]=0
        params["cls"]="clear"
## 
   
    tmp=sp.call(params["cls"],shell=True)  # clear screen use 'cls' for windows
    #tmp=sp.call('clear',shell=True)  # clear screen use 'clear' for unix


    
    print("Genetic algorithm. By Anthony Paech")
    print("===================================")
    print("Platform:",platform.machine(),"\n",platform.platform())
    #print("\n:",platform.processor(),"\n:",platform.version(),"\n:",platform.uname())
    print("Bit Length=",bits["no_of_alleles"],"-> Max CSV datafile rows available is:",2**bits["no_of_alleles"])
    print("Payoff/cost environment file "+str(params["payoff_filename"])+" has "+str(params["total_rows"])+" rows.\n\n")

  #  print("\nTheoretical max no of rows for the CSV datafile file:",payoff_filename,"is:",sys.maxsize)



    

    #individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
    #poparray = np.zeros(ga.pop_size, dtype=individual) 
    #population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")

   # population.index
  #  print(len(population))
   # print("\n\n")
   # print(population)


##    print("\n\n") 
##    answer=""
##    while answer!="y" and answer!="n":
##        answer=input("Create payoff env? (y/n)")
##    if answer=="y":
##        print("Creating payoff/cost environment....file:",ga.payoff_filename)
##        clock_start=time.process_time()
##
##          #      total_rows=generate_payoff_environment_7d_file(linewidth,0,2**1,0,2**1,0,2**1,0,2**1,0,2**1,0,2**6,0,2**4,payoff_filename)  
##       # 7 input variables 2 bits each = 14 bits, 1 floating point payoff.
##        ga.total_rows=generate_payoff_environment_7d_file(0,ga.arange,0,ga.brange,0,ga.crange,0,ga.drange,0,ga.erange,0,ga.frange,0,ga.grange,ga.payoff_filename)

   # clock_end=time.process_time()
   # duration_clock=clock_end-clock_start
   # print("generate payoff/cost environment - Clock: duration_clock =", duration_clock,"seconds.")
   # print("Payoff/cost environment file is:",payoff_filename,"and has",total_rows,"rows.\n\n")


########################################################33

#   load filename.csv into pandas for fast analysis

 #   print("\n")
   # row_find_method="l"
  #  while ga.row_find_method!="c" and ga.row_find_method!="r" and ga.row_find_method!="s" and ga.row_find_method!="l":
  #      ga.row_find_method=input("Use (c)oncatno or (r)owno or (l)inecache?")


 #   if ga.row_find_method=="s" or ga.row_find_method=="l":
  #      csv_envir = pd.DataFrame() #creates a new dataframe that's empty, we are not using it with the fileseek method


  #      if ga.row_find_method=="s":
  #          format_csv(ga)   # format the file to ga.linewidth length to allow the fast file seek method
  #          ga.payoff_filename="new"+ga.payoff_filename  #the input filename now has new in front of it
  

##       
##    else:    
##        print("loading ",ga.payoff_filename," to pandas")
##
##      #  chunksize = 10 ** 6
##      #  for chunk in pd.read_csv(payoff_filename,indexcol=0,dtype=csvenv,delimiter=',', header=None, chunksize=chunksize):
##            #process(chunk)
##            
##       # chunksize = 10 ** 6
##        csv_envir=pd.read_csv(ga.payoff_filename,names=["a","b","c","d","e","f","g","payoff"],delimiter=',', header=None)   #, chunksize=chunksize)
##     
##        # insert an index column which is blank
##        print("adding concatno column")
##        csv_envir.insert(0,"concatno","",allow_duplicates=False)
##
##       #print("adding rownono column")
##       # csv_envir.insert(0,"rowno","",allow_duplicates=False)
##       # csv_envir['rowno'] = np.arange(len(csv_envir))
##
##        print("calculating index value")
##        aforbin="{:0>"+str(ga.alen)+"}"
##        bforbin="{:0>"+str(ga.blen)+"}"
##        cforbin="{:0>"+str(ga.clen)+"}"
##        dforbin="{:0>"+str(ga.dlen)+"}"
##        eforbin="{:0>"+str(ga.elen)+"}"
##        fforbin="{:0>"+str(ga.flen)+"}"
##        gforbin="{:0>"+str(ga.glen)+"}"
##                            
##        csv_envir["concatno"]=csv_envir["a"].apply(bin).str[2:].apply(aforbin.format)+csv_envir["b"].apply(bin).str[2:].apply(bforbin.format)+csv_envir["c"].apply(bin).str[2:].apply(cforbin.format)+csv_envir["d"].apply(bin).str[2:].apply(dforbin.format)+csv_envir["e"].apply(bin).str[2:].apply(eforbin.format)+csv_envir["f"].apply(bin).str[2:].apply(fforbin.format)+csv_envir["g"].apply(bin).str[2:].apply(gforbin.format)    #str.zfill(3)
##        csv_envir.set_index("concatno")
##
##        print("Environment ready..")
##      #  print(csv_envir)
    ##    input("?")
    ##
    #    p=return_a_row_from_envir_using_rowno(88,csv_envir)
    #    print("p=",p)
    #    input("?")

      
                    




##############################################################

 #   print("\n")
 #   ap="n"
 #   while ap!="y" and ap!="n":
 #       ap=input("Append to existing outfile.txt? (y/n)")

  #  print("\n")
   # if ap=="y":
   #     print("appending to existing outfile.txt")
#    else:
 #       print("clearing outfile.txt")

 
##    diag_file=open(params["diagnostic_file"],"w")
##    diag_file.close()   # clear out file
##    
##    diag_file=open(params["diagnostic_file"],"a")


    with open(params["results_file"],"w") as results_file:
        pass
    #results_file.close()

   # with open(params["results_file"],"a") as results_file:      
    results_file=open(params["results_file"],"a")       


    
#    print("\n")
 #   ga.c_over="s"
#    while ga.c_over!="s" and ga.c_over!="p":
#        ga.c_over=input("(s)imple crossover or (p)mx?")


   # print("\n")
    params["advanced_diag"]=""
    while params["advanced_diag"]!="y" and params["advanced_diag"]!="n":
        params["advanced_diag"]=input("Do you want to log advanced diagnostics to file? (y/n)")

  #  print("\n")


  #  outfile.write("\nCounting rows in "+payoff_filename+"\n")
  #  total_rows=count_file_rows(payoff_filename)
    results_file.write("Payoff/cost environment file "+str(params["payoff_filename"])+" has "+str(params["total_rows"])+" rows.\n\n")

   # best_rowno=0
   # fittest=""

   


    print("\n")
    params["batch_run"]=""
    while params["batch_run"]!="y" and params["batch_run"]!="n":
        params["batch_run"]=input("Batch run (y/n)?")

    if params["batch_run"]=="n":
        params["direction"]=""
        while params["direction"]!="x" and params["direction"]!="n":
            params["direction"]=input("Ma(x)imise or Mi(n)imise?")


        print("\nSet constraints")
        print("===============")
################################# constrain payoff? 
        con=""
        constraints["maxp"]=0
        constraints["pconstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain payoff/cost? (y/n)")
            if con=="y":
                constraints["maxp"]=int(input("Maximum payoff/cost?"))
                constraints["minp"]=constraints["maxp"]+1
                while constraints["minp"]>constraints["maxp"]:
                    constraints["minp"]=int(input("Minimum payoff/cost?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["minp"],"<= payoff/cost <=",constraints["maxp"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    constraints["pconstrain"]=True
            else:
                correct="y"    

#################################### constrain a?

        con=""
        constraints["maxa"]=0
        constraints["aconstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column a? (y/n)")
            if con=="y":
                constraints["maxa"]=int(input("Maximum a?"))
                constraints["mina"]=constraints["maxa"]+1
                while constraints["mina"]>constraints["maxa"]:
                    constraints["mina"]=int(input("Minimum a?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["mina"],"<= a <=",constraints["maxa"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    constraints["aconstrain"]=True
            else:
                correct="y"    

######################################  constrain b?


        con=""    
        constraints["maxb"]=0
        constraints["bconstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column b? (y/n)")
            if con=="y":
                constraints["maxb"]=int(input("Maximum b?"))
                constraints["minb"]=constraints["maxb"]+1
                while constraints["minb"]>constraints["maxb"]:
                    constraints["minb"]=int(input("Minimum b?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["minb"],"<= b <=",constraints["maxb"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                   constraints["bconstrain"]=True
            else:
                correct="y"


###########################################  constrain c?
        
        con=""
        constraints["maxc"]=0
        constraints["cconstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column c? (y/n)")
            if con=="y":
                constraints["maxc"]=int(input("Maximum c?"))
                constraints["minc"]=constraints["maxc"]+1
                while constraints["minc"]>constraints["maxc"]:
                    constraints["minc"]=int(input("Minimum c?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["minc"],"<= c <=",constraints["maxc"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    constraints["cconstrain"]=True
            else:
                correct="y"    



##############################################  constrain d?
        con=""
        constraints["maxd"]=0
        constraints["dconstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column d? (y/n)")
            if con=="y":
                constraints["maxd"]=int(input("Maximum d?"))
                constraints["mind"]=constraints["maxd"]+1
                while constraints["mind"]>constraints["maxd"]:
                    constraints["mind"]=int(input("Minimum d?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["mind"],"<= d <=",constraints["maxd"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    constraints["dconstrain"]=True
            else:
                correct="y"    


##############################################  constrain e?
        con=""
        constraints["maxe"]=0
        constraints["econstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column e? (y/n)")
            if con=="y":
                constraints["maxe"]=int(input("Maximum e?"))
                constraints["mine"]=constraints["maxe"]+1
                while constraints["mine"]>constraints["maxe"]:
                    constraints["mine"]=int(input("Minimum e?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["mine"],"<= e <=",constraints["maxe"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    constraints["econstrain"]=True
            else:
                correct="y"


##############################################  constrain f?
        con=""
        constraints["maxf"]=0
        constraints["fconstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column f? (y/n)")
            if con=="y":
                constraints["maxf"]=int(input("Maximum f?"))
                constraints["minf"]=constraints["maxf"]+1
                while constraints["minf"]>constraints["maxf"]:
                    constraints["minf"]=int(input("Minimum f?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["minf"],"<= f <=",constraints["maxf"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    constraints["fconstrain"]=True
            else:
                correct="y"


##############################################  constrain g?
        con=""
        constraints["maxg"]=0
        constraints["gconstrain"]=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column g? (y/n)")
            if con=="y":
                constraints["maxg"]=int(input("Maximum g?"))
                constraints["ming"]=constraints["maxg"]+1
                while constraints["ming"]>constraints["maxg"]:
                    constraints["ming"]=int(input("Minimum g?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(constraints["ming"],"<= g <=",constraints["maxg"])
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    constraints["gconstrain"]=True
            else:
                correct="y"


###############################################################



        GA_mp(bits,constraints,params)  # multiprocessing




###########################################################
    else:
            
        #  batch run.  test every input at every setting both max and min

        for a in range(0,bits["arange"]):
            params["direction"]="x"
            constraints["pconstrain"]=False
            constraints["minp"]=0
            constraints["maxp"]=0
            constraints["aconstrain"]=True
            constraints["mina"]=a
            constraints["maxa"]=a
            constraints["bconstrain"]=False
            constraints["minb"]=0
            constraints["maxb"]=0
            constraints["cconstrain"]=False
            constraints["minc"]=0
            constraints["maxc"]=0
            constraints["dconstrain"]=False
            constraints["mind"]=0
            constraints["maxd"]=0
            constraints["econstrain"]=False
            constraints["mine"]=0
            constraints["maxe"]=0
            constraints["fconstrain"]=False
            constraints["minf"]=0
            constraints["maxf"]=0
            constraints["gconstrain"]=False
            constraints["ming"]=0
            constraints["maxg"]=0

            # maximise results written to outfile.txt
            #genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)
            params["direction"]="n"
            #minimise results written to outfile.txt
            GA_mp(bits,constraints,params)
         #   genetic_algorithm(ga,csv_envir)


        for b in range(0,bits["brange"]):
            params["direction"]="x"
            constraints["pconstrain"]=False
            constraints["minp"]=0
            constraints["maxp"]=0
            constraints["aconstrain"]=False
            constraints["mina"]=0
            constraints["maxa"]=0
            constraints["bconstrain"]=True
            constraints["minb"]=b
            constraints["maxb"]=b
            constraints["cconstrain"]=False
            constraints["minc"]=0
            constraints["maxc"]=0
            constraints["dconstrain"]=False
            constraints["mind"]=0
            constraints["maxd"]=0
            constraints["econstrain"]=False
            constraints["mine"]=0
            constraints["maxe"]=0
            constraints["fconstrain"]=False
            constraints["minf"]=0
            constraints["maxf"]=0
            constraints["gconstrain"]=False
            constraints["ming"]=0
            constraints["maxg"]=0

            # maximise results written to outfile.txt
         #   genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)
            params["direction"]="n"
            #minimise results written to outfile.txt
          #  genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)


        for c in range(0,bits["crange"]):
            params["direction"]="x"
            constraints["pconstrain"]=False
            constraints["minp"]=0
            constraints["maxp"]=0
            constraints["aconstrain"]=False
            constraints["mina"]=0
            constraints["maxa"]=0
            constraints["bconstrain"]=False
            constraints["minb"]=0
            constraints["maxb"]=0
            constraints["cconstrain"]=True
            constraints["minc"]=c
            constraints["maxc"]=c
            constraints["dconstrain"]=False
            constraints["mind"]=0
            constraints["maxd"]=0
            constraints["econstrain"]=False
            constraints["mine"]=0
            constraints["maxe"]=0
            constraints["fconstrain"]=False
            constraints["minf"]=0
            constraints["maxf"]=0
            constraints["gconstrain"]=False
            constraints["ming"]=0
            constraints["maxg"]=0

            # maximise results written to outfile.txt
         #   genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)
            params["direction"]="n"
            #minimise results written to outfile.txt
          #  genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)



        for d in range(0,bits["drange"]):
            params["direction"]="x"
            constraints["pconstrain"]=False
            constraints["minp"]=0
            constraints["maxp"]=0
            constraints["aconstrain"]=False
            constraints["mina"]=0
            constraints["maxa"]=0
            constraints["bconstrain"]=False
            constraints["minb"]=0
            constraints["maxb"]=0
            constraints["cconstrain"]=False
            constraints["minc"]=0
            constraints["maxc"]=0
            constraints["dconstrain"]=True
            constraints["mind"]=d
            constraints["maxd"]=d
            constraints["econstrain"]=False
            constraints["mine"]=0
            constraints["maxe"]=0
            constraints["fconstrain"]=False
            constraints["minf"]=0
            constraints["maxf"]=0
            constraints["gconstrain"]=False
            constraints["ming"]=0
            constraints["maxg"]=0

            # maximise results written to outfile.txt
            #genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)
            params["direction"]="n"
            #minimise results written to outfile.txt
            #genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)


        for e in range(0,bits["erange"]):
            params["direction"]="x"
            constraints["pconstrain"]=False
            constraints["minp"]=0
            constraints["maxp"]=0
            constraints["aconstrain"]=False
            constraints["mina"]=0
            constraints["maxa"]=0
            constraints["bconstrain"]=False
            constraints["minb"]=0
            constraints["maxb"]=0
            constraints["cconstrain"]=False
            constraints["minc"]=0
            constraints["maxc"]=0
            constraints["dconstrain"]=False
            constraints["mind"]=0
            constraints["maxd"]=0
            constraints["econstrain"]=True
            constraints["mine"]=e
            constraints["maxe"]=e
            constraints["fconstrain"]=False
            constraints["minf"]=0
            constraints["maxf"]=0
            constraints["gconstrain"]=False
            constraints["ming"]=0
            constraints["maxg"]=0

            # maximise results written to outfile.txt
         #   genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)
            params["direction"]="n"
            #minimise results written to outfile.txt
         #   genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)

        for f in range(0,bits["frange"]):
            params["direction"]="x"
            constraints["pconstrain"]=False
            constraints["minp"]=0
            constraints["maxp"]=0
            constraints["aconstrain"]=False
            constraints["mina"]=0
            constraints["maxa"]=0
            constraints["bconstrain"]=False
            constraints["minb"]=0
            constraints["maxb"]=0
            constraints["cconstrain"]=False
            constraints["minc"]=0
            constraints["maxc"]=0
            constraints["dconstrain"]=False
            constraints["mind"]=0
            constraints["maxd"]=0
            constraints["econstrain"]=False
            constraints["mine"]=0
            constraints["maxe"]=0
            constraints["fconstrain"]=True
            constraints["minf"]=f
            constraints["maxf"]=f
            constraints["gconstrain"]=False
            constraints["ming"]=0
            constraints["maxg"]=0

            # maximise results written to outfile.txt
          #  genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)
            params["direction"]="n"
            #minimise results written to outfile.txt
            #genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)


        for g in range(0,bits["grange"]):
            params["direction"]="x"
            constraints["pconstrain"]=False
            constraints["minp"]=0
            constraints["maxp"]=0
            constraints["aconstrain"]=False
            constraints["mina"]=0
            constraints["maxa"]=0
            constraints["bconstrain"]=False
            constraints["minb"]=0
            constraints["maxb"]=0
            constraints["cconstrain"]=False
            constraints["minc"]=0
            constraints["maxc"]=0
            constraints["dconstrain"]=False
            constraints["mind"]=0
            constraints["maxd"]=0
            constraints["econstrain"]=False
            constraints["mine"]=0
            constraints["maxe"]=0
            constraints["fconstrain"]=False
            constraints["minf"]=0
            constraints["maxf"]=0
            constraints["gconstrain"]=True
            constraints["ming"]=g
            constraints["maxg"]=g

            # maximise results written to outfile.txt
          #  genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)
            params["direction"]="n"
            #minimise results written to outfile.txt
          #  genetic_algorithm(ga,csv_envir)
            GA_mp(bits,constraints,params)




  #  diag_file.write("Finished at:"+str(datetime.datetime.now())+"\n\n")
    results_file.write("Finished at:"+str(datetime.datetime.now())+"\n\n")


  #  diag_file.close()
    results_file.close()
    print("\nCheck GA_results.txt for a diagnostic log. \nFinished at:",str(datetime.datetime.now()),"\n\n")
#    print("Final population\n")
 #   print(population.to_string())
    


# Warning from Python documentation:
# Functionality within this package requires that the __main__ module be
# importable by the children. This means that some examples, such as the
# multiprocessing.Pool examples will not work in the interactive interpreter.


 


if __name__ == '__main__':
    main()

