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
import datetime
import subprocess as sp
from collections import Counter
from statistics import mean




#import os

#clear = lambda: os.system('cls')      # or os.system('clear') for Unix   'cls' for windows and mac

#for i in range(10,0,-1):
 #   clear()
 #   print i
  #  time.sleep(1)

global EXTRA_EOL_CHAR, LINEWIDTH

EXTRA_EOL_CHAR=0
LINEWIDTH=76


  

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
                                    padding=LINEWIDTH-len(w)-1
                                    w=w+" "*padding
                                    rowno+=1
                                    filein.write(w+"\n")
                                    
    print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
    filein.close()
    print("")
    return(rowno)



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

"""
def return_a_row(row,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(linecache.getline(filename,row+1).rstrip())
        except IndexError:
            print("\nindex error")
            return("Index error")
        except ValueError:
       #     print("\nvalue error row+1=",row+1) 
            return("value error")
        except IOError:
            print("\nIO error")
            return("IO error")

        
def return_a_row2(row,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:
            with open(filename,"r") as f:
                f.seek((LINEWIDTH+EXTRA_EOL_CHAR)*row)   # if a windows machine add an extra char for the '\r' EOL char
                return(f.readline().rstrip())
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
        #    print("\nvalue error row+1=",row+1)
            return(0.0)
        except IOError:
            print("\nIO error")
            return("IO error")


def return_a_row3(seekvalue,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:

            with open(filename, "rb") as csvfile:
                datareader = csv.reader(csvfile)
                #yield next(datareader)  # yield the header row
                count = 0
                for row in datareader:
                    if row[0] == seekvalue:
                        yield row
                    count += 1
                 #   elif count:
                # done when having read a consecutive series of rows 
                return


 #           return(any(seekvalue == row[0] for row in csv.reader(filename))
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
        #    print("\nvalue error row+1=",row+1)
            return(0.0)
        except IOError:
            print("\nIO error")
            return("IO error")



     
      


def find_a_payoff(row,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(float(linecache.getline(filename,row+1).split(",")[-1]))
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
         #   print("\nvalue error row+1=",row+1)
            return(0.0)
        except IOError:
            print("\nIO error")
            return("IO error")

def find_a_payoff2(row,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:
            with open(filename,"r") as f:
                f.seek((LINEWIDTH+EXTRA_EOL_CHAR)*row)
                return(float(f.readline().split(',')[-1]))
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
          #  print("\nvalue error row+1=",row+1)
            return(0.0)
        except IOError:
            print("\nIO error")
            return("IO error")


def find_a_payoff3(row,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:

            with open(filename, "rb") as csvfile:
                datareader = csv.reader(csvfile)
                #yield next(datareader)  # yield the header row
                count = 0
                for row in datareader:
                    if row[0] == seekvalue:
                        yield(float(row.split(',')[-1]))
                    count += 1
                 #   elif count:
                # done when having read a consecutive series of rows 
                return

        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
          #  print("\nvalue error row+1=",row+1)
            return(0.0)
        except IOError:
            print("\nIO error")
            return("IO error")




def return_a_row_as_a_list(row,filename):
        try:
            with open(filename,"r") as f:
                return(linecache.getline(filename,row+1).split(","))
        except IndexError:
            print("\nindex error")
            return([])
        except ValueError:
            #print("\nvalue error row+1=",row+1)
            return([])
        except IOError:
            print("\nIO error")
            return([])


def return_a_row_as_a_list2(row,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:
            with open(filename,"r") as f:
                f.seek((LINEWIDTH+EXTRA_EOL_CHAR)*row)   # if a windows machine add an extra char for the '\r' EOL char
                return(f.readline().split(','))
        except IndexError:
            print("\nindex error")
            return([])
        except ValueError:
            print("\nvalue error row+1=",row+1)
            return([])
        except IOError:
            print("\nIO error")
            return([])

"""

def return_a_row_from_envir(val,env):
        # use the preloaded pandas dataframe of the payoff file
        try:
            
            return(env.loc[env["concatno"]==val].values.tolist()[0])
                
        except IndexError:
            print("\nindex error")
            return([])
        except ValueError:
            print("\nvalue error val=",val)
            return([])
        except IOError:
            print("\nIO error")
            return([])




# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the probability_table m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

"""
>>> def echo(value=None):
...     print "Execution starts when 'next()' is called for the first time."
...     try:
...         while True:
...             try:
...                 value = (yield value)
...             except Exception, e:
...                 value = e
...     finally:
...         print "Don't forget to clean up when 'close()' is called."
...
>>> generator = echo(1)
>>> print generator.next()
Execution starts when 'next()' is called for the first time.
1
>>> print generator.next()
None
>>> print generator.send(2)
2
>>> generator.throw(TypeError, "spam")
TypeError('spam',)
>>> generator.close()
Don't forget to clean up when 'close()' is called
"""
   

def calc_mating_probabilities(newpopulation,r,env):
        count=0
        total_payoff=0.00001

        
        
        #size=len(newpopulation)
       # print("new population size=",size)
        for gene in range(0,r.pop_size):
           # print("gene=",gene,"/",pop_size-1)
            fittest=newpopulation.loc[gene,"expressed"]
          #  val=int(fittest,2)   # binary base turned into integer

            total_payoff+=abs(return_a_row_from_envir(fittest,env)[8])
            count+=1


        count=0
        probability_table=[]
        if len(newpopulation)<=1:
            print("\nlen(dna)<=1!")
     #   else:
     #       print("\nlen dna=",len(dna))
       # nor_payoff=total_payoff*1000
     #   print("\ntotal payoff=",total_payoff)
        for gene in range(0,r.pop_size):
            #val=int(dna[count],2)(newpopulation.loc[gene,"expressed"]
            fittest=newpopulation.loc[gene,"expressed"]
#            val=int(fittest,2)   # binary base turned into integer

            p=abs(return_a_row_from_envir(fittest,env)[8])
       
                
        #    if direction=="x":   # maximise
            probability_table.append(int(round((p/total_payoff)*r.actual_scaling))) # scaling usually > pop_size*20
         #       print("#",count+1,":",elem," val=",val,"payoff=",payoff[val]," prob=",probability_table[count])
 
          #  elif direction=="n":   # minimise
      
            #    probability_table.append(int(round((p/total_payoff)*scaling))) # scaling usually > pop_size*20
      #          probability_table.append(int(round(((total_payoff/p)*min_scaling)/scaling)))   # scaling usually pop_size*20.   need an extra min_scaling 100? times here to offset rounding
           #     probability_table.append(int(round(((total_payoff/p)/min_scaling))))   # scaling usually pop_size*20.   need an extra min_scaling 100? times here to offset rounding


      #         print("#",count+1,":",elem," val=",val,"cost=",payoff[val]," prob=",probability_table[count])
 
          #  else:
            #    print("\ndirection error3")

     #       print("#",count+1,":",elem," val=",val,"payoff=",p," prob=",probability_table[count])
            count+=1
       # print("\nlen probability_table",len(probability_table))
       # print(probability_table)
       # input("?")
        return(probability_table)

    
def spin_the_mating_wheel(probability_table,newpopulation,iterations,direction):
        wheel=[]
        mates=[]
        n=0

   # clock_start=time.clock()

        probability_table_len=len(probability_table)
        if probability_table_len<=1:
            print("\nprobability_table length<=1",probability_table_len)

 #       print("\n\n Probability table \n")
  #      print(probability_table)
   #     sum_ptable=sum(probability_table)    
    #    print("sum=",sum_ptable)
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


      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("spin the mating probability_table find the string nos - Clock: duration_clock =", duration_clock)

     #   print("len mates[]",len(mates))
     #   input("?")
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







def pmx(mates,no_of_alleles,pop_size,individual):    # partially matched crossover
    # select two random cutting points that are not the same on the length of the chromosome
    #  a =   984|567|1320
    #  b =   871|230|9546   (locus positions)
#   map string a to string b
  #  moves to:
  #   a' =   984|230|1320
  #   b' =   871|567|9546
  #
#  then mapping string b onto a,
#   a" =     984|230|1657
#   b" =     801|567|9243
#
#  the other 5,6 and 7 in b swap with the a's 2,3 and 0
  # so each string contains ordering information partially determined by each of the parents
   #

    print("pmx crossover")

    pmx_count=0

    c1_choice_list=[]
    c2_choice_list=[]
    old_chromo_list=[]
    new_chromo_list=[]
    
 #   gene_pool_size=no_of_alleles*ploidy*pop_size
  #  number_of_pmx_needed=int(round(gene_pool_size/pmx_rate))
   # for m in range(0,number_of_pmx_needed):
    if True:
        pmx_bit=""
        chromo=""
   
        c1_choice=random.randint(1,2)   # choose a chromo column
        c1_choice_list.append(c1_choice)
        c2_choice=random.randint(0,pop_size-1)   # choose a member of the popultation
        c2_choice_list.append(c2_choice)
        cut1_choice=random.randint(1,no_of_alleles-1)   # choose the first cut position in the chromosome
#  not right at the start and not right at the finish
        cut2_choice=cut1_choice
        while cut2_choice==cut1_choice:
            cut2_choice=random.randint(1,no_of_alleles-1)   # choose the first cut position in the chromosome


# get them in order so cut1 is first, then cut2
        if cut2_choice < cut1_choice:
            temp=cut1_choice    # swap
            cut1_choice=cut2_choice
            cut2_choice=temp

        print("col choice=",c1_choice," mem choice=",c2_choice," cut1=",cut1_choice," cut2=",cut2_choice)
        input("?")



        # chromo1
       # c4_choice=random.randint(-1,1)   # choose a new bit  -1=%,0=0,1=1

       # if c4_choice==-1:
        #    pmx_bit="%"
       # elif c4_choice==0:
        #    pmx_bit="0"
       # elif c4_choice==1:
        #    pmx_bit="1"
       # else:
        #    print("bit mutation error.  c4_choice=",c4_choice)
            
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
         #   splitpoint=random.randint(1,no_of_alleles-1)

            child1=""
            child2=""
            remain11=mate[mate1col][:cut1_point]
            remain12=mate[mate1col][cut2_point:]
            swap1=mate[mate1col][:cut2_point]
            swap11=swap1[cut1_point:]

            remain21=mate[mate2col][:cut1_point]
            remain22=mate[mate2col][cut2_point:]
            swap2=mate[mate2col][:cut2_point]
            swap21=swap2[cut1_point:]

            print("remain11",remain11,"+swap11+",swap11,"+ remain12",remain12)
            print("remain21",remain21,"+swap21+",swap21,"+ remain22",remain22)
            input("?")


            swap2=mate[mate2col][splitpoint-no_of_alleles:]

            child1=remain11+swap2+remain12
            child2=remain21+swap1+remain22

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





###################################################

def genetic_algorithm(r,env):   # class r contains all the settings for the algorithm run, env is the pandas loaded dataframe from the payoff file csv


    starttime=str(datetime.datetime.now())

    r.outfile.write("Genetic algorithm. Started at: "+starttime+"\n\n\n")
    r.results.write("Genetic algorithm. Started at: "+starttime+"\n\n\n")
        
####################################


    #print("\n")
    if r.pconstrain:
            r.outfile.write(str(r.minp)+" <= payoff/cost <= "+str(r.maxp)+"\n")
            r.results.write(str(r.minp)+" <= payoff/cost <= "+str(r.maxp)+"\n")
            if r.advanced_diag!="s": 
                print(str(r.minp)+" <= payoff/cost <= "+str(r.maxp)+"\n")

    if r.aconstrain:
            r.outfile.write(str(r.mina)+" <= a <= "+str(r.maxa)+"\n")
            r.results.write(str(r.mina)+" <= a <= "+str(r.maxa)+"\n")
            if r.advanced_diag!="s": 
                print(str(r.mina)+" <= a <= "+str(r.maxa)+"\n")

    if r.bconstrain:
            r.outfile.write(str(r.minb)+" <= b <= "+str(r.maxb)+"\n")
            r.results.write(str(r.minb)+" <= b <= "+str(r.maxb)+"\n")
            if r.advanced_diag!="s": 
                print(str(r.minb)+" <= b <= "+str(r.maxb)+"\n")
            
    if r.cconstrain:
            r.outfile.write(str(r.minc)+" <= c <= "+str(r.maxc)+"\n")
            r.results.write(str(r.minc)+" <= c <= "+str(r.maxc)+"\n")
            if r.advanced_diag!="s": 
                print(str(r.minc)+" <= c <= "+str(r.maxc)+"\n")
            
    if r.dconstrain:
            r.outfile.write(str(r.mind)+" <= d <= "+str(r.maxd)+"\n")
            r.results.write(str(r.mind)+" <= d <= "+str(r.maxd)+"\n")
            if r.advanced_diag!="s": 
                print(str(r.mind)+" <= d <= "+str(r.maxd)+"\n")

    if r.econstrain:
            r.outfile.write(str(r.mine)+" <= e <= "+str(r.maxe)+"\n")
            r.results.write(str(r.mine)+" <= e <= "+str(r.maxe)+"\n")
            if r.advanced_diag!="s": 
                print(str(r.mine)+" <= e <= "+str(r.maxe)+"\n")

    if r.fconstrain:
            r.outfile.write(str(r.minf)+" <= f <= "+str(r.maxf)+"\n")
            r.results.write(str(r.minf)+" <= f <= "+str(r.maxf)+"\n")
            if r.advanced_diag!="s": 
                print(str(r.minf)+" <= f <= "+str(r.maxf)+"\n")

    if r.gconstrain:
            r.outfile.write(str(r.ming)+" <= g <= "+str(r.maxg)+"\n\n\n")
            r.results.write(str(r.ming)+" <= g <= "+str(r.maxg)+"\n\n\n")
            if r.advanced_diag!="s": 
                print(str(r.ming)+" <= g <= "+str(r.maxg)+"\n\n\n")

#################################################


           
    if r.direction=="x":
            r.outfile.write("MAXIMISING....\n\n")
            r.results.write("MAXIMISING....\n\n")
            if r.advanced_diag!="s": 
                print("MAXIMISING.....")
    elif r.direction=="n":
            r.outfile.write("MINIMISING....\n\n")
            r.results.write("MINIMISING....\n\n")
            if r.advanced_diag!="s": 
                print("MINIMISING.....")
    else:
            print("direction error.")
            r.outfile.close()
            sys.exit()

    r.outfile.write("===================================================\n\n\n")
    r.outfile.flush()
    r.results.write("===================================================\n\n\n")
    r.results.flush()

#######################################################


    allele_len="S"+str(r.no_of_alleles)
    total_generations=0



    individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
    poparray = np.zeros(r.pop_size, dtype=individual) 
    population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")



    max_payoff=-10000000.0
    min_payoff=10000000.0
    max_fittest=""
    min_fittest=""
    best_epoch=1
    best_rowno=0
    best_gen=1
    best_fittest=""
    update_fittest=True
    
    


    for epoch in range(1,r.no_of_epochs+1):
 
        population=build_population(r.pop_size,r.no_of_alleles,population)
      #  print("len population=",len(population))

    #    print("Epoch=",epoch," - generations in epoch=",epoch_length)
      #  print("\nStarting population len=",len(population),"\n")
      #  input("?")
        # print(population)

        besta=0
        bestb=0
        bestc=0
        bestd=0
        beste=0
        bestf=0
        bestg=0




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

        if r.direction=="x":
            p=max_payoff
        elif r.direction=="n":
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
        
        for generation in range(1,r.epoch_length+1):
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
            len_probability_table=0
            
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

 

            
            for gene in range(0,size):  #newpopulation.iterrows():

                plist=[]
                fittest=population.loc[gene,"expressed"]   # binary base turned into integer
          #      val=int(fittest,2)   # binary base turned into integer
                #rowno=val
                population.loc[gene, "fitness"] = p
       
 
                  
                
                try:

                    plist=return_a_row_from_envir(fittest,env)



                 
                
                except ValueError:
                    print("value error finding p in calc fitness")
                    sys.exit()
                except IOError:
                    print("File IO error on ",r.payoff_filename)
                    sys.exit()


              #  print("plist=",plist)
              #  input("?")
                if len(plist)>6:
                    found=True
             #       print("found plist=",plist)
                    
                    a=int(plist[1])
                    b=int(plist[2])
                    c=int(plist[3])
                    d=int(plist[4])
                    e=int(plist[5])
                    f=int(plist[6])
                    g=int(plist[7])
                    p=float(plist[8])

                    if r.pconstrain and (p<r.minp or p>r.maxp): 
                        pass   # p is outside of constraints, move on to next gene
                    else:
                        if r.aconstrain and (a<r.mina or a>r.maxa): 
                            pass   # a is outside of constraints, move on to next gene
                        else:
                                if r.bconstrain and (b<r.minb or b>r.maxb): 
                                    pass   # b is outside of constraints, move on to next gene
                                else:
                                    if r.cconstrain and (c<r.minc or c>r.maxc): 
                                        pass   # a is outside of constraints, move on to next gene
                                    else:
                                        if r.dconstrain and (d<r.mind or d>r.maxd): 
                                            pass   # d is outside of constraints, move on to next gene
                                        else:
                                            if r.econstrain and (e<r.mine or e>r.maxe): 
                                                pass   # e is outside of constraints, move on to next gene
                                            else:
                                                if r.fconstrain and (f<r.minf or f>r.maxf): 
                                                    pass   # f is outside of constraints, move on to next gene
                                                else:
                                                    if r.gconstrain and (g<r.ming or g>r.maxg): 
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

                                                        if r.direction=="x":  # maximising payoff
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
                                                        elif r.direction=="n":
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



##############################################


                                                            
                else:
                    print("error, row found but no payoff value")
                    pass
             


                count+=1

               # if elements_count>0:
               #     averagep=totalp/elements_count

            if best_copy:
                best_copy=False
                update_fittest=True
                bestpopulation=population.copy()
 



            
            allcount=(population['expressed']).count()
            avefitness=((population['fitness']).sum())/allcount
            if r.advanced_diag=="a":   # extra advance detail
                dupcount=allcount-(((population['expressed']).drop_duplicates()).count())
            # count unique parents from parentid1 and parentid2
                nondup_par1=(population['parentid1'].drop_duplicates()).count()   #+((population['parentid1']).drop_duplicates()).count()
                nondup_par2=(population['parentid2'].drop_duplicates()).count()   #+((population['parentid1']).drop_duplicates()).count()


     #       Counter(probability_table).keys() # equals to list(set(words))
      #      Counter(probability_table).values() # counts the elements' frequency




            if r.advanced_diag=="s": # silent mode
                pass
            else:
                tmp=sp.call(r.cls,shell=True)  # clear screen 'use 'clear for unix, cls for windows

                print("Genetic algorithm. By Anthony Paech.")
                print("Started at:",datetime.datetime.now())
                if r.direction=="x":
                    print("\nMAXIMISING.....")
                elif r.direction=="n":
                    print("\nMINIMISING.....")
                if r.pconstrain:
               # outfile.write(str(minp)+" <= payoff/cost <= "+str(maxp)+"\n")
                    print(str(r.minp)," <= payoff/cost <= ",str(r.maxp))

                if r.aconstrain:
               #     outfile.write(str(mina)+" <= a <= "+str(maxa)+"\n")
                    print(str(r.mina)," <= a <= ",str(r.maxa))

                if r.bconstrain:
                    #outfile.write(str(minb)+" <= b <= "+str(maxb)+"\n")
                    print(str(r.minb)," <= b <= ",str(r.maxb))
            
                if r.cconstrain:
                    #outfile.write(str(minc)+" <= c <= "+str(maxc)+"\n")
                    print(str(r.minc)," <= c <= ",str(r.maxc))
            
                if r.dconstrain:
                    #outfile.write(str(mind)+" <= d <= "+str(maxd)+"\n")
                    print(str(r.mind)," <= d <= ",str(r.maxd))

                if r.econstrain:
                    #outfile.write(str(mine)+" <= e <= "+str(maxe)+"\n")
                    print(str(r.mine)," <= e <= ",str(r.maxe))

                if r.fconstrain:
                   # outfile.write(str(minf)+" <= f <= "+str(maxf)+"\n")
                    print(str(r.minf)," <= f <= ",str(r.maxf))

                if r.gconstrain:
                    #outfile.write(str(ming)+" <= g <= "+str(maxg)+"\n\n\n")
                    print(str(r.ming)," <= g <= ",str(r.maxg))
         
                print("======================================")




            if r.direction=="x":    # maximise
                if r.advanced_diag=="s":  # silent mode
                    pass
                else:
                    print("Epoch",epoch,"of",r.no_of_epochs)
                    print("Epoch progress:[%d%%]" % (generation/r.epoch_length*100)," Generation no:",generation,"fittest of this generation:",fittest)

                if r.advanced_diag=="a":
                    print("\nGenepool. Ave Fitness=",avefitness,"#Duplicates expressed=",dupcount,"of",allcount,". Diversity of probability_table weighting=[",len_wheel,"]. probability_table[] size=",len_probability_table)
                    print("\n",nondup_par1,"unique first parents, and",nondup_par2," unique second parents of",allcount,"chomosomes.")
                    print("\nScaling factor=",r.scaling_factor," * genepool size",r.pop_size," = actual scaling:",r.actual_scaling)

                if r.advanced_diag=="s":  # silent mode
#                    print("\rTotal Progress:[%d%%]" % (round(((total_generations+1)/(r.epoch_length*r.no_of_epochs))*100)),end="\r",flush=True)
                    if r.batch_run=="y":
                        print("\rBATCH Total Progress:[%d%%]" % (round(((r.total_total_generations+1)/(r.epoch_length*r.no_of_epochs*2*(r.arange+r.brange+r.crange+r.drange+r.erange+r.frange+r.grange)))*100)),end="\r",flush=True)
                    else:    
                        print("\rTotal Progress:[%d%%]" % (round(((total_generations+1)/(r.epoch_length*r.no_of_epochs))*100)),end="\r",flush=True)
                        
                else:
                    #print("\n",returned_payoff," payoff.")
                    print("\nFittest inputs: a=",a," b=",b," c=",c," d=",d," e=",e," f=",f," g=",g,"of this generation.")
                    print("\n======================================")        
                    print("\nFittest so far:",best_fittest," best rowno [",best_rowno,"] in best generation [",best_gen,"] in best epoch [",best_epoch,"] max payoff",max_payoff)
                    print("\nBest overall so far: a=",besta," b=",bestb," c=",bestc," d=",bestd," e=",beste," f=",bestf," g=",bestg)
                    print("\nTotal Progress:[%d%%]" % (round(((total_generations+1)/(r.epoch_length*r.no_of_epochs))*100)),"\n",flush=True)

                r.outfile.write("Epoch "+str(epoch)+"/"+str(r.no_of_epochs)+" Gen:[%d%%] " % (generation/r.epoch_length*100)+" generation # "+str(generation)+" fittest of this generation "+fittest+"="+str(returned_payoff)+" best="+str(max_fittest)+"\n")
                r.outfile.write("Best epoch "+str(best_epoch)+" best gen "+str(best_gen)+" best row no "+str(best_rowno)+" max pay off "+str(max_payoff)+"\n")
                if r.advanced_diag=="a":
                    r.outfile.write("Diversity of probability_table weighting=["+str(len_wheel)+"]   len probability_table[]="+str(len_probability_table)+"\n")
          #          r.outfile.write(" keys in probability_table "+Counter(probability_table).keys()+"\n") # equals to list(set(words))
           #        r.outfile.write(" frequency of values in probability_table "+Counter(probability_table).values()+"\n") # counts the elements' frequency
                    r.outfile.write("Genepool. Ave fitness= "+str(avefitness)+" #duplicates expressed="+str(dupcount)+" of "+str(allcount)+"\n")
                    r.outfile.write(str(nondup_par1)+" unique first parents, and "+str(nondup_par2)+" unique second parents of "+str(allcount)+" chomosomes.\n")
                    r.outfile.write("Scaling factor="+str(r.scaling_factor)+" * genepool size "+str(r.pop_size)+" = actual scaling: "+str(r.actual_scaling)+"\n")

                r.outfile.write("Current a="+str(a)+" b="+str(b)+" c="+str(c)+" d="+str(d)+" e="+str(e)+" f="+str(f)+" g="+str(g)+"\n")
      #          r.outfile.write("Fittest so far "+str(best_fittest)+" best epoch "+str(best_epoch)+" best generation in best epoch "+str(best_gen)+" best row no "+str(best_rowno)+" max pay off "+str(max_payoff)+"\n")
                r.outfile.write("Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"] max payoff "+str(max_payoff)+"\n")
                r.outfile.write("Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n")
                if update_fittest:
                    update_fittest=False
                    r.results.write("Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"]\nMax payoff "+str(max_payoff)+"\n")
                    r.results.write("Genepool average fitness= "+str(avefitness)+"\n From pop_size:{"+str(r.pop_size)+"}    \n")
                    r.results.write("Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n")

            elif r.direction=="n":  # minimise
                if r.advanced_diag=="s":  # silent mode
                    pass
                else:
                    print("Epoch",epoch,"of",r.no_of_epochs)
                    print("Epoch progress:[%d%%]" % (generation/r.epoch_length*100),"Generation no:",generation,"fittest of this generation:",fittest)

                if r.advanced_diag=="a":
                    print("\nGenepool. Ave fitness=",avefitness," #Duplicates expressed=",dupcount,"of",allcount,". Diversity of probability_table weighting=[",len_wheel,"].  probability_table[] size=",len_probability_table)
                    print("\n",nondup_par1,"unique first parents, and",nondup_par2," unique second parents of",allcount,"chomosomes.")
             #       print("\n Genepool size",pop_size," * min_scaling factor",min_scaling," / Scaling factor=",scaling_factor," = actual scaling:",(pop_size*min_scaling)/scaling_factor)
                    print("\nScaling factor=",r.scaling_factor," * genepool size",r.pop_size," = actual scaling:",r.actual_scaling)

                if r.advanced_diag=="s":  # silent mode
                    if r.batch_run=="y":
                        print("\rBATCH Total Progress:[%d%%]" % (round(((r.total_total_generations+1)/(r.epoch_length*r.no_of_epochs*2*(r.arange+r.brange+r.crange+r.drange+r.erange+r.frange+r.grange)))*100)),end="\r",flush=True)
                    else:    
                        print("\rTotal Progress:[%d%%]" % (round(((total_generations+1)/(r.epoch_length*r.no_of_epochs))*100)),end="\r",flush=True)
                else:
                  #  print("\nRow number",rowno,"=",returned_payoff," cost.")
                    print("\nFittest inputs: a=",a," b=",b," c=",c," d=",d," e=",e," f=",f," g=",g,"of this generation.")
                    print("\n======================================")        
#                   print("\n\nFittest so far:",min_fittest,"best epoch:[",best_epoch,"] best generation in best epoch [",best_gen,"] best rowno [",best_rowno,"] min cost",min_payoff)
                    print("\nFittest so far:",best_fittest," best rowno [",best_rowno,"] in best generation [",best_gen,"] in best epoch [",best_epoch,"] min cost",min_payoff)
                    print("\nBest overall so far: a=",besta," b=",bestb," c=",bestc," d=",bestd," e=",beste," f=",bestf," g=",bestg)
                    print("\nTotal Progress:[%d%%]" % (round(((total_generations+1)/(r.epoch_length*r.no_of_epochs))*100)),"\n",flush=True)

                r.outfile.write("Epoch "+str(epoch)+"/"+str(r.no_of_epochs)+" Gen:[%d%%] " % (generation/r.epoch_length*100)+" generation # "+str(generation)+" fittest of this generation "+fittest+"="+str(returned_payoff)+" best="+str(min_fittest)+"\n")
                r.outfile.write("Best epoch "+str(best_epoch)+" best gen "+str(best_gen)+" best row no "+str(best_rowno)+" min cost "+str(min_payoff)+"\n")
                if r.advanced_diag=="a":
                    r.outfile.write("Diversity of probability_table weighting=["+str(len_wheel)+"]   len probability_table[]="+str(len_probability_table)+"\n")
        #           r.outfile.write(" keys in probability_table "+Counter(probability_table).keys()+"\n") # equals to list(set(words))
        #            r.outfile.write(" frequency of values in probability_table "+Counter(probability_table).values()+"\n") # counts the elements' frequency
                    r.outfile.write("Genepool. Ave fitness="+str(avefitness)+" #duplicates expressed="+str(dupcount)+" of "+str(allcount)+"\n")
                    r.outfile.write(str(nondup_par1)+" unique first parents, and "+str(nondup_par2)+" unique second parents of "+str(allcount)+" chomosomes.\n")
       #             r.outfile.write("Genepool size "+str(pop_size)+" * min_scaling factor "+str(min_scaling)+" / scaling factor "+str(scaling_factor)+" = actual scaling: "+str((pop_size*min_scaling)/scaling_factor)+"\n")
                    r.outfile.write("Scaling factor="+str(r.scaling_factor)+" * genepool size "+str(r.pop_size)+" = actual scaling: "+str(r.actual_scaling)+"\n")

                r.outfile.write("Current a="+str(a)+" b="+str(b)+" c="+str(c)+" d="+str(d)+" e="+str(e)+" f="+str(f)+" g="+str(g)+"\n")
         #       r.outfile.write("Fittest so far "+str(best_fittest)+" best epoch "+str(best_epoch)+" best generation in best epoch "+str(best_gen)+" best row no "+str(best_rowno)+" min cost "+str(min_payoff)+"\n")
                r.outfile.write("Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"] min cost "+str(min_payoff)+"\n")
                r.outfile.write("Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n")
                if update_fittest:
                    update_fittest=False
                    r.results.write("Fittest so far:"+str(best_fittest)+" best rowno ["+str(best_rowno)+"] in best generation ["+str(best_gen)+"] in best epoch ["+str(best_epoch)+"]\nMin cost "+str(min_payoff)+"\n")
                    r.results.write("Genepool average fitness= "+str(avefitness)+"\n From pop_size:{"+str(r.pop_size)+"}    \n")
                    r.results.write("Best a="+str(besta)+" b="+str(bestb)+" c="+str(bestc)+" d="+str(bestd)+" e="+str(beste)+" f="+str(bestf)+" g="+str(bestg)+"\n\n")



            else:
                print("direction error")
                sys.exit()
 
            r.outfile.flush()
            r.results.flush()
            
            probability_table=calc_mating_probabilities(population,r,env)
            # scaling figure is the last.  this multiplies the payoff up so that divsity is not lost on the probability_table when probs are rounded

            len_probability_table=len(probability_table)
            if len_probability_table==0:
                    print("probability_table empty")
                    sys.exit
            #print(probability_table)


            if r.advanced_diag=="a":
                print(" Keys in probability_table",Counter(probability_table).keys()) # equals to list(set(words))
                print(" Frequency of values in probability_table",Counter(probability_table).values()) # counts the elements' frequency
                r.outfile.write(" Keys in probability_table "+str(Counter(probability_table).keys())+"\n") # equals to list(set(words))
                r.outfile.write(" Frequency of values in probability_table "+str(Counter(probability_table).values())+"\n\n") # counts the elements' frequency
 


            mates,len_wheel=spin_the_mating_wheel(probability_table,population,r.pop_size,r.direction)  # wheel_len is the size of the unique gene pool to select from in the probability_table

            if r.c_over=="s":    
                population=crossover(mates,r.no_of_alleles,individual)   # simple crossover
            elif r.c_over=="p":
                population=pmx(mates,r.no_of_alleles,individual,r.pop_size)   # partial matching crossover
                #if advanced_diag=="y" :
                 #   print("\nPartially mixed crossover (PMX) at member index:",whichmember)
             #     print(" member id=",whichmember)
                  #  outfile.write("Partially mixed crossover (PMX) at member index:"+str(whichmember)+"\n\n\n")
           
            else:
                print("Crossover choice error.")
                sys.exit()
            #    print("crossover finished.")
             #   print(population)
              #  input("?")


              

            population, mutation_count, whichchromo,whichmember=mutate(population,r.no_of_alleles,r.ploidy,r.pop_size,r.mutation_rate)   # 1000 means mutation 1 in a 1000 bits processed


            if r.advanced_diag=="a":
                print("Mutation rate: 1/",r.mutation_rate,"alleles. Mutation in individuals:",whichmember)
             #   print(" member id=",whichmember)
                r.outfile.write("Mutation rate: 1/"+str(r.mutation_rate)+" alleles. Mutation in individuals:"+str(whichmember)+"\n\n\n")


            if r.batch_run=="y" and r.advanced_diag!="s":
                print("\nBATCH Total Progress:[%d%%]" % (round(((r.total_total_generations+1)/(r.epoch_length*r.no_of_epochs*2*(r.arange+r.brange+r.crange+r.drange+r.erange+r.frange+r.grange)))*100)),end="\n",flush=True)



        #    print("after mutation pop size=",len(population))
            for p in range(0,r.pop_size):    #fill out the express column for the population:
                population.loc[p,"expressed"]=express(population.loc[p,"chromo1"],population.loc[p,"chromo2"])

        
              #  total_genes=total_genes+gene_pool_size
            mutations=mutations+mutation_count
             #   generation_number+=1

     #   else:
       #     pass   # nothing found
            


            total_generations+=1
            r.total_total_generations+=1

        if r.advanced_diag!="s":    
            print("")


    r.results.write("\nOptimisation GA finished.\n\n")
    r.results.write("==========================================\n\n")
    r.outfile.write("\n\nBest Population:")
    r.outfile.write("\n\n\n"+bestpopulation.to_string())
    r.outfile.write("\n\n")


    return


###########################################






def main():

    class ga(object):
        pass

    ga.ploidy=2  # number of chromosomes per individual.  To increase this you will need to change the dataframes also!
    ga.no_of_alleles=14  # length of each chromosome
    ga.pop_size=256  # population size
    ga.payoff_filename="payoff_7d.csv"
    ga.total_rows=count_file_rows(ga.payoff_filename)
    ga.number_of_cols=8   # 7 input vars and 1 out =9
  #  ga.row_find_method="l"
 
    ga.pconstrain=False
    ga.minp=1000000
    ga.maxp=-1000000
    ga.aconstrain=False
    ga.mina=1000000
    ga.maxa=-1000000
    ga.bconstrain=False
    ga.minb=1000000
    ga.maxb=-1000000
    ga.cconstrain=False
    ga.minc=1000000
    ga.maxc=-1000000
    ga.dconstrain=False
    ga.mind=1000000
    ga.maxd=-1000000
    ga.econstrain=False
    ga.mine=1000000
    ga.maxe=-1000000
    ga.fconstrain=False
    ga.minf=1000000
    ga.maxf=-1000000
    ga.gconstrain=False
    ga.ming=1000000
    ga.maxg=-1000000

    
    ga.found=False
    ga.rowno=0
    ga.best_rowno=0

    ga.besta=0
    ga.bestb=0
    ga.bestc=0
    ga.bestd=0
    ga.beste=0
    ga.bestf=0
    ga.bestg=0
 
    ga.epoch_length=16
    ga.no_of_epochs=2
    ga.total_generations=0
    ga.total_total_generations=0
    
    ga.direction="x"
    ga.scaling_factor=25  # scaling figure is the last. this is related to the size of the genepool. this multiplies the payoff up so that diversity is not lost on the probability_table when probs are rounded
    ga.actual_scaling=ga.scaling_factor*ga.pop_size
  #  min_scaling=100  # when minimising, an extra factor is needed to prevent rounding cutting the diverity of the probability_table
    
    ga.mutation_count=0
    ga.mutations=0
    ga.mutation_rate=400   # mutate 1 bit in every 400.  but the mutation is random 0 or 1 so we need to double the try to mutate rate. but there are also 2 chromos

    ga.advanced_diag=""
    ga.c_over=""
    ga.batch_run=""

    ga.outfile=""
    ga.results=""
    ga.cls=""




################################################
# length in bits of each input variable

    ga.alen=2
    ga.blen=2
    ga.clen=2
    ga.dlen=2
    ga.elen=2
    ga.flen=2
    ga.glen=2

    ga.arange=2**ga.alen    
    ga.brange=2**ga.blen
    ga.crange=2**ga.clen
    ga.drange=2**ga.dlen
    ga.erange=2**ga.elen
    ga.frange=2**ga.flen
    ga.grange=2**ga.glen




#################################################3
    
    if platform.system().lower()[:7]=="windows":
        EXTRA_EOL_CHAR=1
        ga.cls="cls"
    else:
        EXTRA_EOL_CHAR=0
        ga.cls="clear"
 
   
    tmp=sp.call(ga.cls,shell=True)  # clear screen use 'cls' for windows
    #tmp=sp.call('clear',shell=True)  # clear screen use 'clear' for unix


    
    print("Genetic algorithm. By Anthony Paech")
    print("===================================")
    print("Platform:",platform.machine(),"\n",platform.platform())
    #print("\n:",platform.processor(),"\n:",platform.version(),"\n:",platform.uname())
    print("Bit Length=",ga.no_of_alleles,"-> Max CSV datafile rows available is:",2**ga.no_of_alleles)
  #  print("\nTheoretical max no of rows for the CSV datafile file:",payoff_filename,"is:",sys.maxsize)



    

    #individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
    #poparray = np.zeros(ga.pop_size, dtype=individual) 
    #population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")

   # population.index
  #  print(len(population))
   # print("\n\n")
   # print(population)


    print("\n\n") 
    answer=""
    while answer!="y" and answer!="n":
        answer=input("Create payoff env? (y/n)")
    if answer=="y":
        print("Creating payoff/cost environment....file:",ga.payoff_filename)
        clock_start=time.process_time()

          #      total_rows=generate_payoff_environment_7d_file(linewidth,0,2**1,0,2**1,0,2**1,0,2**1,0,2**1,0,2**6,0,2**4,payoff_filename)  
       # 7 input variables 2 bits each = 14 bits, 1 floating point payoff.
        ga.total_rows=generate_payoff_environment_7d_file(0,ga.arange,0,ga.brange,0,ga.crange,0,ga.drange,0,ga.erange,0,ga.frange,0,ga.grange,ga.payoff_filename)

        clock_end=time.process_time()
        duration_clock=clock_end-clock_start
        print("generate payoff/cost environment - Clock: duration_clock =", duration_clock,"seconds.")
        print("Payoff/cost environment file is:",ga.payoff_filename,"and has",ga.total_rows,"rows.\n\n")


########################################################33

#   load filename.csv into pandas for fast analysis

 
    print("loading ",ga.payoff_filename," to pandas")

  #  chunksize = 10 ** 6
  #  for chunk in pd.read_csv(payoff_filename,indexcol=0,dtype=csvenv,delimiter=',', header=None, chunksize=chunksize):
        #process(chunk)
        
   # chunksize = 10 ** 6
    csv_envir=pd.read_csv(ga.payoff_filename,names=["a","b","c","d","e","f","g","payoff"],delimiter=',', header=None)   #, chunksize=chunksize)
 
    # insert an index column which is blank
    print("adding concatno column")
    csv_envir.insert(0,"concatno","",allow_duplicates=False)

    print("calculating index value")
    aforbin="{:0>"+str(ga.alen)+"}"
    bforbin="{:0>"+str(ga.blen)+"}"
    cforbin="{:0>"+str(ga.clen)+"}"
    dforbin="{:0>"+str(ga.dlen)+"}"
    eforbin="{:0>"+str(ga.elen)+"}"
    fforbin="{:0>"+str(ga.flen)+"}"
    gforbin="{:0>"+str(ga.glen)+"}"
                        
    csv_envir["concatno"]=csv_envir["a"].apply(bin).str[2:].apply(aforbin.format)+csv_envir["b"].apply(bin).str[2:].apply(bforbin.format)+csv_envir["c"].apply(bin).str[2:].apply(cforbin.format)+csv_envir["d"].apply(bin).str[2:].apply(dforbin.format)+csv_envir["e"].apply(bin).str[2:].apply(eforbin.format)+csv_envir["f"].apply(bin).str[2:].apply(fforbin.format)+csv_envir["g"].apply(bin).str[2:].apply(gforbin.format)    #str.zfill(3)
    csv_envir.set_index("concatno")
 #   print(csv_envir)
 #   input("?")

  #  print("test")
  #  inb="01010101010101"
  #  payoff=return_a_row_from_envir(inb,csv_envir)[8]
  #  print(payoff)
  #  print("payoff=",payoff)

   # input("?")
    


##############################################################

    print("\n")
    ap=""
    while ap!="y" and ap!="n":
        ap=input("Append to existing outfile.txt? (y/n)")

    print("\n")
    if ap=="y":
        print("appending to existing outfile.txt")
        ga.outfile=open("outfile.txt","a")
    else:
        print("clearing outfile.txt")
        ga.outfile=open("outfile.txt","w")



    ga.results=open("results.txt","w")


    
    print("\n")
    ga.c_over=""
    while ga.c_over!="s" and ga.c_over!="p":
        ga.c_over=input("(s)imple crossover or (p)mx?")

#    print("\n")
#    ga.row_find_method=""
#    while ga.row_find_method!="l" and ga.row_find_method!="s" and ga.row_find_method!="r":
#        ga.row_find_method=input("Use (l)ine cache {fast but memory intensive.  Good for bits<=23} or \n(s)eek {slow but memory frugal. Good for bits >=24}\n or (r)ow seek for no row number?")

    print("\n")
    ga.advanced_diag=""
    while ga.advanced_diag!="p" and ga.advanced_diag!="a" and ga.advanced_diag!="s":
        ga.advanced_diag=input("Do you want to display standard-(p)rogress, (a)dvanced diagnostics or (s)ilent mode?")

  #  print("\n")


    ga.outfile.write("\nCounting rows in "+ga.payoff_filename+"\n")
    ga.total_rows=count_file_rows(ga.payoff_filename)
    ga.outfile.write("Payoff/cost environment file has "+str(ga.total_rows)+" rows.\n\n")

   # best_rowno=0
   # fittest=""

   


    print("\n")
    ga.batch_run=""
    while ga.batch_run!="y" and ga.batch_run!="n":
        ga.batch_run=input("Batch run (y/n)?")

    if ga.batch_run=="n":
        ga.direction=""
        while ga.direction!="x" and ga.direction!="n":
            ga.direction=input("Ma(x)imise or Mi(n)imise?")


        print("\nSet constraints")
        print("===============")
################################# constrain payoff? 
        con=""
        ga.maxp=0
        pconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain payoff/cost? (y/n)")
            if con=="y":
                ga.maxp=int(input("Maximum payoff/cost?"))
                ga.minp=ga.maxp+1
                while ga.minp>ga.maxp:
                    ga.minp=int(input("Minimum payoff/cost?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.minp,"<= payoff/cost <=",ga.maxp)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    ga.pconstrain=True
            else:
                correct="y"    

#################################### constrain a?

        con=""
        ga.maxa=0
        ga.aconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column a? (y/n)")
            if con=="y":
                ga.maxa=int(input("Maximum a?"))
                ga.mina=ga.maxa+1
                while ga.mina>ga.maxa:
                    ga.mina=int(input("Minimum a?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.mina,"<= a <=",ga.maxa)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    ga.aconstrain=True
            else:
                correct="y"    

######################################  constrain b?


        con=""    
        ga.maxb=0
        ga.bconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column b? (y/n)")
            if con=="y":
                ga.maxb=int(input("Maximum b?"))
                ga.minb=ga.maxb+1
                while ga.minb>ga.maxb:
                    ga.minb=int(input("Minimum b?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.minb,"<= b <=",ga.maxb)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                   ga.bconstrain=True
            else:
                correct="y"


###########################################  constrain c?
        
        con=""
        ga.maxc=0
        ga.cconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column c? (y/n)")
            if con=="y":
                ga.maxc=int(input("Maximum c?"))
                ga.minc=ga.maxc+1
                while ga.minc>ga.maxc:
                    ga.minc=int(input("Minimum c?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.minc,"<= c <=",ga.maxc)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    ga.cconstrain=True
            else:
                correct="y"    



##############################################  constrain d?
        con=""
        ga.maxd=0
        ga.dconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column d? (y/n)")
            if con=="y":
                ga.maxd=int(input("Maximum d?"))
                ga.mind=ga.maxd+1
                while ga.mind>ga.maxd:
                    ga.mind=int(input("Minimum d?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.mind,"<= d <=",ga.maxd)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    ga.dconstrain=True
            else:
                correct="y"    


##############################################  constrain e?
        con=""
        ga.maxe=0
        ga.econstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column e? (y/n)")
            if con=="y":
                ga.maxe=int(input("Maximum e?"))
                ga.mine=ga.maxe+1
                while ga.mine>ga.maxe:
                    ga.mine=int(input("Minimum e?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.mine,"<= e <=",ga.maxe)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    ga.econstrain=True
            else:
                correct="y"


##############################################  constrain f?
        con=""
        ga.maxf=0
        ga.fconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column f? (y/n)")
            if con=="y":
                ga.maxf=int(input("Maximum f?"))
                ga.minf=ga.maxf+1
                while ga.minf>ga.maxf:
                    ga.minf=int(input("Minimum f?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.minf,"<= f <=",ga.maxf)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    ga.fconstrain=True
            else:
                correct="y"


##############################################  constrain g?
        con=""
        ga.maxg=0
        ga.gconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column g? (y/n)")
            if con=="y":
                ga.maxg=int(input("Maximum g?"))
                ga.ming=ga.maxg+1
                while ga.ming>ga.maxg:
                    ga.ming=int(input("Minimum g?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ga.ming,"<= g <=",ga.maxg)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    ga.gconstrain=True
            else:
                correct="y"


###############################################################

 

        genetic_algorithm(ga,csv_envir)      #  call the algorithm for each setting


    else:
            
        #  batch run.  test every input at every setting both max and min

        for a in range(0,ga.arange):
            ga.direction="x"
            ga.pconstrain=False
            ga.minp=0
            ga.maxp=0
            ga.aconstrain=True
            ga.mina=a
            ga.maxa=a
            ga.bconstrain=False
            ga.minb=0
            ga.maxb=0
            ga.cconstrain=False
            ga.minc=0
            ga.maxc=0
            ga.dconstrain=False
            ga.mind=0
            ga.maxd=0
            ga.econstrain=False
            ga.mine=0
            ga.maxe=0
            ga.fconstrain=False
            ga.minf=0
            ga.maxf=0
            ga.gconstrain=False
            ga.ming=0
            ga.maxg=0

            # maximise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)
            ga.direction="n"
            #minimise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)


        for b in range(0,ga.brange):
            ga.direction="x"
            ga.pconstrain=False
            ga.minp=0
            ga.maxp=0
            ga.aconstrain=False
            ga.mina=0
            ga.maxa=0
            ga.bconstrain=True
            ga.minb=b
            ga.maxb=b
            ga.cconstrain=False
            ga.minc=0
            ga.maxc=0
            ga.dconstrain=False
            ga.mind=0
            ga.maxd=0
            ga.econstrain=False
            ga.mine=0
            ga.maxe=0
            ga.fconstrain=False
            ga.minf=0
            ga.maxf=0
            ga.gconstrain=False
            ga.ming=0
            ga.maxg=0

            # maximise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)
            ga.direction="n"
            #minimise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)


        for c in range(0,ga.crange):
            ga.direction="x"
            ga.pconstrain=False
            ga.minp=0
            ga.maxp=0
            ga.aconstrain=False
            ga.mina=0
            ga.maxa=0
            ga.bconstrain=False
            ga.minb=0
            ga.maxb=0
            ga.cconstrain=True
            ga.minc=c
            ga.maxc=c
            ga.dconstrain=False
            ga.mind=0
            ga.maxd=0
            ga.econstrain=False
            ga.mine=0
            ga.maxe=0
            ga.fconstrain=False
            ga.minf=0
            ga.maxf=0
            ga.gconstrain=False
            ga.ming=0
            ga.maxg=0

            # maximise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)
            ga.direction="n"
            #minimise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)




        for d in range(0,ga.drange):
            ga.direction="x"
            ga.pconstrain=False
            ga.minp=0
            ga.maxp=0
            ga.aconstrain=False
            ga.mina=0
            ga.maxa=0
            ga.bconstrain=False
            ga.minb=0
            ga.maxb=0
            ga.cconstrain=False
            ga.minc=0
            ga.maxc=0
            ga.dconstrain=True
            ga.mind=d
            ga.maxd=d
            ga.econstrain=False
            ga.mine=0
            ga.maxe=0
            ga.fconstrain=False
            ga.minf=0
            ga.maxf=0
            ga.gconstrain=False
            ga.ming=0
            ga.maxg=0

            # maximise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)
            ga.direction="n"
            #minimise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)


        for e in range(0,ga.erange):
            ga.direction="x"
            ga.pconstrain=False
            ga.minp=0
            ga.maxp=0
            ga.aconstrain=False
            ga.mina=0
            ga.maxa=0
            ga.bconstrain=False
            ga.minb=0
            ga.maxb=0
            ga.cconstrain=False
            ga.minc=0
            ga.maxc=0
            ga.dconstrain=False
            ga.mind=0
            ga.maxd=0
            ga.econstrain=True
            ga.mine=e
            ga.maxe=e
            ga.fconstrain=False
            ga.minf=0
            ga.maxf=0
            ga.gconstrain=False
            ga.ming=0
            ga.maxg=0

            # maximise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)
            ga.direction="n"
            #minimise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)

        for f in range(0,ga.frange):
            ga.direction="x"
            ga.pconstrain=False
            ga.minp=0
            ga.maxp=0
            ga.aconstrain=False
            ga.mina=0
            ga.maxa=0
            ga.bconstrain=False
            ga.minb=0
            ga.maxb=0
            ga.cconstrain=False
            ga.minc=0
            ga.maxc=0
            ga.dconstrain=False
            ga.mind=0
            ga.maxd=0
            ga.econstrain=False
            ga.mine=0
            ga.maxe=0
            ga.fconstrain=True
            ga.minf=f
            ga.maxf=f
            ga.gconstrain=False
            ga.ming=0
            ga.maxg=0

            # maximise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)
            ga.direction="n"
            #minimise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)


        for g in range(0,ga.grange):
            ga.direction="x"
            ga.pconstrain=False
            ga.minp=0
            ga.maxp=0
            ga.aconstrain=False
            ga.mina=0
            ga.maxa=0
            ga.bconstrain=False
            ga.minb=0
            ga.maxb=0
            ga.cconstrain=False
            ga.minc=0
            ga.maxc=0
            ga.dconstrain=False
            ga.mind=0
            ga.maxd=0
            ga.econstrain=False
            ga.mine=0
            ga.maxe=0
            ga.fconstrain=False
            ga.minf=0
            ga.maxf=0
            ga.gconstrain=True
            ga.ming=g
            ga.maxg=g

            # maximise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)
            ga.direction="n"
            #minimise results written to outfile.txt
            genetic_algorithm(ga,csv_envir)














        


    ga.outfile.write("Finished at:"+str(datetime.datetime.now())+"\n\n")
    ga.results.write("Finished at:"+str(datetime.datetime.now())+"\n\n")


    ga.outfile.close()
    ga.results.close()
    print("\nCheck outfile.txt for log, check results.txt for optimisation. \nFinished at:",str(datetime.datetime.now()),"\n\n")
#    print("Final population\n")
 #   print(population.to_string())
    

 



if __name__ == '__main__':
    main()


