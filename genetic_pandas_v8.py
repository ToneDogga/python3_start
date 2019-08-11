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
# create a biased roulette wheel where the % breakdown score is the probability of the wheel landing on that string
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
import math
import linecache
import sys
import platform



def generate_payoff_environment_7d_file(linewidth,astart_val,asize_of_env,bstart_val,bsize_of_env,cstart_val,csize_of_env,dstart_val,dsize_of_env,estart_val,esize_of_env,fstart_val,fsize_of_env,gstart_val,gsize_of_env,filename):   
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
                                    w=str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(f)+","+str(g)+","+str(payoff)
                                    padding=linewidth-len(w)-1
                                    w=w+" "*padding
                                    rowno+=1
                                    filein.write(w+"\n")
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
                f.seek((linewidth+extra_EOL_char)*row)   # if a windows machine add an extra char for the '\r' EOL char
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
                f.seek((linewidth+extra_EOL_char)*row)
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


"""
def find_a_row_and_column(row,col,filename):   # assumes the payoff is the last field in a CSV delimited by ","
       # print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(float(linecache.getline(filename,row+1).split(",")[col]))
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
          #  print("\nvalue error row+1=",row+1,"col=",col)
            return(0.0)
        except IOError:
            print("\nIO error")
            return("IO error")
        

def find_a_row_and_column2(row,col,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:
            with open(filename,"r") as f:
                f.seek((linewidth+extra_EOL_char)*row)
                return(float(f.readline().split(',')[col]))
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
         #   print("\nvalue error row+1=",row+1)
            return(0.0)
        except IOError:
            print("\nIO error")
            return("IO error")
"""

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
                f.seek((linewidth+extra_EOL_char)*row)   # if a windows machine add an extra char for the '\r' EOL char
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
def calc_fitness(newpopulation,gen,direction,max_fittest,max_payoff,min_fittest,min_payoff,best_rowno,payoff_filename):   # the whole population 
    #  loop through the whole population dna and update the fitness based on the payoff file
    p=0.0
    plist=[]
       # element_list=[]
    elements_count=0
    averagep=0
    totalp=0
    count=0
    returned_payoff=0
    best=""
    #best_rowno=0
    fittest=""
    row_find_method="l"
    found=False   # found flag is a True if dna is found and a bestrow returned
    

    #newpopulation = population.copy(deep=True)

    size=len(newpopulation)
   # print("len genepool=",size)
    for gene in range(0,size-1):  #newpopulation.iterrows():
        plist=[]
        val=int(newpopulation.loc[gene,"expressed"],2)   # binary base turned into integer
      #  print("val=",val)
        if val <= size:
                try:
                    if row_find_method=="l":  # linecache
                        plist=return_a_row_as_a_list(val,payoff_filename)
                        #found=True
                        
                   # elif row_find_method=="s":    
 #                       p=self.find_a_payoff2(val,payoff_filename)
                     #   plist=return_a_row_as_a_list2(val,payoff_filename)
                     #   found=True

                    else:
                        print("row find method error.")
                        sys.exit()
                        
                except ValueError:
                    print("value error finding p in calc fitness")
                    sys.exit()
                except IOError:
                    print("File IO error on ",payoff_filename)
                    sys.exit()


              #  print("plist=",plist)
              #  input("?")
                if len(plist)>7:
                    found=True
             #       print("found plist=",plist)
                    row_number=int(plist[0])
                    a=int(plist[1])
                    b=int(plist[2])
                    c=int(plist[3])
                    d=int(plist[4])
                    e=int(plist[5])
                    f=int(plist[6])
                    g=int(plist[7])
                    p=float(plist[8].rstrip())
                    totalp+=p

                    #if p>returned_payoff:
                    returned_payoff=p
                    fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                    rowno=val
                   # best_rowno=val

                    elements_count+=1

                    #newpopulation.loc[row_number,["fitness"]]=p
                   # newpopulation.loc[int(newpopulation.loc[gene,"expressed"],2), "fitness"] = p
                    newpopulation.loc[val, "fitness"] = p

                    if direction=="x":  # maximising payoff
                        if p>max_payoff:
                                    
                                  #  fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                    max_payoff=p
                                    max_fittest=fittest
                                    best_rowno=rowno
                                    best_gen=gen
                        else:
                             pass
                    elif direction=="n":
                        if p<min_payoff:
                                    
                                   # fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                    min_payoff=p
                                    min_fittest=fittest
                                    best_rowno=rowno
                                    best_gen=gen
                        else:
                             pass
                    else:
                        print("direction error")
                        sys.exit()
                else:
                    print("error, row found but no payoff value")
                    pass     
        else:
            pass
                
              #  print("\nval ",val," is greater than total environment (",total_rows,")")
          #  print("#",count+1,":",elem," val=",val,"payoff=",p," fitness=",fitness[count])
         #   print("bestrow=",bestrow,"best=",best," max",max_payoff)
        count+=1

    if elements_count>0:
        averagep=totalp/elements_count
          #  print("\n val=",val," totalp=",totalp," elements_count=",elements_count," average payoff=",averagep)
        


       # print("\nbestrow=",bestrow,"best=",best," max",max_payoff," min",min_payoff)
       # input("?")
   # print("calc fitness count=",count)   
    if direction=="x":
        return(newpopulation,found,rowno,fittest,returned_payoff,best_gen,best_rowno,max_fittest,max_payoff)
    elif direction=="n":
        return(newpopulation,found,rowno,fittest,returned_payoff,best_gen,best_rowno,min_fittest,min_payoff)
    else:
        print("direction error..")
        return(newpopulation,False,0,"",0,0,0,"",0)
"""



# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette wheel where the % breakdown score is the probability of the wheel landing on that string
# spin the wheel m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

   

def calc_mating_probabilities(newpopulation,pop_size,direction,scaling,row_find_method,payoff_filename):
        count=0
        total_payoff=0.00001
        #size=len(newpopulation)
       # print("new population size=",size)
        for gene in range(0,pop_size-1):
           # print("gene=",gene,"/",pop_size-1)
            fittest=newpopulation.loc[gene,"expressed"]
            val=int(fittest,2)   # binary base turned into integer

          #  total_payoff+=payoff[val]
            if row_find_method=="l":
                total_payoff+=find_a_payoff(val,payoff_filename)
            elif row_find_method=="s": 
                total_payoff+=find_a_payoff2(val,payoff_filename)
            else:
                print("row find method error.")
                sys.exit()
        #print(val,payoff[val])
            count+=1

#    print("payoff=",payoff)
#    input("?")

        count=0
        wheel=[]
        if len(newpopulation)<=1:
            print("\nlen(dna)<=1!")
     #   else:
     #       print("\nlen dna=",len(dna))
       # nor_payoff=total_payoff*1000
     #   print("\ntotal payoff=",total_payoff)
        for gene in range(0,pop_size-1):
            #val=int(dna[count],2)(newpopulation.loc[gene,"expressed"]
            fittest=newpopulation.loc[gene,"expressed"]
            val=int(fittest,2)   # binary base turned into integer

            if row_find_method=="l":
                p=find_a_payoff(val,payoff_filename)
            elif row_find_method=="s":   
                p=find_a_payoff2(val,payoff_filename)
            else:
                print("row find method error.")
                sys.exit()
                
            if direction=="x":   # maximise
                wheel.append(int(round(p/total_payoff*scaling)))
         #       print("#",count+1,":",elem," val=",val,"payoff=",payoff[val]," prob=",wheel[count])
 
            elif direction=="n":   # minimise
                wheel.append(int(round(-p/total_payoff*scaling)))
      #         print("#",count+1,":",elem," val=",val,"cost=",payoff[val]," prob=",wheel[count])
 
            else:
                print("\ndirection error3")

     #       print("#",count+1,":",elem," val=",val,"payoff=",p," prob=",wheel[count])
            count+=1
       # print("\nlen wheel",len(wheel))
       # input("?")
        return(wheel)

    
def spin_the_mating_wheel(wheel,newpopulation,iterations):
        sel=[]
        mates=[]
        n=0

   # clock_start=time.clock()

        wheel_len=len(wheel)
        if wheel_len<=1:
            print("\nwheel length<=1",wheel_len)
            
        while n<=wheel_len-1: 
            sel=sel+([n+1] * abs(wheel[n]))
            n=n+1

        len_sel=len(sel)
   #     print("\nlen(sel)=",len_sel,"sel=",sel,"\n\nwheel=",wheel)
   #     input("?")
       
        if len_sel<=20:
            print("\n Warning! increase your total_payoff scaling. len sel <=20",len_sel," wheel len=",wheel_len)
        for i in range(0,iterations):
            go_back=True
            while go_back:
                # pick a random string for mating
                first_string_no=random.randint(1,wheel_len)
                # choose its mate from the wheel
                second_string_no=first_string_no
                while second_string_no==first_string_no:
                    second_string_no=sel[random.randint(0,len_sel-1)]
                   # print("mate ",first_string_no,dna[first_string_no-1]," with ",second_string_no,dna[second_string_no-1])

                    # if the string to mate with is the same, try again
                go_back=False
                if newpopulation.loc[first_string_no-1,"chromo1"]==newpopulation.loc[second_string_no-1,"chromo2"]:
                    go_back=True

            mates=mates+[[0.0,first_string_no-1,0,second_string_no-1,0,newpopulation.loc[first_string_no-1,"chromo1"],newpopulation.loc[second_string_no-1,"chromo2"],"","",""]]      # mates is a list of tuples to be mated               


      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("spin the mating wheel find the string nos - Clock: duration_clock =", duration_clock)

     #   print("len mates[]",len(mates))
     #   input("?")
        return(mates,len_sel)   # if len_sel gets small, there is a lack of genetic diversity





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

    gene_pool_size=no_of_alleles*ploidy*pop_size
    number_of_mutations_needed=int(round(gene_pool_size/mutation_rate))
    for m in range(0,number_of_mutations_needed):
        mutated_bit=""
        chromo=""
   
        c1_choice=random.randint(1,2)   # choose a chromo column
        c2_choice=random.randint(0,pop_size-1)   # choose a member of the popultation
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
            chromo=newpopulation.loc[c2_choice,"chromo1"]
            
        else:
            # chromo2
            chromo=newpopulation.loc[c2_choice,"chromo2"]


    #    print("chromo before mutate=",chromo," at",c1_choice,c2_choice)

        newchromo=chromo[:c3_choice]+mutated_bit+chromo[c3_choice+1:]
   #     print("new chromo after mutate=",newchromo," at",c3_choice,c4_choice)


        if c1_choice==1:
            # chromo1
            newpopulation.loc[c2_choice,"chromo1"]=newchromo
            
        else:
            # chromo2
            newpopulation.loc[c2_choice,"chromo2"]=newchromo


       # print(newpopulation.to_string())
        #input("?")

        mutation_count+=1  
        
    return(newpopulation,mutation_count)






##########################################

def main():
    ploidy=2  # number of chromosomes per individual
    no_of_alleles=21  # length of each chromosome
    pop_size=512   # population size
    epoch_count=0
    no_of_epochs=3
    generation_count=0

    payoff_filename="payoff_7d.csv"
    total_rows=count_file_rows(payoff_filename)
    number_of_cols=9   # rowno, 6 input vars and 1 out =8
    linewidth=76   # 66 bytes
    extra_EOL_char=0
    row_find_method="l"

    pconstrain=False
    minp=1000000
    maxp=-1000000
    aconstrain=False
    mina=1000000
    maxa=-1000000
    bconstrain=False
    minb=1000000
    maxb=-1000000
    cconstrain=False
    minc=1000000
    maxc=-1000000
    dconstrain=False
    mind=1000000
    maxd=-1000000
    econstrain=False
    mine=1000000
    maxe=-1000000
    fconstrain=False
    minf=1000000
    maxf=-1000000
    gconstrain=False
    ming=1000000
    maxg=-1000000

    
    found=False
    rowno=0
    best_rowno=0

    besta=0
    bestb=0
    bestc=0
    bestd=0
    beste=0
    bestf=0
    bestg=0
 
    epoch_length=40
    
    direction="x"
    scaling_factor=10000  # scaling figure is the last.  this multiplies the payoff up so that diversity is not lost on the wheel when probs are rounded
    mutation_count=0
    mutations=0
    mutation_rate=500   # mutate 1 bit in every 1000.  but the mutation is random 0 or 1 so we need to double the try to mutate rate

    
    allele_len="S"+str(no_of_alleles)

#########################################

    print("\n\nGenetic algorithm. By Anthony Paech")
    print("===================================")
    print("Platform:",platform.machine(),"\n:",platform.platform())
    #print("\n:",platform.processor(),"\n:",platform.version(),"\n:",platform.uname())
    print("Bit Length=",no_of_alleles,"-> Max CSV datafile rows available is:",2**no_of_alleles)
  #  print("\nTheoretical max no of rows for the CSV datafile file:",payoff_filename,"is:",sys.maxsize)




    

    individual = np.dtype([('fitness','f16'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
    poparray = np.zeros(pop_size, dtype=individual) 
    population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")

   # population.index
  #  print(len(population))
   # print("\n\n")
   # print(population)


 
    answer=""
    while answer!="y" and answer!="n":
        answer=input("Create payoff env? (y/n)")
    if answer=="y":
        print("Creating payoff/cost environment....file:",payoff_filename)
        clock_start=time.process_time()

          #      total_rows=generate_payoff_environment_7d_file(linewidth,0,2**1,0,2**1,0,2**1,0,2**1,0,2**1,0,2**6,0,2**4,payoff_filename)  
       # 7 input variables 2 bits each = 14 bits, 1 floating point payoff.
        total_rows=generate_payoff_environment_7d_file(linewidth,0,2**3,0,2**3,0,2**3,0,2**3,0,2**3,0,2**3,0,2**3,payoff_filename)

        clock_end=time.process_time()
        duration_clock=clock_end-clock_start
        print("generate payoff/cost environment - Clock: duration_clock =", duration_clock,"seconds.")
        print("Payoff/cost environment file is:",payoff_filename,"and has",total_rows,"rows.")

    
    outfile=open("outfile.txt","a")




    if platform.system().lower()[:7]=="windows":
        extra_EOL_char=1
    else:
        extra_EOL_char=0

    print("counting rows in ",payoff_filename)
    total_rows=count_file_rows(payoff_filename)
    print("\nPayoff/cost environment file is:",payoff_filename,"and has",total_rows,"rows.")

   # best_rowno=0
   # fittest=""
    max_payoff=-10000000.0
    min_payoff=10000000.0
    max_fittest=""
    min_fittest=""
    best_epoch=1
    best_rowno=0
    bestpopulation=population.copy()
     

    batch=""
    while batch!="y" and batch!="n":
        batch=input("Batch run? (y/n)")

    if batch=="n":
    
        print("\n\nPayoff/Cost Environment")


        direction=""
        while direction!="x" and direction!="n":
            direction=input("Ma(x)imise or Mi(n)imise?")


        print("\nSet constraints")
        print("===============")
################################# constrain payoff? 
        con=""
        maxp=0
        pconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain payoff/cost? (y/n)")
            if con=="y":
                maxp=int(input("Maximum payoff/cost?"))
                minp=maxp+1
                while minp>maxp:
                    minp=int(input("Minimum payoff/cost?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(minp,"<= payoff/cost <=",maxp)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    pconstrain=True
            else:
                correct="y"    

#################################### constrain a?

        con=""
        maxa=0
        aconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column a? (y/n)")
            if con=="y":
                maxa=int(input("Maximum a?"))
                mina=maxa+1
                while mina>maxa:
                    mina=int(input("Minimum a?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(mina,"<= a <=",maxa)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    aconstrain=True
            else:
                correct="y"    

######################################  constrain b?


        con=""    
        maxb=0
        bconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column b? (y/n)")
            if con=="y":
                maxb=int(input("Maximum b?"))
                minb=maxb+1
                while minb>maxb:
                    minb=int(input("Minimum b?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(minb,"<= b <=",maxb)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    bconstrain=True
            else:
                correct="y"


###########################################  constrain c?
        
        con=""
        maxc=0
        cconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column c? (y/n)")
            if con=="y":
                maxc=int(input("Maximum c?"))
                minc=maxc+1
                while minc>maxc:
                    minc=int(input("Minimum c?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(minc,"<= c <=",maxc)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    cconstrain=True
            else:
                correct="y"    



##############################################  constrain d?
        con=""
        maxd=0
        dconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column d? (y/n)")
            if con=="y":
                maxd=int(input("Maximum d?"))
                mind=maxd+1
                while mind>maxd:
                    mind=int(input("Minimum d?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(mind,"<= d <=",maxd)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    dconstrain=True
            else:
                correct="y"    


##############################################  constrain e?
        con=""
        maxe=0
        econstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column e? (y/n)")
            if con=="y":
                maxe=int(input("Maximum e?"))
                mine=maxe+1
                while mine>maxe:
                    mine=int(input("Minimum e?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(mine,"<= e <=",maxe)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    econstrain=True
            else:
                correct="y"


##############################################  constrain f?
        con=""
        maxf=0
        fconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column f? (y/n)")
            if con=="y":
                maxf=int(input("Maximum f?"))
                minf=maxf+1
                while minf>maxf:
                    minf=int(input("Minimum f?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(minf,"<= f <=",maxf)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    fconstrain=True
            else:
                correct="y"


##############################################  constrain g?
        con=""
        maxg=0
        gconstrain=False
        correct="n"
        while correct=="n":
            while con!="y" and con!="n":
                con=input("Constrain column g? (y/n)")
            if con=="y":
                maxg=int(input("Maximum g?"))
                ming=maxg+1
                while ming>maxg:
                    ming=int(input("Minimum g?"))
                correct=""
                while correct!="y" and correct!="n":
                    print(ming,"<= g <=",maxg)
                    correct=input("Correct? (y/n)")
                if correct=="y":
                    gconstrain=True
            else:
                correct="y"


###############################################################


        print("\n")
        if pconstrain:
            outfile.write(str(minp)+" <= payoff/cost <= "+str(maxp)+"\n")
            print(str(minp)+" <= payoff/cost <= "+str(maxp)+"\n")

        if aconstrain:
            outfile.write(str(mina)+" <= a <= "+str(maxa)+"\n")
            print(str(mina)+" <= a <= "+str(maxa)+"\n")

        if bconstrain:
            outfile.write(str(minb)+" <= b <= "+str(maxb)+"\n")
            print(str(minb)+" <= b <= "+str(maxb)+"\n")
            
        if cconstrain:
            outfile.write(str(minc)+" <= c <= "+str(maxc)+"\n")
            print(str(minc)+" <= c <= "+str(maxc)+"\n")
            
        if dconstrain:
            outfile.write(str(mind)+" <= d <= "+str(maxd)+"\n")
            print(str(mind)+" <= d <= "+str(maxd)+"\n")

        if econstrain:
            outfile.write(str(mine)+" <= e <= "+str(maxe)+"\n")
            print(str(mine)+" <= e <= "+str(maxe)+"\n")

        if fconstrain:
            outfile.write(str(minf)+" <= f <= "+str(maxf)+"\n")
            print(str(minf)+" <= f <= "+str(maxf)+"\n")

        if gconstrain:
            outfile.write(str(ming)+" <= g <= "+str(maxg)+"\n\n\n")
            print(str(ming)+" <= g <= "+str(maxg)+"\n\n\n")

#################################################
            
        if direction=="x":
            outfile.write("MAXIMISING....\n\n")
            print("MAXIMISING.....")
        elif direction=="n":
            outfile.write("MINIMISING....\n\n")
            print("MINIMISING.....")
        else:
            print("direction error.")
            outfile.close()
            sys.exit()

        outfile.write("===================================================\n\n\n")
        outfile.flush()

#####################################################



    for epoch in range(1,no_of_epochs+1):
 
        population=build_population(pop_size,no_of_alleles,population)
      #  print("len population=",len(population))

    #    print("Epoch=",epoch," - generations in epoch=",epoch_length)
      #  print("\nStarting population len=",len(population),"\n")
      #  input("?")
        # print(population)

        fittest=""
       # best_rowno=0
        mutations=0
        totalp=0.0
        averagep=0.0
        elements_count=0
        best_gen=1
        len_sel=0
        
        p=0.0
        plist=[]

      #  best_flag=True
        best=""
        fittest=""
            # element_list=[]
        
        for generation in range(1,epoch_length+1):
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
            for gene in range(0,size-1):  #newpopulation.iterrows():
                plist=[]
                fittest=population.loc[gene,"expressed"]   # binary base turned into integer
                val=int(fittest,2)   # binary base turned into integer
 
                  #  print("val=",val)
                if val <= total_rows:   #total_rows:
                        try:
                            if row_find_method=="l":  # linecache
                                plist=return_a_row_as_a_list(val,payoff_filename)
                                #found=True
                        
                           # elif row_find_method=="s":    
 #                             p=self.find_a_payoff2(val,payoff_filename)
                             #   plist=return_a_row_as_a_list2(val,payoff_filename)
                             #   found=True

                            else:
                                print("row find method error.")
                                sys.exit()
                        
                        except ValueError:
                            print("value error finding p in calc fitness")
                            sys.exit()
                        except IOError:
                            print("File IO error on ",payoff_filename)
                            sys.exit()


                      #  print("plist=",plist)
                      #  input("?")
                        if len(plist)>7:
                            found=True
                     #       print("found plist=",plist)
                            row_number=int(plist[0])
                            a=int(plist[1])
                            b=int(plist[2])
                            c=int(plist[3])
                            d=int(plist[4])
                            e=int(plist[5])
                            f=int(plist[6])
                            g=int(plist[7])
                            p=float(plist[8].rstrip())
                            totalp+=p

                            #if p>returned_payoff:
                            returned_payoff=p
                           # try:
                            #    fittest=population.loc[val,"expressed"]   # if not constrained at all
                           # except IndexError:
                            #    print("index error. val=",val)

                                
                            rowno=val
                           # best_rowno=val

                            elements_count+=1

                            #newpopulation.loc[row_number,["fitness"]]=p
                           # newpopulation.loc[int(newpopulation.loc[gene,"expressed"],2), "fitness"] = p
                    #        population.loc[val, "fitness"] = p
                            population.loc[gene, "fitness"] = p


                            if direction=="x":  # maximising payoff
                                if p>max_payoff:
                                    
                                          #  fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                    max_payoff=p
                                    max_fittest=fittest
                                    best_rowno=gene   #rowno
                                    best_gen=generation
                                    best_epoch=epoch
                                   # bestflag=True
                                    bestpopulation=population.copy()
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
                            elif direction=="n":
                                if p<min_payoff:
                                    
                                   # fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                    min_payoff=p
                                    min_fittest=fittest
                                    best_rowno=gene  #rowno
                                    best_gen=generation
                                    best_epoch=epoch
                                 #   best_flag=True
                                    bestpopulation=population.copy()
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
                            print("error, row found but no payoff value")
                            pass
                else:
                    pass
                    print("\nval ",val," is greater than total environment (",total_rows,")")
                  #  print("#",count+1,":",elem," val=",val,"payoff=",p," fitness=",fitness[count])
                 #   print("bestrow=",bestrow,"best=",best," max",max_payoff)
                 


                count+=1

                if elements_count>0:
                    averagep=totalp/elements_count

            if direction=="x":
                print("\rEpoch",epoch,"/",no_of_epochs,"Gen:[%d%%]" % (generation/epoch_length*100),"fittest",fittest,"row",rowno,"=",returned_payoff,"best",max_fittest,"be",best_epoch,"bgen",best_gen,"brno",best_rowno,"maxpoff",max_payoff,"        ",end='\r',flush=True)
            else:
                print("\rEpoch",epoch,"/",no_of_epochs,"Gen:[%d%%]" % (generation/epoch_length*100),"fittest",fittest,"row",rowno,"=",returned_payoff,"best",min_fittest,"be",best_epoch,"bgen",best_gen,"brno",best_rowno,"mincost",min_payoff,"        ",end='\r',flush=True)
 
 

            wheel=calc_mating_probabilities(population,pop_size,direction,scaling_factor,row_find_method,payoff_filename)
            # scaling figure is the last.  this multiplies the payoff up so that divsity is not lost on the wheel when probs are rounded
            if len(wheel)==0:
                    print("wheel empty")
                    sys.exit
            #print(wheel)


            mates,len_sel=spin_the_mating_wheel(wheel,population,pop_size)  # sel_len is the size of the unique gene pool to select from in the wheel

                
            population=crossover(mates,no_of_alleles,individual)
            #    print("crossover finished.")
             #   print(population)
              #  input("?")

            population, mutation_count=mutate(population,no_of_alleles,ploidy,pop_size,mutation_rate)   # 1000 means mutation 1 in a 1000 bits processed

        #    print("after mutation pop size=",len(population))
            for p in range(0,pop_size):    #fill out the express column for the population:
                population.loc[p,"expressed"]=express(population.loc[p,"chromo1"],population.loc[p,"chromo2"])

        
              #  total_genes=total_genes+gene_pool_size
            mutations=mutations+mutation_count
             #   generation_number+=1

        else:
            pass   # nothing found
            

         
        print("")



        
    outfile.close()
  #  print("\n\nMutations=",mutations," Best population=")
    #population,found,fittest,rowno,returned_payoff,best_gen,best_rowno,max_fittest,max_payoff=calc_fitness(population,generation,direction,max_fittest,max_payoff,min_fittest,min_payoff,best_rowno,payoff_filename)  # use linecache method for fin row no in payoff file.csv
    #print("Epoch no:",epoch,"/",no_of_epochs,"  Generation progress: [%d%%]" % (generation/epoch_length*100)," fittest=",fittest," rowno=",rowno," returned payoff=",returned_payoff," best=",max_fittest,"bestgen=",best_gen," bestrowno=",best_rowno," max_payoff=",max_payoff,"    ",flush=True)
  #  print("\nBest epoch",best_epoch," Best gen no:",best_gen,"\n")
    print("best a=",besta," b=",bestb," c=",bestc," d=",bestd," e=",beste," f=",bestf," g=",bestg)
 
   # print(bestpopulation.to_string())
   # print("\n")
#    print("Final population\n")
 #   print(population.to_string())
    

 



if __name__ == '__main__':
    main()





"""
            if found:   # something found
             #   print("\rfound=",found," fittest=",fittest," returned payoff=",returned_payoff," max_payoff=",max_payoff,"    ",end='\r',flush=True)
                if int(fittest,2)<=total_rows:
                      #  print("report")
                        axis=["0"] * number_of_cols

                     #   print("returned payoff:",returned_payoff)
                        if direction=="x":
                            if returned_payoff>max_payoff:
                              #  best=fittest
                                col=0
                                while col<=number_of_cols-1:
                                    if row_find_method=="l":
                                       axis[col]=return_a_row(int(fittest,2),payoff_filename).split(",")[col]
                                    elif row_find_method=="s":   
                                       axis[col]=return_a_row2(int(fittest,2),payoff_filename).split(",")[col]
                                    else:
                                       print("row find method error.")
                                       sys.exit()

                                    col+=1

                                gen=generation_number
                                max_payoff=returned_payoff
                                print("\rbest fittest=",fittest," value=",int(fittest,2)," a=",axis[1]," b=",axis[2]," c=",axis[3]," d=",axis[4]," e=",axis[5]," f=",axis[6]," g=",axis[7]," generation no:",gen,"max_payoff=",max_payoff, end='\r',flush=True)
                                if elements>0:
                                    print("number of elements found satisfying constraint=",elements," average=",averagep)
                                outfile.write("best fittest= "+fittest+" value= "+str(int(fittest,2))+" row number= "+" a="+str(axis[1])+" b="+str(axis[2])+" c="+str(axis[3])+" d="+str(axis[4])+" e="+str(axis[5])+" f="+str(axis[6])+" g="+str(axis[7])+" generation no: "+str(gen)+" max_payoff= "+str(max_payoff)+"\n")
                                if elements>0:
                                    outfile.write("number of elements found satisfying constraint="+str(elements)+" average="+str(averagep)+"\n")
                                
                        elif direction=="n":
                            if returned_payoff<min_cost:
                              #  best=fittest
                                col=0
                                while col<=number_of_cols-1:
                                    if row_find_method=="l":
                                        axis[col]=return_a_row(int(fittest,2),payoff_filename).split(",")[col]
                                    elif row_find_method=="s":
                                        axis[col]=return_a_row2(int(fittest,2),payoff_filename).split(",")[col]
                                    else:
                                        print("row find method error.")
                                        sys.exit()

                                    col+=1

                                gen=generation_number
                                min_cost=returned_payoff
  

                                print("\rbest fittest=",fittest," value=",int(fittest,2)," a=",axis[1]," b=",axis[2]," c=",axis[3]," d=",axis[4]," e=",axis[5]," f=",axis[6]," g=",axis[7],"generation no:",gen,"min_cost=",min_cost,end='\r',flush=True)
                                if elements>0:
                                    print("number of elements found satisfying constraint=",elements," average=",averagep)
                                outfile.write("best fittest="+fittest+" value="+str(int(fittest,2))+" a="+str(axis[1])+" b="+str(axis[2])+" c="+str(axis[3])+" d="+str(axis[4])+" e="+str(axis[5])+" f="+str(axis[6])+" g="+str(axis[7])+" generation no: "+str(gen)+" min_cost= "+str(min_cost)+"\n")
                                if elements>0:
                                    outfile.write("number of elements found satisfying constraint="+str(elements)+" average="+str(averagep)+" \n")



                        else:
                            print("direction error1 direction=",direction)

               #     print("report on fittest")
                 #   report(fittest,returned_payoff,max_fittest,max_payoff,max_payoff,number_of_cols,row_find_method,generation,direction,payoff_filename)
                else:
                    print("fittest",fittest," is beyond the environment max (",total_rows,").")


                    
"""





