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







def calc_fitness(newpopulation,direction,payoff_filename):   # the whole population 
    #  loop through the whole population dna and update the fitness based on the payoff file
    max_payoff=-10000000.0
    min_payoff=10000000.0
    p=0.0
    plist=[]
       # element_list=[]
    elements_count=0
    averagep=0
    totalp=0
    count=0
    returned_payoff=0
    best=""
    fittest=""
    row_find_method="l"
    found=False   # found flag is a True if dna is found and a bestrow returned
    

    #newpopulation = population.copy(deep=True)

    size=len(newpopulation)
    print("len genepool=",size)
    for gene in range(0,size-1):  #newpopulation.iterrows():
        plist=[]
        val=int(newpopulation.loc[gene,"expressed"],2)   # binary base turned into integer
      #  print("val=",val)
        if val <= size:
                try:
                    if row_find_method=="l":  # linecache
                        plist=return_a_row_as_a_list(val,payoff_filename)
                        
                   # elif row_find_method=="s":    
 #                       p=self.find_a_payoff2(val,payoff_filename)
                     #   plist=return_a_row_as_a_list2(val,payoff_filename)

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

                    returned_payoff=p                                
                    elements_count+=1

                    #newpopulation.loc[row_number,["fitness"]]=p
                   # newpopulation.loc[int(newpopulation.loc[gene,"expressed"],2), "fitness"] = p
                    newpopulation.loc[val, "fitness"] = p

                    if direction=="x":  # maximising payoff
                        if p>max_payoff:
                                    
                                    fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                    max_payoff=p
                                    found=True
                        else:
                             pass
                    elif direction=="n":
                        if p<min_payoff:
                                    
                                    fittest=newpopulation.loc[val,"expressed"]   # if not constrained at all
                                    min_payoff=p
                                    found=True
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
        return(newpopulation,found,fittest,returned_payoff,max_payoff)
    elif direction=="n":
        return(newpopulation,found,fittest,returned_payoff,min_payoff)
    else:
        print("direction error..")
        return(newpopulation,False,"",0,0)














# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette wheel where the % breakdown score is the probability of the wheel landing on that string
# spin the wheel m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

   

def calc_mating_probabilities(newpopulation,direction,scaling,row_find_method,payoff_filename):
        count=0
        total_payoff=0.00001
        size=len(newpopulation)
        for gene in range(0,size-1):
            
            val=int(newpopulation.loc[gene,"expressed"],2)   # binary base turned into integer

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
        for gene in range(0,size-1):
            #val=int(dna[count],2)
            val=int(newpopulation.loc[gene,"expressed"],2)   # binary base turned into integer

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
                if newpopulation.loc[first_string_no-1,"expressed"]==newpopulation.loc[second_string_no-1,"expressed"]:
                    go_back=True

            mates=mates+[(newpopulation.loc[first_string_no-1,"expressed"],newpopulation.loc[second_string_no-1,"expressed"])]      # mates is a list of tuples to be mated               


      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("spin the mating wheel find the string nos - Clock: duration_clock =", duration_clock)

     #   print("len mates[]",len(mates))
     #   input("?")
        return(mates,len_sel)   # if len_sel gets small, there is a lack of genetic diversity


def crossover(self,mates,length):
        crossed=[]
        for i in mates:
            splitpoint=random.randint(1,length-1)

            child1=""
            child2=""
            remain1=i[0][:splitpoint]
            swap1=i[0][splitpoint-length:]
            remain2=i[1][:splitpoint]
            swap2=i[1][splitpoint-length:]

            child1=remain1+swap2
            child2=remain2+swap1

            crossed.append(child1)   #+[child1,child2)]
            crossed.append(child2)
        #print("crossed len",len(crossed))    
        return(crossed)


def mutate(self,dna,crossed,length,mutation_rate):
         mutation=False
         mutation_count=0
         temp=""
         gene_pool_size=len(dna)*length
 #       print("total number of bits=",totalbits)
         if gene_pool_size==0:
             print("\ntotalbits=0! error")
         #like=int(round(mutation_rate/totalbits))
         number_of_mutations_needed=int(round(gene_pool_size/mutation_rate))
   #    print("number of mutations needed=",number_of_mutations_needed)
         for m in range(0,number_of_mutations_needed):
            # flip=str(random.randint(0,1)) # a bit to change

             mut_elem=random.randint(0,len(crossed)-1)
             mut_bit=random.randint(0,length-1)
            # print("before mut",crossed[mut_elem])
            # print("mut bit no",mut_bit,"mut bit",crossed[mut_elem][mut_bit:mut_bit+1])   #=flip
         
             gene=str(crossed[mut_elem])   #v[mut_bit:mut_bit+1]="0"
              #   print(gene," to mutate at position",mut_bit)
             temp=""
             bit=0
             mut_flag=False
             for letter in gene:
                 if bit==mut_bit:              
                     new_bit=str(random.randint(0,1))  # random mutation in a random place
                     if new_bit!=gene[bit:bit+1]:
                         mut_flag=True   
                         mutation_count+=1
                     #    temp2="0"
                     #elif gene[bit:bit+1]=="0":
                     #    temp2="1"
                     #else:
                     #    print("mutation error1")
                     temp=temp+new_bit    
                 else:        
                     temp=temp+gene[bit:bit+1]
                 bit=bit+1    
                
          #   if mut_flag:
          #       print(gene,"mutated to",temp)
             crossed[mut_elem]=temp
            
             #mutation_count+=1
               #  print("dna before mutation:",dna)
           
        
         new_dna=crossed
         
         #print("new dna len=",len(new_dna))
         #input("?")

         
         return(new_dna,mutation_count,gene_pool_size)       




def main():
    ploidy=2  # number of chromosomes per individual
    no_of_alleles=7  # length of each chromosome
    pop_size=128   # population size
    epoch_count=0
    max_epochs=2
    generation_count=0

    payoff_filename="payoff_7d.csv"
    total_rows=count_file_rows(payoff_filename)
    number_of_cols=9   # rowno, 6 input vars and 1 out =8
    linewidth=76   # 66 bytes
    extra_EOL_char=0
    row_find_method="l"
    direction="x"

    epoch_length=100
    direction="x"
    scaling_factor=10000  # scaling figure is the last.  this multiplies the payoff up so that diversity is not lost on the wheel when probs are rounded
    extinctions=0
    mutation_count=0
    mutations=0
    mutation_rate=500   # mutate 1 bit in every 1000.  but the mutation is random 0 or 1 so we need to double the try to mutate rate

    
    allele_len="S"+str(no_of_alleles)

    individual = np.dtype([('fitness','f16'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
    poparray = np.zeros(pop_size, dtype=individual) 
    population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")

   # population.index
  #  print(len(population))
   # print("\n\n")
   # print(population)


    # 7 input variables 2 bits each = 14 bits, 1 floating point payoff.
    generate_payoff_environment_7d_file(linewidth,0,2**1,0,2**1,0,2**1,0,2**1,0,2**1,0,2**1,0,2**1,payoff_filename)



    for epoch in range(1,max_epochs+1):
 
        population=build_population(pop_size,no_of_alleles,population)

        print("\n\n epoch=",epoch)
        print(population)



        for generation in range(1,epoch_length+1):
            print("generation=",generation)
            print("Epochs:",epoch+1,"Generation progress: [%d%%]" % (generation/epoch_length*100))   # ,"diversity (len_sel)=",len_sel,"Tot gene bits:%d" % total_genes,"Tot Mutations:%d    "  % (mutations), end='\r', flush=True)

            # update all the fitness values in the population based on the payoff file
            population,found,fittest,returned_payoff,max_payoff=calc_fitness(population,direction,payoff_filename)  # use linecache method for fin row no in payoff file.csv

            print("\n\n calc fitness finished. fittest=",fittest)
            print(population)
            input("?")

            


            if found:   # something found   
                if int(fittest,2)<=total_rows: 
                        axis=["0"] * number_of_cols

       #print("returned payoff:",returned_payoff)
                        if direction=="x":
                            if returned_payoff>max_payoff:
                                best=fittest
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
                                print("best fittest=",best," value=",int(best,2)," row number=",int(bestrow)," a=",axis[1]," b=",axis[2]," c=",axis[3]," d=",axis[4]," e=",axis[5]," f=",axis[6]," g=",axis[7]," generation no:",gen,"max_payoff=",max_payoff,flush=True)
                                if elements>0:
                                    print("number of elements found satisfying constraint=",elements," average=",averagep)
                                outfile.write("best fittest= "+best+" value= "+str(int(best,2))+" row number= "+str(int(bestrow))+" a="+str(axis[1])+" b="+str(axis[2])+" c="+str(axis[3])+" d="+str(axis[4])+" e="+str(axis[5])+" f="+str(axis[6])+" g="+str(axis[7])+" generation no: "+str(gen)+" max_payoff= "+str(max_payoff)+"\n")
                                if elements>0:
                                    outfile.write("number of elements found satisfying constraint="+str(elements)+" average="+str(averagep)+"\n")
                                
                        elif direction=="n":
                            if returned_payoff<min_cost:
                                best=fittest
                                col=0
                                while col<=number_of_cols-1:
                                    if row_find_method=="l":
                                        axis[col]=return_a_row(int(best,2),payoff_filename).split(",")[col]
                                    elif row_find_method=="s":
                                        axis[col]=return_a_row2(int(best,2),payoff_filename).split(",")[col]
                                    else:
                                        print("row find method error.")
                                        sys.exit()

                                    col+=1

                                gen=generation_number
                                min_cost=returned_payoff
  

                                print("best fittest=",best," value=",int(best,2)," row number=",int(bestrow)," a=",axis[1]," b=",axis[2]," c=",axis[3]," d=",axis[4]," e=",axis[5]," f=",axis[6]," g=",axis[7],"generation no:",gen,"min_cost=",min_cost,flush=True)
                                if elements>0:
                                    print("number of elements found satisfying constraint=",elements," average=",averagep)
                                outfile.write("best fittest="+best+" value="+str(int(best,2))+" row number="+str(int(bestrow))+" a="+str(axis[1])+" b="+str(axis[2])+" c="+str(axis[3])+" d="+str(axis[4])+" e="+str(axis[5])+" f="+str(axis[6])+" g="+str(axis[7])+" generation no: "+str(gen)+" min_cost= "+str(min_cost)+"\n")
                                if elements>0:
                                    outfile.write("number of elements found satisfying constraint="+str(elements)+" average="+str(averagep)+" \n")



                        else:
                            print("direction error1 direction=",direction)
                else:
                    print("fittest",fittest," is beyond the environment max (",total_rows,").")
                 #  else:
                  #      print("\npayoff/cost ",returned_payoff," is outside of constraints > max",xgene.maxp," or < min",xgene.minp)

                  #  clock_start=time.process_time()
   


                wheel=calc_mating_probabilities(population,direction,scaling_factor,row_find_method,payoff_filename)  # scaling figure is the last.  this multiplies the payoff up so that divsity is not lost on the wheel when probs are rounded
                if len(wheel)==0:
                    print("wheel empty")
                    sys.exit
            #print(wheel)


                mates,len_sel=spin_the_mating_wheel(wheel,population,pop_size)  # sel_len is the size of the unique gene pool to select from in the wheel
                # mates is a lkist of tuples

                print("len mates=",len(mates))
                print("mates=",mates)
                print(population)
                print("\ncall crossover")
               # population,crossed=crossover(population,mates,length)
                print("crossover finished.")
                print(population)
                input("?")

                #population,mutation_count,gene_pool_size=mutate(population,crossed,length,mutation_rate)   # 1000 means mutation 1 in a 1000 bits processed

 
        
              #  total_genes=total_genes+gene_pool_size
                mutations=mutations+mutation_count
             #   generation_number+=1

            else:
                pass   # nothing found
            

            
          #  starting_population=int(round(starting_population/2))    #  increase the starting population between the different epochs to test the results
         #   clock_end=time.process_time()
          #  duration_clock=clock_end-clock_start
          #  print("\n\nFinished - Clock: duration_clock =", duration_clock)
          #  outfile.write("\nFinished - Clock: duration_clock ="+str(duration_clock)+"\n")
          
            print("")



 



if __name__ == '__main__':
    main()





#print("col=",pop.columns)

#   #row=int(input("row?"))
    #col=input("col?")


    #print("=",population[col][row])    
    #print("?",population.loc[row,col])
#print("\n\n")
#print(pop.index)


#list(pop.items())
#population["xpoint1"]=45

#print(pop.keys())

#pop.keys()
#print(population["xpoint1"])

#print(population["xpoint1"][2])
  



