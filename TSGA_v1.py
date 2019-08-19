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
  
        



def return_a_row_from_envir_using_rowno(val,env):
        # use the preloaded pandas dataframe of the payoff file
        try:
            
            return(env.iloc[val].values.tolist())
         #   if ret:
         #       return(ret)
         #   else:
         #       print("val not found in environment")
         #       return([])
            
        except IndexError:
            print("\nindex error")
            return([])
        except ValueError:
            print("\nvalue error val=",val)
            return([])
        except IOError:
            print("\nIO error")
            return([])





def return_a_row_from_envir_using_concatno(val,env):
        # use the preloaded pandas dataframe of the payoff file
        try:
            
            return(env.loc[env["concatno"]==val].values.tolist()[0])
         #   if ret:
         #       return(ret)
         #   else:
         #       print("val not found in environment")
         #       return([])
            
        except IndexError:
            print("\nindex error")
            return([])
        except ValueError:
            print("\nvalue error val=",val)
            return([])
        except IOError:
            print("\nIO error")
            return([])



def return_a_row_from_file(val,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
        try:
            with open(filename,"r") as f:
                f.seek((r.linewidth+r.extra_eol_char)*val)   # if a windows machine add an extra char for the '\r' EOL char       
                p=["0",f.readline().split(',')]  # add an extra element to the row because the concatno field is not in the payoff file
                return(list(itertools.chain(*p)))
                
        except IndexError:
            print("\nindex error")
            return([])
        except ValueError:
            print("\nvalue error row+1=",val+1)
            return([])
        except IOError:
            print("\nIO error")
            return([])


def return_a_row_from_linecache(val,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            l=linecache.getline(filename,val+1).rstrip()
            p=["0",l.split(',')]  # add an extra element to the row because the concatno field is not in the payoff file
         #   print("p=",p)
            return(list(itertools.chain(*p)))
 
            
        except IndexError:
            print("\nindex error")
            return("Index error")
        except ValueError:
       #     print("\nvalue error row+1=",row+1) 
            return("value error")
        except IOError:
            print("\nIO error")
            return("IO error")




# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the probability_table m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

  

def calc_mating_probabilities(newpopulation,r,env):
      count=0
      total_payoff=0.00001

      for gene in range(0,r.pop_size):
        fittest=newpopulation.loc[gene,"expressed"]
      #  val=int(fittest,2)   # binary base turned into integer
        total_payoff+=abs(return_a_row_from_envir_using_rowno(int(fittest,2),env)[8])

        count=0
        probability_table=[]
        if len(newpopulation)<=1:
            print("\nlen(dna)<=1!")

            
        for gene in range(0,r.pop_size):
            #val=int(dna[count],2)(newpopulation.loc[gene,"expressed"]
            fittest=newpopulation.loc[gene,"expressed"]
#            val=int(fittest,2)   # binary base turned into integer
            p=abs(return_a_row_from_envir_using_rowno(int(fittest,2),env)[8])

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


    mates_df = pd.DataFrame(mates, columns=["fitness","parentid1","xpoint1","parentid2","xpoint2","chromo1","chromo2","newchromo1","newchromo2","expressed"])


  
    mates_df=mates_df.drop(columns=["chromo1","chromo2"])   # delete old chromosomes columns in population
    mates_df.columns=["fitness","parentid1","xpoint1","parentid2","xpoint2","chromo1","chromo2","expressed"]  # rename newchromos to chromos

    mates_df.index
    
    return(mates_df)





def pmx(mates,no_of_alleles,individual,pop_size):    # partially matched crossover


    # this is used when the order of the bits matters sych as in a travelling salesman problem
    # select two random cutting points that are not the same on the length of the chromosome
    #  a =   012|3456|789
    #  a =   01%|001%|101
    #  b =   012|3456|789   (locus positions)
    #  b =   110|%%00|111
#   map string a to string b
  #  moves to:
  #   a' =   01%|%%00|101
  #   b' =   110|001%|111
  #

  # now we haveto swap back the "duplicates"
  # number 3,4,5,6 have been mapped string b to string a
  # now on b, 
#  then mapping string a onto b,
#   a" =     
#   b" =     
#
#  the other 5,6 and 7 in b swap with the a's 2,3 and 0
  # so each string contains ordering information partially determined by each of the parents
   #

 #   print("pmx crossover. alleles=",no_of_alleles)

    pmx_count=0

    c1_choice_list=[]
    c2_choice_list=[]
    old_chromo_list=[]
    new_chromo_list=[]
    
 #   gene_pool_size=no_of_alleles*ploidy*pop_size
  #  number_of_pmx_needed=int(round(gene_pool_size/pmx_rate))
   # for m in range(0,number_of_pmx_needed):
    
    pmx_bit=""
    chromo=""


    mate1col=5
    mate2col=6

    xpoint1col=2
    xpoint2col=4

    newchromo1col=7
    newchromo2col=8

    for mate in mates:
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



        child1=""
        child2=""
        remain11=mate[mate1col][:cut1_choice]
        remain12=mate[mate1col][cut2_choice:]
        swap1=mate[mate1col][:cut2_choice]
        swap11=swap1[cut1_choice:]

        remain21=mate[mate2col][:cut1_choice]
        remain22=mate[mate2col][cut2_choice:]
        swap2=mate[mate2col][:cut2_choice]
        swap21=swap2[cut1_choice:]

        print("remain11",remain11,"+swap11+",swap11,"+ remain12",remain12)
        print("remain21",remain21,"+swap21+",swap21,"+ remain22",remain22)
        input("?")

       
        child1=remain11+swap21+remain12
        child2=remain21+swap11+remain22

        print("swap")
        print("before:",mate[mate1col],mate[mate2col])
        print("after:",child1,child2)


        swaps=list(range(cut1_choice,cut2_choice))
        print("swaps=",swaps)

     #   for a,i in swap11:
     #       result = []
     #       for i, c in enumerate(swap11):
##                if c<a or c>b:
##                    result.append(i)
##    
##        print(result)    
## 
##
##
##
##        mates[row][newchromo1col]=child1   #+[child1,child2)]
##        mates[row][newchromo2col]=child2
##        mates[row][xpoint1col]=splitpoint
##        mates[row][xpoint2col]=splitpoint
##        row+=1    



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


        newc=chromo[:c3_choice]+mutated_bit+chromo[c3_choice+1:]


        if c1_choice==1:
            # chromo1
            newpopulation.loc[c2_choice,"chromo1"]=newc
            
        else:
            # chromo2
            newpopulation.loc[c2_choice,"chromo2"]=newc

        mutation_count+=1  
        
    return(newpopulation,mutation_count, c1_choice_list, c2_choice_list)

def create_journey(j):
    j.journey=[0]   # starting point
    for s in range(1,j.stops-1):
        found=True
        while found:
            point=(random.randint(0,j.xsize-1),random.randint(0,j.ysize-1))
            # check if point already exists
            for h in j.journey:
               if (point==h):
                   found=True
                   break
               else:
                   found=False

        j.journey.append(point)
    j.journey.append(0)   # return to starting point  (0)
    print("journey=",j.journey)
    return(j)  


#######################################################

def main():

    class ts(object):
        pass

    ts.xsize=0
    ts.ysize=0
    ts.stops=0
    ts.journey=[]
    ts.distances=[]

    ts.poolsize=16
    ts.genepool=[]

    epoch_length=100
    

    ts.xsize=input("x size?")
    ts.ysize=input("y size?")
    ts.stops=int(input("number of stops?"))

    ts.stops+=1   #  has to return back to the start

    print("create journey")
    create_journey(ts)
##    print("create distance table")
##    create_distance_table(ts)
##    print("create starting genepool")
##    create_starting_genepool(ts)
##
##    
##        
##
##    
##    print("starting....")
##    for gencount in range(1,epoch_length+1):
##        mutation_count=0
##
##        size=len(population)
##        
##        for gene in range(0,size):  #newpopulation.iterrows():
##
##            plist=[]
##            fittest=population.loc[gene,"expressed"]   # binary base turned into integer
##            population.loc[gene, "fitness"] = p
##          
##            try:            
##                 plist=return_a_row_from_linecache(int(fittest,2),r.payoff_filename)                  
##                       
##            
##            except ValueError:
##                print("value error finding p in calc fitness")
##                sys.exit()
##            except IOError:
##                print("File IO error on ",r.payoff_filename)
##                sys.exit()
##
##            if len(plist)>7:
##                found=True
##                a=int(plist[1])
##                returned_payoff=p
## 
##                elements_count+=1
##
##                population.loc[gene, "fitness"] = p
##
##                if p>max_payoff:
##                    max_payoff=p
##                    max_fittest=fittest
##                    best_rowno=gene   #rowno
##                    best_gen=generation
##                    best_epoch=epoch
##                    best_fittest=fittest
##                   # bestflag=True
##                    #bestpopulation=population.copy()
##                    best_copy=True
##                    besta=a
##                                                         
##            else:
##                r.outfile.write("Row "+fittest+" not found in environment")
##                
##         
##
##
##            count+=1
##
##           # if elements_count>0:
##           #     averagep=totalp/elements_count
##
##        if best_copy:
##            best_copy=False
##            update_fittest=True
##            bestpopulation=population.copy()
##
##
##
##
##        print("Epoch",epoch,"of",r.no_of_epochs)
##        print("Epoch progress:[%d%%]" % (generation/r.epoch_length*100)," Generation no:",generation,"fittest of this generation:",fittest)
##
##
##        print("\nGenepool. Ave Fitness=",avefitness,"#Duplicates expressed=",dupcount,"of",allcount,". Diversity of probability_table weighting=[",len_wheel,"]. probability_table[] size=",len_probability_table)
##        print("\n",nondup_par1,"unique first parents, and",nondup_par2," unique second parents of",allcount,"chomosomes.")
##        print("\nScaling factor=",r.scaling_factor," * genepool size",r.pop_size," = actual scaling:",r.actual_scaling)
##
##        print("\nFittest so far:",best_fittest," best rowno [",best_rowno,"] in best generation [",best_gen,"] in best epoch [",best_epoch,"] max payoff",max_payoff)
##        print("\nBest overall so far: a=",besta," b=",bestb," c=",bestc," d=",bestd," e=",beste," f=",bestf," g=",bestg)
##        print("\nTotal Progress:[%d%%]" % (round(((total_generations+1)/(r.epoch_length*r.no_of_epochs))*100)),"\n",flush=True)
##
##        r.outfile.flush()
##        r.results.flush()
##        
##        probability_table=calc_mating_probabilities(population,r,env)
##        # scaling figure is the last.  this multiplies the payoff up so that divsity is not lost on the probability_table when probs are rounded
##
##        len_probability_table=len(probability_table)
##        if len_probability_table==0:
##            print("probability_table empty")
##            sys.exit
## 
##
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


