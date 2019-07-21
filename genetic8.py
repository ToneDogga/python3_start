# Genetic algorithms started 17/7/19
#
# Basic structure for a simple algorthim

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

from __future__ import division

import sys
import hashlib, os
import random
import math
import time
import linecache
import platform

global payoff_filename,number_of_cols, total_rows
payoff_filename="/home/pi/Python_Lego_projects/payoff_3d.csv"
number_of_cols=5



def generate_payoff_environment_1d(xstart_val,xsize_of_env):   
    payoff = [1-10*math.sin(x/44)*12*math.cos(x/33)*1000/(x+1) for x in range(xstart_val,xstart_val+xsize_of_env)]
        #for x in range(xstart_val,xstart_val+xsize_of_env):
        #    payoff[x]=x**2  # example function
    return(payoff)    


def generate_payoff_environment_1d_file(xstart_val,xsize_of_env,filename):   
    rowno=0
    with open(filename,"w") as f:
        for x in range(xstart_val,xstart_val+xsize_of_env):
            payoff=-10*math.sin(x/44)*12*math.cos(x/33)*1000/(x+1) 
            f.write(str(rowno)+","+str(x)+","+str(payoff)+"\n")
            rowno+=1
        #for x in range(xstart_val,xstart_val+xsize_of_env):
        #    payoff[x]=x**2  # example function
    f.close()   
    return(rowno)


def generate_payoff_environment_2d(xstart_val,xsize_of_env,ystart_val,ysize_of_env):
    payoff = [[x**2+y*3 for x in range(xstart_val,xstart_val+xsize_of_env)] for y in range(ystart_val,ystart_val+ysize_of_env)]
        #for x in range(xstart_val,xstart_val+xsize_of_env):
        #    for y in range(ystart_val,ystart_val+ysize_of_env):
        #        payoff[x][y]=x**2+3*y   # example function
    print(payoff)
    input("?")
    return(payoff)

def generate_payoff_environment_2d_file(xstart_val,xsize_of_env,ystart_val,ysize_of_env,filename):   
    rowno=0
    
    with open(filename,"w") as f:
        for x in range(xstart_val,xstart_val+xsize_of_env):
            for y in range(ystart_val,ystart_val+ysize_of_env):
                payoff=-10*math.sin(x/44)*12*math.cos(y/33)
                f.write(str(rowno)+","+str(x)+","+str(y)+","+str(payoff)+"\n")
                rowno+=1
        #for x in range(xstart_val,xstart_val+xsize_of_env):
        #    payoff[x]=x**2  # example function
        
    f.close()
    return(rowno)


def generate_payoff_environment_3d_file(xstart_val,xsize_of_env,ystart_val,ysize_of_env,zstart_val,zsize_of_env,filename):   
    rowno=0
    total_rows=xsize_of_env*ysize_of_env*zsize_of_env    
    with open(filename,"w") as f:
        for x in range(xstart_val,xstart_val+xsize_of_env):
            print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
            for y in range(ystart_val,ystart_val+ysize_of_env):
                for z in range(zstart_val,zstart_val+zsize_of_env):
                    payoff=-10*math.sin(x/44)*12*math.cos(y/33)-13*math.tan(z/55)
                    f.write(str(rowno)+","+str(x)+","+str(y)+","+str(z)+","+str(payoff)+"\n")
                    rowno+=1        
    f.close()
    print("")
    return(rowno)


def count_file_rows(filename):
    with open(filename,'r') as f:
        return sum(1 for row in f)

######################################################


class gene_string(object):

    def __init__(self,length,starting_population): 
       # self.length=16   # 16 length of dna bit strings
     #   self.max_payoff=0
     #   self.min_payoff=0
       # self.starting_population=256   #256 (16 bits), 64 (13 bits), 32 (10 bits)   #16 for 8 bits #4 for 5 bits
    #    best=""
      #  self.payoff_filename="/home/pi/Python_Lego_projects/payoff_1d.csv"
         self.gen=0
    #    epoch_length=60
    #    extinction_events=4
    #    extinctions=0
    #    mutation_count=0
    #    mutations=0
   

    def generate_dna(self, length, number):
        dna=[]
        for count in range(0,number):
            go_back=True
            while go_back:
                d=""
                for size in range(0,length):
                   bit=str(random.randint(0,1))
                   d=d+bit
                # check if we have already that string in the population. if so discard and generate another
                #print(d)
                go_back=False
                for elem in dna:
                    if d==elem:
                        go_back=True
                        break
                                        
            dna.append(d)   
          #  print(" dna[",count,"]=",dna[count]," dna value=",int(dna[count],2))
      #  print(dna)
      #  input("?")
        return(dna)    

    def return_a_row(self,row,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(linecache.getline(filename,row+1).rstrip())
        except IndexError:
            print("\nindex error")
            return("Index error")
        except ValueError:
            print("\nvalue error") 
            return("value error")
        except IOError:
            print("\nIO error")
            return("IO error")
       


    def find_a_payoff(self,row,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(float(linecache.getline(filename,row+1).split(",")[-1]))
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
            print("\nvalue error")
            return("Value error")
        except IOError:
            print("\nIO error")
            return("IO error")

    def find_a_row_and_column(self,row,col,filename):   # assumes the payoff is the last field in a CSV delimited by ","
       # print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(float(linecache.getline(filename,row+1).split(",")[col]))
        except IndexError:
            print("\nindex error")
            return(0.0)
        except ValueError:
            print("\nvalue error")
            return("Value error")
        except IOError:
            print("\nIO error")
            return("IO error")



    def calc_fitness(self,dna,direction):
        max_payoff=-100000.0
        min_payoff=100000.0
        p=0.0
        count=0
        best=""
      #  fitness=[]
        for elem in dna:
            val=int(dna[count],2)
            if val <= total_rows:
              #  p=payoff[val]
              #  p=self.find_a_payoff_1d(val,self.payoff_filename)
                p=self.find_a_payoff(val,payoff_filename)
      #          print("p=",p," val=",val)
       #         fitness.append(p)
                if direction=="x":  # maximising payoff
                    if p>max_payoff:
                        best=dna[count]
                        bestrow=self.find_a_row_and_column(val,0,payoff_filename)
                        max_payoff=p
                elif direction=="n":    # minimising cost
                    if p<min_payoff:
                        best=dna[count]
                        bestrow=self.find_a_row_and_column(val,0,payoff_filename)
                        min_payoff=p
                else:
                    print("\ncalc fitness direction error.")
            else:
                print("\nval ",val," is greater than total environment (",total_rows,")")
          #  print("#",count+1,":",elem," val=",val,"payoff=",p," fitness=",fitness[count])
          #  print("best=",best," max",max_payoff)
            count+=1
        if direction=="x":
            return(bestrow,best,max_payoff)
        elif direction=="n":
            return(bestrow,best,min_payoff)
        else:
            print("direction error..")
            return(0,0,0)


# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette wheel where the % breakdown score is the probability of the wheel landing on that string
# spin the wheel m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

   

    def calc_mating_probabilities(self, dna,direction,scaling):
        count=0
        total_payoff=0.00001
        for elem in dna:
            val=int(dna[count],2)
          #  total_payoff+=payoff[val]
            total_payoff+=self.find_a_payoff(val,payoff_filename)
        #print(val,payoff[val])
            count+=1

#    print("payoff=",payoff)
#    input("?")

        count=0
        wheel=[]
        if len(dna)<=1:
            print("\nlen(dna)<=1!")
     #   else:
     #       print("\nlen dna=",len(dna))
       # nor_payoff=total_payoff*1000
      #  print("\ntotal payoff=",total_payoff)
        for elem in dna:
            val=int(dna[count],2)
            p=self.find_a_payoff(val,payoff_filename)
            if direction=="x":   # maximise
                wheel.append(int(round(p/total_payoff*scaling)))
         #       print("#",count+1,":",elem," val=",val,"payoff=",payoff[val]," prob=",wheel[count])
 
            elif direction=="n":   # minimise
                wheel.append(int(round(-p/total_payoff*scaling)))
      #         print("#",count+1,":",elem," val=",val,"cost=",payoff[val]," prob=",wheel[count])
 
            else:
                print("\ndirection error3")

       #     print("#",count+1,":",elem," val=",val,"payoff=",p," prob=",wheel[count])
            count+=1
       # print("\nlen wheel",len(wheel))
       # input("?")
        return(wheel)

    
    def spin_the_mating_wheel(self,wheel,dna,iterations):
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
    #    print("\nlen(sel)=",len_sel,"sel=",sel)
    #    input("?")
       
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
                if dna[first_string_no-1]==dna[second_string_no-1]:
                    go_back=True

            mates=mates+[(dna[first_string_no-1],dna[second_string_no-1])]      # mates is a list of tuples to be mated               


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


    def mutate(self,crossed,length,mutation_rate):
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






length=22   #16   # 16 length of dna bit strings
max_payoff=0
min_cost=0
starting_population=10000   #256   #256 (16 bits), 64 (13 bits), 32 (10 bits)   #16 for 8 bits #4 for 5 bits
best=""
returnedpayoff=0
gen=0
epoch_length=60
extinction_events=1
scaling_factor=10000  # scaling figure is the last.  this multiplies the payoff up so that diversity is not lost on the wheel when probs are rounded
extinctions=0
mutation_count=0
mutations=0
mutation_rate=500   # mutate 1 bit in every 1000.  but the mutation is random 0 or 1 so we need to double the try to mutate rate

# xaxis string length 16, starting population 256
xgene=gene_string(length,starting_population)  # instantiate the gene string object for the x axis
#ygene=gene_string(length,starting_population)


print("\n\nGenetic algorithm. By Anthony Paech")
print("===================================")
print("Platform:",platform.machine(),"\n:",platform.platform(),"\n:",platform.system())
print("\n:",platform.processor(),"\n:",platform.version(),"\n:",platform.uname())
print("\n\nTheoretical max no of rows for the environment file:",payoff_filename,"is:",sys.maxsize)

#payoff=generate_payoff_environment_1d(0,32)  # 5 bits
#payoff=generate_payoff_environment_1d(0,64)  # 6 bits
#payoff=generate_payoff_environment_1d(0,256)  #8 bits
#payoff=generate_payoff_environment_1d(0,512)  #9 bits
#payoff=generate_payoff_environment_1d(0,1024)  #10 bits
#payoff=generate_payoff_environment_1d(0,4096)  #12 bits
#payoff=generate_payoff_environment_1d(0,8192)  #13 bits
#payoff=generate_payoff_environment_1d(0,2**length)  #16 bits
#generate_payoff_environment_1d_file(0,2**length+1,payoff_filename)  #"/home/pi/Python_Lego_projects/payoff_1d.csv")  #16 bits

#total_rows=generate_payoff_environment_2d_file(0,2**8,0,2**8,payoff_filename)  #"/home/pi/Python_Lego_projects/payoff_2d.csv")  # 8x8 bits
print("counting rows in ",payoff_filename)
total_rows=count_file_rows(payoff_filename)
print("Payoff/cost environment file is:",payoff_filename,"and has",total_rows,"rows.")
answer=""
while answer!="y" and answer!="n":
    answer=input("Create payoff env? (y/n)")
if answer=="y":
    print("Creating payoff/cost environment....file:",payoff_filename)
    clock_start=time.process_time()

    total_rows=generate_payoff_environment_3d_file(0,2**8,0,2**8,0,2**6,payoff_filename)  
    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("generate payoff/cost environment - Clock: duration_clock =", duration_clock,"seconds.")
    print("Payoff/cost environment file is:",payoff_filename,"and has",total_rows,"rows.")

    
#size=2**length
#for r in range(1,20):
#print(xgene.find_a_payoff(3245,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,0,payoff_filename))  #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,1,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,2,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,3,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,4,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")




#print("total rows=",total_rows)
#input("?")

print("\n\nPayoff/Cost Environment")
#print(payoff)
direction=""
while direction!="x" and direction!="n":
    direction=input("Ma(x)imise or Mi(n)imise?")
#payoff=generate_payoff_environment_2d(1,5,6,13)
#print(payoff)
while extinctions<=extinction_events:

     
    generation_number=1
    print("Generating",starting_population,"unique elements of random DNA of length:",length,". Please wait....")
    dna=xgene.generate_dna(length,starting_population)
    len_sel=len(dna)
    #print(dna)
    total_genes=0
    gene_pool_size=0
    mutations=0
    mutation_count=0
    print("Number of extinctions:",extinctions,"Running for",epoch_length,"generations.")
    while generation_number<=epoch_length:
        print("Extinctions:",extinctions,"Generation progress: [%d%%]" % (generation_number/epoch_length*100) ,"diversity (len_sel)=",len_sel,"Tot gene bits:%d" % total_genes,"Tot Mutations:%d"  % (mutations), end='\r', flush=True)

     #   clock_start=time.clock()

        bestrow,fittest,returned_payoff=xgene.calc_fitness(dna,direction)
     #   print("fittest=",fittest," returned=",returned_payoff)
     #   clock_end=time.clock()
     #   duration_clock=clock_end-clock_start
     #   print("calc fitness - Clock: duration_clock =", duration_clock)


        if int(fittest,2)<=total_rows: 
            axis=["0"] * number_of_cols

            if direction=="x":
                if returned_payoff>max_payoff:
                    best=fittest
                    col=0
                    while col<=number_of_cols-1:
                        axis[col]=xgene.return_a_row(int(best,2),payoff_filename).split(",")[col]
                        col+=1

                    gen=generation_number
                    max_payoff=returned_payoff
                    print("best fittest=",best," value=",int(best,2)," row number=",int(bestrow)," x=",axis[1]," y=",axis[2]," z=",axis[3],"generation no:",gen,"max_payoff=",max_payoff,flush=True)
            elif direction=="n":
                if returned_payoff<min_cost:
                    best=fittest
                    col=0
                    while col<=number_of_cols-1:
                        axis[col]=xgene.return_a_row(int(best,2),payoff_filename).split(",")[col]
                        col+=1

                    gen=generation_number
                    min_cost=returned_payoff
                    col=0
                    print("best fittest=",best," value=",int(best,2)," row number=",int(bestrow)," x=",axis[1]," y=",axis[2]," z=",axis[3],"generation no:",gen,"min_cost=",min_cost,flush=True)
            else:
                print("direction error1 direction=",direction)
        else:
            print("fittest",fittest," is beyond the environment max (",total_rows,").")
      #  clock_start=time.clock()
   
        wheel=xgene.calc_mating_probabilities(dna,direction,scaling_factor)  # scaling figure is the last.  this multiplies the payoff up so that divsity is not lost on the wheel when probs are rounded
        if len(wheel)==0:
            print("wheel empty")
            sys.exit
        #print(wheel)

      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("calc mating probabilities - Clock: duration_clock =", duration_clock)

      #  clock_start=time.clock()

        mates,len_sel=xgene.spin_the_mating_wheel(wheel,dna,starting_population)  # sel_len is the size of the uniqeu gene pool to select from in the wheel

      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("spin the mating wheel - Clock: duration_clock =", duration_clock)

      #  clock_start=time.clock()
        
        crossed=xgene.crossover(mates,length)

      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("crossover - Clock: duration_clock =", duration_clock)
        
      #  clock_start=time.clock()
       
        #print(crossed)

        dna,mutation_count,gene_pool_size=xgene.mutate(crossed,length,mutation_rate)   # 1000 means mutation 1 in a 1000 bits processed

      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("mutate - Clock: duration_clock =", duration_clock)

        
        total_genes=total_genes+gene_pool_size
        mutations=mutations+mutation_count
        generation_number+=1

    extinctions+=1
    print("")







###############################################
