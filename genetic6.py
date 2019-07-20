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

global payoff_filename,number_of_cols
payoff_filename="/home/pi/Python_Lego_projects/payoff_2d.csv"
number_of_cols=4


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
       # print(dna)
       # input("?")
        return(dna)    

    def return_a_row(self,row,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(linecache.getline(filename,row+1).rstrip())
        except IndexError:
            return("Index error")
        


    def find_a_payoff(self,row,filename):   # assumes the payoff is the last field in a CSV delimited by ","
      #  print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(float(linecache.getline(filename,row+1).split(",")[-1]))
        except IndexError:
            return(0.0)
        

    def find_a_row_and_column(self,row,col,filename):   # assumes the payoff is the last field in a CSV delimited by ","
       # print("line no:",row,":",linecache.getline(filename,row+1))   #.split(",")[-1])
        try:
            return(float(linecache.getline(filename,row+1).split(",")[col]))
        except IndexError:
            return(0.0)



    def calc_fitness(self,dna,direction):
        max_payoff=-100000.0
        min_payoff=100000.0
        p=0.0
        count=0
        best=""
        fitness=[]
        for elem in dna:
            val=int(dna[count],2)
          #  p=payoff[val]
          #  p=self.find_a_payoff_1d(val,self.payoff_filename)
            p=self.find_a_payoff(val,payoff_filename)
      #      print("p=",p," val=",val)
            fitness.append(p)
            if direction=="x":  # maximising payoff
                if p>max_payoff:
                    best=dna[count]
                    max_payoff=p
            elif direction=="n":    # minimising cost
                if p<min_payoff:
                    best=dna[count]
                    min_payoff=p
            else:
                print("calc fitness direction error.")

          #  print("#",count+1,":",elem," val=",val,"payoff=",p," fitness=",fitness[count])
          #  print("best=",best," max",max_payoff)
            count+=1
        if direction=="x":
            return(best,max_payoff)
        elif direction=="n":
            return(best,min_payoff)
        else:
            return(0,0)


# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette wheel where the % breakdown score is the probability of the wheel landing on that string
# spin the wheel m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.

   

    def calc_mating_probabilities(self, dna,direction):
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
       # nor_payoff=total_payoff*1000
       # print("np=",nor_payoff)
        for elem in dna:
            val=int(dna[count],2)
            p=self.find_a_payoff(val,payoff_filename)
            if direction=="x":   # maximise
                wheel.append(int(round(p/total_payoff*1000)))
         #       print("#",count+1,":",elem," val=",val,"payoff=",payoff[val]," prob=",wheel[count])
 
            elif direction=="n":   # minimise
                wheel.append(int(round(-p/total_payoff*1000)))
      #         print("#",count+1,":",elem," val=",val,"cost=",payoff[val]," prob=",wheel[count])
 
            else:
                print("direction error3")

#           print("#",count+1,":",elem," val=",val,"payoff=",payoff[val]," prob=",wheel[count])
            count+=1
       # print(wheel)
       # input("?")
        return(wheel)

    
    def spin_the_mating_wheel(self,wheel,dna,iterations):
        sel=[]
        mates=[]
        n=0

   # clock_start=time.clock()

        wheel_len=len(wheel)

        while n<=wheel_len-1: 
            sel=sel+([n+1] * wheel[n])
            n=n+1



        len_sel=len(sel)
      #  if len_sel==0:
      #      print("len =0"," wheel len=",wheel_len)
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

            mates=mates+[(dna[first_string_no-1],dna[second_string_no-1])]                     


      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("spin the mating wheel find the string nos - Clock: duration_clock =", duration_clock)

        #print(mates,len(mates))
        return(mates)


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
             print("totalbits=0! error")
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
             for letter in gene:
                 if bit==mut_bit:
                     if gene[bit:bit+1]=="1":
                         temp2="0"
                     elif gene[bit:bit+1]=="0":
                         temp2="1"
                     else:
                         print("mutation error1")
                     temp=temp+temp2    
                 else:        
                     temp=temp+gene[bit:bit+1]
                 bit=bit+1    
                
            # print("mutated=",temp)
             crossed[mut_elem]=temp
            
             mutation_count+=1
               #  print("dna before mutation:",dna)
           
        
         new_dna=crossed
         
         #print("new dna len=",len(new_dna))
         #input("?")

         
         return(new_dna,mutation_count,gene_pool_size)       






length=16   #16   # 16 length of dna bit strings
max_payoff=0
min_cost=0
starting_population=256   #256   #256 (16 bits), 64 (13 bits), 32 (10 bits)   #16 for 8 bits #4 for 5 bits
best=""
returnedpayoff=0
gen=0
epoch_length=60
extinction_events=1
extinctions=0
mutation_count=0
mutations=0
mutation_rate=1000   # mutate 1 bit in every 1000

# xaxis string length 16, starting population 256
xgene=gene_string(length,starting_population)  # instantiate the gene string object for the x axis
ygene=gene_string(length,starting_population)



print("Creating payoff environment....")
#payoff=generate_payoff_environment_1d(0,32)  # 5 bits
#payoff=generate_payoff_environment_1d(0,64)  # 6 bits
#payoff=generate_payoff_environment_1d(0,256)  #8 bits
#payoff=generate_payoff_environment_1d(0,512)  #9 bits
#payoff=generate_payoff_environment_1d(0,1024)  #10 bits
#payoff=generate_payoff_environment_1d(0,4096)  #12 bits
#payoff=generate_payoff_environment_1d(0,8192)  #13 bits
#payoff=generate_payoff_environment_1d(0,2**length)  #16 bits
#generate_payoff_environment_1d_file(0,2**length+1,payoff_filename)  #"/home/pi/Python_Lego_projects/payoff_1d.csv")  #16 bits

generate_payoff_environment_2d_file(0,2**8,0,2**8,payoff_filename)  #"/home/pi/Python_Lego_projects/payoff_2d.csv")  # 8x8 bits
#size=2**length
#for r in range(1,20):
#print(xgene.find_a_payoff(3245,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,0,payoff_filename))  #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,1,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,2,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,3,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")
#print(xgene.find_a_row_and_column(3245,4,payoff_filename))   #"/home/pi/Python_Lego_projects/payoff_1d.csv")





#input("?")

print("Payoff/Cost Environment")
#print(payoff)
direction=""
while direction!="x" and direction!="n":
    direction=input("Ma(x)imise or Mi(n)imise?")
#payoff=generate_payoff_environment_2d(1,5,6,13)
#print(payoff)
while extinctions<=extinction_events:

  
    generation_number=1
    dna=xgene.generate_dna(length,starting_population)
    #print(dna)
    total_genes=0
    gene_pool_size=0
    mutations=0
    mutation_count=0
    print("Number of extinctions:",extinctions," Running for ",epoch_length," generations.")
    while generation_number<=epoch_length:
        print("\rExtinctions:",extinctions," Generation progress: [%d%%] " % (generation_number/epoch_length*100) ," Tot gene bits:%d  " % total_genes," Tot Mutations:%d"  % (mutations), end='\r', flush=True)

     #   clock_start=time.clock()

        fittest,returned_payoff=xgene.calc_fitness(dna,direction)
     #   print("fittest=",fittest," returned=",returned_payoff)
     #   clock_end=time.clock()
     #   duration_clock=clock_end-clock_start
     #   print("calc fitness - Clock: duration_clock =", duration_clock)
    
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
                print("best fittest=",best," value=",int(best,2)," row number=",axis[0]," x=",axis[1]," y=",axis[2]," generation no:",gen," max_payoff=",max_payoff)
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
                print("best fittest=",best," value=",int(best,2)," row number=",axis[0]," x=",axis[1]," y=",axis[2]," generation no:",gen," min_cost=",min_cost)
        else:
            print("direction error1 direction=",direction)

      #  clock_start=time.clock()
   
        wheel=xgene.calc_mating_probabilities(dna,direction)
        #print(wheel)

      #  clock_end=time.clock()
      #  duration_clock=clock_end-clock_start
      #  print("calc mating probabilities - Clock: duration_clock =", duration_clock)

      #  clock_start=time.clock()

        mates=xgene.spin_the_mating_wheel(wheel,dna,starting_population*3)

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
def read_chunks(file_handle, chunk_size=8192):
    while True:
        data = file_handle.read(chunk_size)
        if not data:
            break
        yield data

def sha256(file_handle):
    hasher = hashlib.sha256()
    for chunk in read_chunks(file_handle):
        hasher.update(chunk)
    return hasher.hexdigest()



def display_a_hash(hash_object):
    if len(hash_object)==64:
        print("SHA256 hash")  #,hash_object)
        print("############")
        print("#          #")  
        for row in range(0,63,8):
            print("# "+hash_object[row:row+8]+" #")
        print("#          #")
        print("############\n")
    else:
        print("Hash not 64 bytes.  error")
    
               



def split_a_file_in_2(infile):

        #infile = open("input","r")

        with open(infile,'r') as f:
            linecount= sum(1 for row in f)

        splitpoint=linecount/2

        f.close()

        infilename=os.path.splitext(infile)[0]

        f = open(infile,"r")
        outfile1 = open(infilename+"001.csv","w")
        outfile2 = open(infilename+"002.csv","w")

        print("linecount=",linecount , "splitpoint=",splitpoint)

        linecount=0

        for line in f:
            linecount=linecount+1
            if ( linecount <= splitpoint ):
                outfile1.write(line)
            else:
                outfile2.write(line)

        f.close()
        outfile1.close()
        outfile2.close()


    
def count_file_rows(filename):
        with open(filename,'r') as f:
            return sum(1 for row in f)

   

def join2files_dos(in1,in2,out):
        os.system("copy /b "+in1+"+"+in2+" "+out)

def join2files_deb(in1,in2,out):
        os.system("cat "+in1+" "+in2+" "+out)

"""

try:
    with open("salestrans060719.csv", 'rb') as f:
        hash_string = sha256(f)
    print("hash=",hash_string)
except IOError as e:
    print("error test")



display_a_hash(hash_string)

print(count_file_rows("salestrans060719.csv"))

split_a_file_in_2("salestrans060719.csv")


join2files_deb("salestrans060719001.csv","salestrans060719002.csv","newsalestrans060719.csv")
print(count_file_rows("newsalestrans060719.csv"))

try:
    with open("newsalestrans060719.csv", 'rb') as f:
        hash_string = sha256(f)
    print("hash=",hash_string)
except IOError as e:
    print("error test")

display_a_hash(hash_string)
"""
