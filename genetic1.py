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

import sys
import hashlib, os
import random



def generate_dna(number, length):
    dna=[]
    for count in range(0,number):
        d=""
        for size in range(0,length):
           bit=str(random.randint(0,1))
           d=d+bit
        dna.append(d)   
     #   print(" dna[",count,"]=",dna[count]," dna value=",int(dna[count],2))
    return(dna)    

def generate_payoff_environment_1d(xstart_val,xsize_of_env):
    
    payoff = [-x**3+3000*x for x in range(xstart_val,xstart_val+xsize_of_env)]
    #for x in range(xstart_val,xstart_val+xsize_of_env):
    #    payoff[x]=x**2  # example function
    return(payoff)    

def generate_payoff_environment_2d(xstart_val,xsize_of_env,ystart_val,ysize_of_env):
    payoff = [[x**2+y*3 for x in range(xstart_val,xstart_val+xsize_of_env)] for y in range(ystart_val,ystart_val+ysize_of_env)]
    #for x in range(xstart_val,xstart_val+xsize_of_env):
    #    for y in range(ystart_val,ystart_val+ysize_of_env):
    #        payoff[x][y]=x**2+3*y   # example function
    return(payoff)

def calc_fitness(dna,payoff):
    max_payoff=0
    count=0
    best=""
    fitness=[]
    for elem in dna:
        val=int(dna[count],2)
        p=int(round(payoff[val]))
        fitness.append(p)
        if p>max_payoff:
            best=dna[count]
            max_payoff=p
    #    print("#",count+1,":",elem," val=",val,"payoff=",payoff[val]," fitness=",fitness[count])
        count+=1
    return(best,max_payoff)
   

def calc_mating_probabilities(dna,payoff):
    count=0
    total_payoff=1
    for elem in dna:
        val=int(dna[count],2)
        total_payoff+=payoff[val]
    #print(val,payoff[val])
        count+=1

#print(total_payoff)

    count=0
    wheel=[]
    for elem in dna:
        val=int(dna[count],2)
        wheel.append(int(round(payoff[val]/total_payoff*1000)))
    #    print("#",count+1,":",elem," val=",val,"payoff=",payoff[val]," prob=",wheel[count])
        count+=1

    
#    wheel_sort = sorted(wheel, key = lambda s: s[1])  # find the move that reduces the distance to the goal the most
#    sorted_wheel= [s[0] for s in wheel_sort]
    return(wheel)

    
def spin_the_mating_wheel(wheel,dna,iterations):
    sel=[]
    mates=[]
    n=0
    while n<=len(wheel)-1: 
        sel=sel+([n+1] * wheel[n])
        n=n+1

  #  print("\nlen(sel)=",len(sel))

    for i in range(0,iterations):
        # pick a random string for mating
        first_string_no=random.randint(1,len(wheel))
        # choose its mate from the wheel
        second_string_no=first_string_no
        while second_string_no==first_string_no:
            second_string_no=sel[random.randint(0,len(sel)-1)]
       # print("mate ",first_string_no,dna[first_string_no-1]," with ",second_string_no,dna[second_string_no-1])
        mates=mates+[(dna[first_string_no-1],dna[second_string_no-1])]                     

    #print(mates,len(mates))
    return(mates)


def crossover(mates,length):
    crossed=[]
    for i in mates:
    #    print(i[0],i[1])
        # pick a random cut point in the gene strings of the mates
        splitpoint=random.randint(1,length-1)
    #    print("splitpoint=",splitpoint)
        temp1=i[0]
        temp2=i[1]
        new1=""
        new2=""
        remain1=temp1[:splitpoint]
        swap1=temp1[splitpoint-length:]
        remain2=temp2[:splitpoint]
        swap2=temp2[splitpoint-length:]
        new1=remain1+swap2
        new2=remain2+swap1

     #   print("temp1=",temp1," remain1=",remain1," swap1",swap1)
     #   print("temp2=",temp2," remain2=",remain2," swap2",swap2)
     #   print("new1=",new1," new2=",new2)
        crossed=crossed+[(new1,new2)]
    return(crossed)


def mutate(dna,length,mutation_rate):
     mutation=False
     temp=""
     totalbits=len(dna)*length*2
 #    print("total number of bits=",totalbits)
     if totalbits==0:
         print("totalbits=0! error")
     like=int(round(mutation_rate/totalbits))
     if like<1:
         like=1
     chance=random.randint(1,like) # one in a thousand chance of a mutation to one random bit
     if chance==1: #>=1:   #==1:
     #    print("mutation occured")
         #string all the bits together
         #all=""
         #for c in crossed:
         #    all=all+c[0]+c[1]
             
        # print(dna)    
         mut_list=random.randint(0,len(dna)-1)
         mut_tup=random.randint(0,1)
         mut_bit=random.randint(0,length-1)
  #       print("mutation at:",mut_list,mut_tup,mut_bit)
         
         
         gene=str(dna[mut_list][mut_tup])   #v[mut_bit:mut_bit+1]="0"
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
                
   #      print("mutated=",temp)
            
         mutation=True 
           #  print("dna before mutation:",dna)   
 

     else:
    #     print("no mutation occured")
         mutation=False  

     new_dna=[]
     count=0
     if mutation:
         listno=mut_list*2
     else:
         listno=99999
     for c in dna:
         if count==listno and mutation:
            if mut_tup==0:
                new_dna.append(temp)
                new_dna.append(c[1])
            elif mut_tup==1:
                new_dna.append(c[0])
                new_dna.append(temp)             
            else:
                print("mutation error2")
         else:        
             new_dna.append(c[0])
             new_dna.append(c[1])
         count=count+1

         
 #    print(new_dna)


         
     return(new_dna)       




length=8   # length of dna bit strings
max_payoff=0
starting_population=16
best=""
returnedpayoff=0
gen=0
epoch_length=5000

#payoff=generate_payoff_environment_1d(0,32)  # 5 bits
#payoff=generate_payoff_environment_1d(0,64)  # 6 bits
payoff=generate_payoff_environment_1d(0,256)  #8 bits


print("Payoff Environment")
print(payoff)
input("?")
#payoff=generate_payoff_environment_2d(1,5,6,13)
#print(payoff)
generation_number=1
dna=generate_dna(starting_population,length)
print(dna)

while generation_number<=epoch_length:
    #print("Generation:",generation_number)
    fittest,returned_payoff=calc_fitness(dna,payoff)
    if returned_payoff>max_payoff:
        best=fittest
        gen=generation_number
        max_payoff=returned_payoff
        print("best fittest=",best," generation no:",gen," max_payoff=",max_payoff)
    
    wheel=calc_mating_probabilities(dna,payoff)
    #print(wheel)

    mates=spin_the_mating_wheel(wheel,dna,starting_population*2)
    crossed=crossover(mates,length)

    #print(crossed)

    dna=mutate(crossed,length,400)   # 1000 means mutation 1 in a 1000 bits processed
    generation_number+=1









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
