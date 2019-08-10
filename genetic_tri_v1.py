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

import sys
import hashlib, os
import random
import math
import time
import linecache
import platform
import datetime


#from multiprocessing.dummy import Pool, TimeoutError, Process, Lock, freeze_support, Queue, Lock, Manager
#from multiprocessing import TimeoutError, Pool, Process, Lock, freeze_support, Queue, Lock, Manager
from multiprocessing import Pool, Process, Lock, freeze_support, Queue, Lock, Manager

#from  multiprocessing import Pool, Queue
from os import getpid
from time import sleep
#from random import random
import random
import itertools
import platform
import math
import sys
import time
import os
import hashlib
from pathlib import Path

EOL='\n'
FILENAME="testfile1.csv"
import multiprocessing 
from timeit import default_timer as timer

fn = 'c:/temp/temp.txt'

"""
class AESCipher(object):

    def __init__(self, key): 
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        # iv is initialisation vector
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]
#!/usr/bin/env python

class Human:

    def __init__(self):
        self.name = 'Guido'
        self.head = self.Head()
        self.brain = self.Brain()
    
class Head:
    def talk(self):
        return 'talking...'

class Brain:
    def think(self):
        return 'thinking...'

if __name__ == '__main__':
    guido = Human()
    print(guido.name)
    print(guido.head.talk())
    print(guido.brain.think())



"""

#!/usr/bin/env python
# a population has a size - a number of individuals
# each individual has a chromopack, a number of chomosomes grouped together
# ploidy is the number of chromosomes in a chromopack.  2 is diploidy
# each chromosome has a number of alleles
# each allele can be -1 (recessive 1 displayed as %), 0 or 1 (dominant 1)
#

# so pop=Population(100,2,16,[])
# creates a population of 100 individuals
# with 2 chromosomes each of length 16 bits (alleles)

class Population:
    def __init__(self,population_size,ploidy,no_of_alleles):
        self.population_size=population_size
        self.ploidy=ploidy
        self.no_of_alleles=no_of_alleles
        self.individual=[]
        self.build_population()
   
    def build_population(self):
        for nameid in range(0,self.population_size):
            self.individual.append((nameid,self.add_chromopack()))  #,self.ploidy,self.alleles)
            # nameid is the uniaue number of the individual, return as a tuple (name,[list of [lists],[],[]]

  
    def add_chromopack(self):
        chromopack=[]
        for p in range(0,self.ploidy):
            chromopack.append(self.build_chromosome())
        return(chromopack)        

    def build_chromosome(self):
        dna=[]
        dna_string=""
        for locus in range(0,self.no_of_alleles):
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

    def return_chromopack(self,idnumber):
        try:   
            return(self.individual[idnumber][1])
        except IndexError:
            print("return chromopack function err.")
            print("idnumber=",idnumber," out of range of len(pop)=",len(self.individual))
            return

    def return_chromosome(self,idnumber,ploidy_no):
        try:   
            return(self.individual[idnumber][1][ploidy_no])
        except IndexError:
            print("return chromosome function err.")
            print("Ploidy=",ploidy_no," or idnumber=",idnumber," out of range of len(pop)=",len(self.individual))
            return


    def return_allele(self,idnumber,ploidy_no,locus):
        try:   
            return(self.individual[idnumber][1][ploidy_no][locus])
        except IndexError:
            print("return allele function err.")
            print("allele=",locus," Ploidy=",ploidy_no," or idnumber=",idnumber," out of range of len(pop)=",len(self.individual))
            return

        

    def return_allele_as_number(self,idnumber,ploidy_no,locus):
        try:
            a=self.individual[idnumber][1][ploidy_no][locus]
        except IndexError:
            print("return allele as number function err.")
            print("allele=",locus," Ploidy=",ploidy_no," or idnumber=",idnumber," out of range of len(pop)=",len(self.individual))
            return
            
        if a=="%":
            return(-1)
        else:
            return(int(a))



           


if __name__ == '__main__':
    if(len(sys.argv) < 2) :
        print("Usage : python3 mp_simple1.py rownumbers")
        sys.exit()

    pop_numbers = int(sys.argv[1])


    pop=Population(pop_numbers,2,16)
  #  print(pop.individual)
  #  print(pop.individual[0])
  #  print(pop.individual[0][0])
    
  #  print(pop.individual[1])
  #  print(pop.individual[0][1])

   # print(pop.individual[1])
   # print(pop.individual[1][0])
    print("pn=",pop_numbers)
    print("lp=",len(pop.individual))
    
    print("\n")
   # print(pop.individual[0])
    print(pop.individual[1])
    print("\n")
    
    print(pop.individual[1][0])
    print("\n cp=")

    print(pop.return_chromopack(1199))
    print("\n cs=")


    print(pop.return_chromosome(11,1))   # first chromosome
    
  #  print(pop.individual[0][1][1])
    print("\n all=")

    print(pop.return_allele(11,1,3))
    
  #  print(pop.individual[1][1][1][1])
    print("\n")

    print(pop.return_allele_as_number(11,1,3))
    
  #  print(pop.individual[1][1][1][1])
    print("\n")



    
    #print(pop.name)
    #print(pop.head.talk())
    #print(guido.brain.think())




       
"""

name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

# Use a compound data type for structured arrays
data = np.zeros(4, dtype={"names":("name", "age", "weight"),
                          "formats":('U10', 'i4', 'f8')})
print(data.dtype)


data["name"] = name
data["age"] = age
data["weight"] = weight
print(data)

# Get all names
print(data["name"])

# Get first row of data
print(data[0])

# Get the name from the last row
print(data[-1]["name"])

# Get names where age is under 30
print(data[data['age'] < 30]['name'])

matrix = np.zeros((3,5), dtype = int)

matrix=np.array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 2, 0]])

print(matrix)


print("data3\n")
data3=np.dtype([("name", 'S10'), ("age", 'i4'), ("weight", 'f8')])

#data3=data
data["name"] = name
#data3["age"] = age
#data3["weight"] = weight
print(data3.dtype)

# Get all names
print(data3["name"])

# Get first row of data
print(data3[0])

# Get the name from the last row
print(data3[-1]["name"])

# Get names where age is under 30
print(data3[data3['age'] < 30]['name'])



print("data4\n")
data4=np.dtype('S10,i4,f8')

data4=data
#data4["name"] = name
#data4["age"] = age
#data4["weight"] = weight
print(data4)

# Get all names
print(data4["name"])

# Get first row of data
print(data4[0])

# Get the name from the last row
print(data4[-1]["name"])

# Get names where age is under 30
print(data4[data4['age'] < 30]['name'])



Character	Description	Example
'b'	Byte	np.dtype('b')
'i'	Signed integer	np.dtype('i4') == np.int32
'u'	Unsigned integer	np.dtype('u1') == np.uint8
'f'	Floating point	np.dtype('f8') == np.int64
'c'	Complex floating point	np.dtype('c16') == np.complex128
'S', 'a'	String	np.dtype('S5')
'U'	Unicode string	np.dtype('U') == np.str_
'V'	Raw data (void)	np.dtype('V') == np.void





class Particle:
    def __init__(self, mass, position, velocity, force):
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.force = force

    @property
    def acceleration(self):
        return self.force / self.mass

particle = Particle(2, 3, 3, 8)
print(particle.acceleration)  # 4.0




class groupClass(object):
    def __init__(a, b, c):
        self.a = a
        self.b = b
        self.c = c
self.group = groupClass(1, 2, 3)
print self.group.a



class Pet:
    def __init__(self):
        pass

    def method1(self):
        print "Method 1 has been called."

    def method2(self):
        print "Method 2 has been called."

    def yelp(self):
        print "I am yelping"


class Dog(Pet):
    def __init__(self):
        Pet.__init__(self)

    def yelp(self):
        print "I am barking"


class Cat(Pet):
    def __init__(self):
        Pet.__init__(self)

    def yelp(self):
        print "I am meowing"


class PetFactory:
    def __init__(self):
        pass

    def acquire_dog(self):
        return Dog()

    def acquire_cat(self):
        return Cat()

    def acquire_pet_by_name(self, pet_type):
        if pet_type == "dog":
            return Dog()
        elif pet_type == "cat":
            return Cat()

>>> pet = Pet()
>>> dog = Dog()
>>> cat = Cat()
>>> dog.yelp()
I am barking
>>> cat.yelp()
I am meowing
>>> pet.yelp()
I am yelping
>>> pet_factory = PetFactory()
>>> pet_factory.acquire_cat().yelp()
I am meowing
>>> pet_factory.acquire_pet_by_name("cat").yelp()
I am meowing
>>> cat.method1()
Method 1 has been called.
>>> dog.method2()
Method 2 has been called.










class Dog:

    def __init__(self, name):
        self.name = name
        self.tricks = []    # creates a new empty list for each dog

    def add_trick(self, trick):
        self.tricks.append(trick)

>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.add_trick('roll over')
>>> e.add_trick('play dead')
>>> d.tricks
['roll over']
>>> e.tricks
['play dead']
class Individual:
    def __init__(self,s):
        self.id=s
        self.chromopack = self.Chromopack()

class Chromopack:
    def __init__(self):
        for p in range(0,Population.ploidy):
            self.chromosome[p]=self.Chromosome()

class Chromosome:
    def __init__(self):
        self.dna=[]
        for a in range(0,Population.alleles):
            if random.randint(0,1)==0:
                self.dna[a].append("0")
            else:
                if random.randint(0,1)==1:
                    # dominant 1
                    self.dna[a].append("1")
                else:
                    # recessive 1
                    self.dna[a].append("%")


class Head:
    def talk(self):
        return 'talking...'

class Brain:
    def think(self):
        return 'thinking...'
"""
        

        
        




        
        





"""
class AESCipher(object):

    def __init__(self, key): 
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        # iv is initialisation vector
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]








def listener(q):    #,l_lock):
    '''listens for messages on the q, writes to file. '''
    print("Queue listener started on:",os.getpid())
   # l_lock.acquire()
    f=open(FILENAME,"w")   #,buffering=8096)
    #f.flush()
    while True:
       # if q.full():
       #     print("q full")
       #     break
        m = q.get()    #timeout=20)  get_nowait
        if m == "kill":
            print("trying to kill listener process.  Flushing q.  qsize=",q.qsize())
            #f.write("killed\n")
            while not q.empty():
                m=q.get()
                f.write(m)            
                f.flush()
            break
        else: 
            f.write(m)  #+":"+q.qsize()+"\n")
          #  f.write(str(q.qsize())+'\n')
            f.flush()
       
  #  q.close()
    f.close()
    print("Queue listener closed on:",os.getpid())
   # l_lock.release()
    return(True)

def add_to_queue(msg,q):
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

def generate_payoff_environment_1d_mp(astart_val,asize_of_env,q,linewrite,lwremainder):    #l_lock  
    rowno=0
    print("payoff calc function started:",os.getpid())

    clock_start=timer()  #.process_time()

    pline=""
    for a in range(astart_val,astart_val+asize_of_env):
        payoff=-10*math.sin(a/44)*12*math.cos(a/33)*1000/(a+1)
        pline=pline+(str(rowno)+","+str(a)+","+str(payoff)+"\n")
        if rowno%linewrite==0:
            add_to_queue(pline,q)
            pline=""     
        rowno+=1
        
    if pline:
        add_to_queue(pline,q)

 
    clock_end=timer()  #.process_time()
    duration_clock=clock_end-clock_start

    print("payoff pid:[",os.getpid(),"] finished chunk",rowno,"rows in:",duration_clock,"secs.")
    return   #(payoff)    #(os.getpid(),payoff)

def generate_payoff_environment_1d(astart_val,asize_of_env):    #l_lock  
    rowno=0
    print("payoff calc function started:",os.getpid())

    #clock_start=time.process_time()

  #  l_lock.acquire()

    f=open("2"+FILENAME,"w")   #,buffering=0)
    #f.flush()
    for a in range(astart_val,astart_val+asize_of_env):
        payoff=-10*math.sin(a/44)*12*math.cos(a/33)*1000/(a+1)
        pline=(str(rowno)+","+str(a)+","+str(payoff)+"\n")
        f.write(pline)
      #  add_to_queue(pline,q)
      #  q.put(pline)
      #  if q.full():
      #      print("q full")
        rowno+=1
    f.flush()
    f.close()

 #   l_lock.release()
    #clock_end=time.process_time()
    #duration_clock=clock_end-clock_start

    #print("payoff single process pid:[",os.getpid(),"] finished chunk",rowno,"rows in:",duration_clock,"secs.")
    return   #(payoff)    #(os.getpid(),payoff)


        
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



# Warning from Python documentation:
# Functionality within this package requires that the __main__ module be
# importable by the children. This means that some examples, such as the
# multiprocessing.Pool examples will not work in the interactive interpreter.

#if __name__ == '__main__':

def main():
    freeze_support()

    if(len(sys.argv) < 2) :
        print("Usage : python3 mp_simple1.py rownumbers")
        sys.exit()

    row_numbers = int(sys.argv[1])

    if len(sys.argv)<3:
        linewrite=10000
    else:    
        linewrite=int(sys.argv[2])
        if linewrite<=0:
            linewrite=10000
    
        
    manager = multiprocessing.Manager()
    q = manager.Queue()    
   # pool = mp.Pool(mp.cpu_count() + 2)


    cpus = multiprocessing.cpu_count()
    #put listener to work first
   # watcher = pool.apply_async(listener, (q,))
   # with Pool(processes=cpus) as pool:  # processes=cpus
   #     watcher = pool.apply_async(listener, args=(q, ))


    windows=platform.system().lower()[:7]
    print("platform=",windows)
    if windows=="windows":
        EOL="\r\n"
    else:
        EOL='\n'

 #   cpus = cpu_count()
    print("cpus=",cpus)
   # cpus=cpus-2    # allow one for existing processes and one for watcher

 #   my_file = Path("/home/pi/Python_Lego_projects/testfile1.csv")
  #  my_file = Path("/home/pi/Python_Lego_projects/testfile1.csv")

    #my_file=FILENAME  #basename(my_file)
  #  if my_file.is_file():
  #      print(FILENAME," #=",hash_a_file(FILENAME))
   #     os.remove(FILENAME)
  #  f=open(FILENAME,"w")
  #  f.close()
  #  f=open("2"+FILENAME,"w")
  #  f.close()

    #    print(FILENAME,":removed")    
  
    #clock_start=time.process_time()

   # with Pool(processes=1) as pool2:  # processes=cpus-1
   #     watcher = pool2.apply_async(listener, args=(q, ))

    clock_start=timer() #time.process_time()
    #lock=Lock()

    chunk=int(row_numbers/(cpus-1))   # less one cpu for listener
    remainder=row_numbers%(cpus-1)

    #linewrite=int(row_numbers/lineschunk)
    lwremainder=row_numbers%linewrite
    
    print("rownumbers=",row_numbers," chunk size=",chunk," remainder=",remainder)
    print("lines per write=",linewrite," remainder=",lwremainder)

    with Pool(processes=cpus) as pool:  # processes=cpus-1

      #  watcher = pool.apply_async(listener, args=(q, ))
        watcher = pool.apply_async(listener, args=(q, ))


        if remainder!=0:
            multiple_results = [pool.apply_async(generate_payoff_environment_1d_mp,args=(0,remainder,q,linewrite,lwremainder ))]
        else:
            multiple_results=[]
        for i in range(0,cpus-1):
            multiple_results.append(pool.apply_async(generate_payoff_environment_1d_mp,args=(i*chunk+remainder,chunk,q,linewrite,lwremainder )))

       # print("mr=",len(multiple_results))
        for res in multiple_results:
            result=res.get(timeout=None)       
            res.wait()

       # sleep(5)
        print("Generate payoff results finished")

        print("killing listener")
        q.put("kill")
        result=watcher.get(timeout=None) 
        watcher.wait()

        
    # Waits a bit for the child processes to do some work
    # because when the parent exits, childs are terminated.
 
   
    
   # pool.close()
   # pool.join()




   # clock_end=time.process_time()
   # duration_clock=clock_end-clock_start
   # print("gen payoff rows=",row_numbers," Time:", duration_clock," secs.")



    #sleep(5)
   
   # sleep(1)
 
    #result=watcher.get(timeout=5)
   # pool2.close()
  #  result=watcher.get(timeout=40) 
   # sleep(2)
    print("try to close pool")
    pool.close()
    print("pool closed.  trying to join() pool")
    pool.join()
    print("pool join() complete")
   # sleep(1)


    clock_end=timer()   #time.process_time()
    duration_clock=clock_end-clock_start
    print("gen payoff file windows multi processor rows=",row_numbers," Time:", duration_clock," secs.")

   
    print("count file rows=",count_file_rows(FILENAME))
    print(FILENAME," #=",hash_a_file(FILENAME))      



    print("\n\n Benchmarking against single processor")
    clock_start=timer()  #.process_time()


    generate_payoff_environment_1d(0,row_numbers)
   

    clock_end=timer()   #.process_time()
    duration_clock=clock_end-clock_start
    print("gen payoff single processor rows=",row_numbers," Time:", duration_clock," secs.")



if __name__ == '__main__':
    main()


"""
