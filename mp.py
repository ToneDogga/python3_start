


import multiprocessing as mp
import os

#import csv

#!/usr/bin/env python


def f(l):
    print(l)


pool=mp.Pool(4)
jobs=[]

with open("mytext.txt") as f:
    for line in f:
        jobs.append(pool.apply_async(f,(line,)))
        

for job in jobs:
    job.get()

    

pool.close()



"""
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3,56]))


from multiprocessing import Process

def f(name):
    print('hello', name)

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()



from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()



import multiprocessing as mp

def foo(q):
    q.put('hello')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    q = mp.Queue()
    p = mp.Process(target=foo, args=(q,))
    p.start()
    print(q.get())
    p.join()

"""



from multiprocessing import Pool, TimeoutError
import time
import os
import random

def f(x):
    return (os.getpid(),x*random.randint(1,100))

if __name__ == '__main__':
    # start 4 worker processes
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
      #  print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
       # for i in pool.imap_unordered(f, range(10)):
       #     print(i)

        # evaluate "f(20)" asynchronously
       # res = pool.apply_async(f, (20,))      # runs in *only* one process
       # print(res.get(timeout=1))             # prints "400"

        # evaluate "os.getpid()" asynchronously
       # res = pool.apply_async(os.getpid, ()) # runs in *only* one process
       # print(res.get(timeout=1))             # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
 #       multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
 #       print([res.get(timeout=1) for res in multiple_results])

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(f,(i,)) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])


       # launching multiple evaluations asynchronously *may* use more processes
      #  multiple_results = [pool.apply_async(f, (10,)) for i in range(4)]
      #  print([res.get(timeout=1) for res in multiple_results])


        # make a single worker sleep for 10 secs
      #  res = pool.apply_async(time.sleep, (10,))
      #  try:
      #      print(res.get(timeout=1))
      #  except TimeoutError:
      #      print("We lacked patience and got a multiprocessing.TimeoutError")

      #  print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")


"""
def process(l):
    print("1",l)


def process_wrapper(lineByte):
    print("pw")
    with open("mytext.txt") as f:
        f.seek(lineByte)
        line=f.readline()
        process(line)
    
# init objects
pool=mp.Pool(4)   # 4 cores in a raspberry pi
jobs=[]


# create jobs
with open("mytext.txt","r",buffering=1) as f:
    nlineByte=f.readline()
    print("nlb=",nlineByte)
    for line in f:
        jobs.append(pool.apply_async(process_wrapper,(nlineByte)))
        nlineByte=f.readlines()

def do_something_with_line(l):
    print(l)

def read_my_lines(csv_reader, lines_list):
    # make sure every line number shows up only once:
    lines_set = set(lines_list)
    for line_number, row in enumerate(csv_reader):
        if line_number in lines_set:
            yield line_number, row
            lines_set.remove(line_number)
            # Stop when the set is empty
            if not lines_set:
                raise StopIteration


def read_a_line(csv_reader, lineno):
    # make sure every line number shows up only once:
   # lines_set = set(lines_list)
    for line_number, row in enumerate(csv_reader):
        if line_number==lineno:
            yield line_number, row
            #lines_set.remove(line_number)
            # Stop when the set is empty
            #if not lines_set:
             #   raise StopIteration


#L=[0,34,13892,67]
#with open("payoff_5d.csv") as f:
#    r = csv.DictReader(f)
#    for i, line in enumerate(r):
#        if i in L:    # or (i+2) in L: from your second example
#            print(line)

#L = [2, 5, 15, 98, ...]
#L=[134554]
row=123454
with open("payoff_5d.csv","r") as f:
    f.seek(56*row)
    print(f.readline())


with open("payoff_3d.csv") as f:
    r = csv.DictReader(f)
    for line_number, line in read_my_lines(r, L):
        do_something_with_line(line)



with open("payoff_5d.csv","r",buffering=1) as f:
  line = f.readline()
  while line:
    #lineno=f.tell() #returns the location of the next line
    line = f.readline()
    print("line=",line)
    

# wait for all jobs to finish
for job in jobs:
    job.get()

#clean up
pool.close()                    
"""
