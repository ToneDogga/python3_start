import multiprocessing 
import re

def mp_worker(item):
    # Do something
    return item, count

def mp_handler():
    cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpus)
    # The below 2 lines populate the list. This listX will later be accessed parallely. This can be replaced as long as listX is passed on to the next step.
    with open('testfile1.csv') as f:
        listX = [line for line in (l.strip() for l in f) if line]
    with open('results.txt', 'w') as f:
        for result in p.imap(mp_worker, listX):
            # (item, count) tuples from worker
            f.write('%s: %d\n' % result)

if __name__=='__main__':
    mp_handler()
