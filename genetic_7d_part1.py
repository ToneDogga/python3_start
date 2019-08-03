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
import datetime

global payoff_filename,number_of_cols, total_rows, linewidth,extra_EOL_char,row_find_method
payoff_filename="shopsales3.csv"
number_of_cols=9   # rowno, 6 input vars and 1 out =8
linewidth=76   # 66 bytes
extra_EOL_char=0
row_find_method="l"

# csv
#      row number, x, y, z, payoff(or cost)


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
                    payoff=100*math.sin(x/44)*120*math.cos(y/33)-193*math.tan(z/55)
                    f.write(str(rowno)+","+str(x)+","+str(y)+","+str(z)+","+str(payoff)+"\n")
                    rowno+=1        
    f.close()
    print("")
    return(rowno)


def generate_payoff_environment_5d_file(linewidth,astart_val,asize_of_env,bstart_val,bsize_of_env,cstart_val,csize_of_env,dstart_val,dsize_of_env,estart_val,esize_of_env,filename):   
    rowno=0
    
    total_rows=asize_of_env*bsize_of_env*csize_of_env*dsize_of_env*esize_of_env    
    with open(filename,"w") as f:
        for a in range(astart_val,astart_val+asize_of_env):
            print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
            for b in range(bstart_val,bstart_val+bsize_of_env):
                for c in range(cstart_val,cstart_val+csize_of_env):
                    for d in range(dstart_val,dstart_val+dsize_of_env):
                        for e in range(estart_val,estart_val+esize_of_env):
                            payoff=100*math.sin(a/44)*120*math.cos(b/33)-193*math.tan(c/55)+78*math.sin(d/11)-98*math.cos(e/17)
#                            f.write(str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(payoff)+"\n")
                            w=str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(payoff)
                            padding=linewidth-len(w)-1
                            w=w+" "*padding
                            rowno+=1
                            f.write(w+"\n")
    f.close()
    print("")
    return(rowno)


def generate_payoff_environment_6d_file(linewidth,astart_val,asize_of_env,bstart_val,bsize_of_env,cstart_val,csize_of_env,dstart_val,dsize_of_env,estart_val,esize_of_env,fstart_val,fsize_of_env,filename):   
    rowno=0
    
    total_rows=asize_of_env*bsize_of_env*csize_of_env*dsize_of_env*esize_of_env*fsize_of_env    
    with open(filename,"w") as filein:
        for a in range(astart_val,astart_val+asize_of_env):
            print("\rProgress: [%d%%] " % (rowno/total_rows*100),end='\r', flush=True)
            for b in range(bstart_val,bstart_val+bsize_of_env):
                for c in range(cstart_val,cstart_val+csize_of_env):
                    for d in range(dstart_val,dstart_val+dsize_of_env):
                        for e in range(estart_val,estart_val+esize_of_env):
                            for f in range(fstart_val,fstart_val+fsize_of_env):
                                payoff=100*math.sin(a/44)*120*math.cos(b/33)-193*math.tan(c/55)+78*math.sin(d/11)-98*math.cos(e/17)+f
#                               filein.write(str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(f)+","+str(payoff)+"\n")
                                w=str(rowno)+","+str(a)+","+str(b)+","+str(c)+","+str(d)+","+str(e)+","+str(f)+","+str(payoff)
                                padding=linewidth-len(w)-1
                                w=w+" "*padding
                                rowno+=1
                                filein.write(w+"\n")
    filein.close()
    print("")
    return(rowno)



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

######################################################


class gene_string(object):

    def __init__(self,length,starting_population): 
       # self.length=16   # 16 length of dna bit strings
     #   self.max_payoff=0
     #   self.min_payoff=0
       # self.starting_population=256   #256 (16 bits), 64 (13 bits), 32 (10 bits)   #16 for 8 bits #4 for 5 bits
    #    best=""
      #  self.payoff_filename="/home/pi/Python_Lego_projects/payoff_1d.csv"
         self.pconstrain=False
         self.minp=1000000
         self.maxp=-1000000
        # self.targetpayoff=0
         self.aconstrain=False
         self.mina=1000000
         self.maxa=-1000000
         self.bconstrain=False
         self.minb=1000000
         self.maxb=-1000000
         self.cconstrain=False
         self.minc=1000000
         self.maxc=-1000000
         self.dconstrain=False
         self.mind=1000000
         self.maxd=-1000000
         self.econstrain=False
         self.mine=1000000
         self.maxe=-1000000
         self.fconstrain=False
         self.minf=1000000
         self.maxf=-1000000
         self.gconstrain=False
         self.ming=1000000
         self.maxg=-1000000

         
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
       #     print("\nvalue error row+1=",row+1) 
            return("value error")
        except IOError:
            print("\nIO error")
            return("IO error")

        
    def return_a_row2(self,row,filename):
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

      


    def find_a_payoff(self,row,filename):   # assumes the payoff is the last field in a CSV delimited by ","
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

    def find_a_payoff2(self,row,filename):
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



    def find_a_row_and_column(self,row,col,filename):   # assumes the payoff is the last field in a CSV delimited by ","
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
        

    def find_a_row_and_column2(self,row,col,filename):
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


    def return_a_row_as_a_list(self,row,filename):
        # we know that the each line in the file is exactly global "linewidth" bytes long including the '\n'
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


    def return_a_row_as_a_list2(self,row,filename):
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





    def calc_fitness(self,dna,direction):
        max_payoff=-10000000.0
        min_payoff=10000000.0
        p=0.0
        plist=[]
       # element_list=[]
        elements_count=0
        averagep=0
        totalp=0
        count=0
        best=""
        bestrow=0
        found=False   # found flag is a True if dna is found and a bestrow returned
     #   print("self vars",self.pconstraint,self.maxp,self.minp)
      #  fitness=[]
        for elem in dna:
            plist=[]
            val=int(dna[count],2)
            if val <= total_rows:
                try:
                    if row_find_method=="l":  # linecache
                      #  p=self.find_a_payoff(val,payoff_filename)
                        plist=self.return_a_row_as_a_list(val,payoff_filename)
                        
                    elif row_find_method=="s":    
 #                       p=self.find_a_payoff2(val,payoff_filename)
                        plist=self.return_a_row_as_a_list2(val,payoff_filename)

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
                    #print(plist)
                    row_number=plist[0]
                    a=int(plist[1])
                    b=int(plist[2])
                    c=int(plist[3])
                    d=int(plist[4])
                    e=int(plist[5])
                    f=int(plist[6])
                    g=int(plist[7])
                    p=float(plist[8].rstrip())
    
                    #rowlist=[]

                     #  element_list.append((row_number,p))

                    if not (self.pconstrain or self.aconstrain or self.bconstrain or self.cconstrain or self.dconstrain or self.econstrain or self.fconstrain or self.gconstrain):
                        pass   # here if no constriants
                    else:
                                if (self.pconstrain and p<=self.maxp and p>=self.minp):
                            
                                    totalp=totalp+p
                                    elements_count+=1
                        
                                elif (self.aconstrain and a<=self.maxa and a>=self.mina):
                                        
                                        totalp=totalp+p
                                        elements_count+=1                                
                                    
                                elif (self.bconstrain and b<=self.maxb and b>=self.minb):
                                                
                                            totalp=totalp+p
                                            elements_count+=1
                                        
                                elif (self.cconstrain and c<=self.maxc and c>=self.minc):
                                              
                                                totalp=totalp+p
                                                elements_count+=1
                        
                                elif (self.dconstrain and d<=self.maxd and d>=self.mind):
                                                    
                                                    totalp=totalp+p
                                                    elements_count+=1
                                                
                                elif (self.econstrain and e<=self.maxe and e>=self.mine):
                                                        
                                                        totalp=totalp+p
                                                        elements_count+=1
                                                    
                                elif (self.fconstrain and f<=self.maxf and f>=self.minf):
                                                            
                                                            totalp=totalp+p
                                                            elements_count+=1
                                                
                                elif (self.gconstrain and g<=self.maxg and g>=self.ming):
                                                                
                                                                totalp=totalp+p
                                                                elements_count+=1
                                    
                          


                   
                        # fitness is the highest payoff
                   # print("\np=",p," val=",val)
                    if direction=="x":  # maximising payoff
                        if p>max_payoff:
                                if not (self.pconstrain or self.aconstrain or self.bconstrain or self.cconstrain or self.dconstrain or self.econstrain or self.fconstrain or self.gconstrain):
                                    best=dna[count]
                                    bestrow=row_number   # if not constrained at all
                                    max_payoff=p
                                    found=True
                               
                                if (self.pconstrain and p<=self.maxp and p>=self.minp):
                                    best=dna[count]
                                #    totalp=totalp+p
                                #    elements_count+=1
                                    bestrow=row_number     
                                    max_payoff=p
                                    found=True
                                elif (self.aconstrain and a<=self.maxa and a>=self.mina):
                                        best=dna[count]
                                 #       totalp=totalp+p
                                 #       elements_count+=1                                
                                        bestrow=row_number     
                                        max_payoff=p
                                        found=True
                                elif (self.bconstrain and b<=self.maxb and b>=self.minb):
                                            best=dna[count]
                                            bestrow=row_number     
                                  #          totalp=totalp+p
                                  #          elements_count+=1
                                            max_payoff=p
                                            found=True
                                elif (self.cconstrain and c<=self.maxc and c>=self.minc):
                                                best=dna[count]
                                                bestrow=row_number    
                                    #            totalp=totalp+p
                                   #             elements_count+=1
                                                max_payoff=p
                                                found=True
                                elif (self.dconstrain and d<=self.maxd and d>=self.mind):
                                                    best=dna[count]
                                                    bestrow=row_number
                                    #                totalp=totalp+p
                                     #               elements_count+=1
                                                    max_payoff=p
                                                    found=True
                                elif (self.econstrain and e<=self.maxe and e>=self.mine):
                                                        best=dna[count]
                                                        bestrow=row_number
                                      #                  totalp=totalp+p
                                      #                  elements_count+=1
                                                        max_payoff=p
                                                        found=True
                                elif (self.fconstrain and f<=self.maxf and f>=self.minf):
                                                            best=dna[count]
                                                            bestrow=row_number
                                       #                     totalp=totalp+p
                                       #                     elements_count+=1
                                                            max_payoff=p
                                                            found=True
                                elif (self.gconstrain and g<=self.maxg and g>=self.ming):
                                                                best=dna[count]
                                                                bestrow=row_number
                                        #                        totalp=totalp+p
                                        #                        elements_count+=1
                                                                max_payoff=p
                                                                found=True
                                                       

                      
                                
                    elif direction=="n":    # minimising cost
                        if p<min_payoff:
                             #if ((self.pconstrain and p<self.minp) or (self.aconstrain and a<self.mina) or (self.bconstrain and b<self.minb) or (self.cconstrain and c<self.minc) or (self.dconstrain and d<self.mind) or (self.econstrain and e<self.mine)):
                                # ignore the result
                               # print("\nignore 1 min")
                             #   pass
                             #else:
                                if not (self.pconstrain or self.aconstrain or self.bconstrain or self.cconstrain or self.dconstrain or self.econstrain or self.fconstrain or self.gconstrain):
                                    best=dna[count]
                                    bestrow=row_number   
                                    min_payoff=p
                                    found=True
                
                                if (self.pconstrain and p<=self.maxp and p>=self.minp):
                                    best=dna[count]
                                    bestrow=row_number
                                   # totalp=totalp+p
                                   # elements_count+=1
                                    min_payoff=p
                                    found=True
                                elif (self.aconstrain and a<=self.maxa and a>=self.mina):
                                        best=dna[count]
                                        bestrow=row_number
                                   #     totalp=totalp+p
                                   #     elements_count+=1
                                        min_payoff=p
                                        found=True
                                elif (self.bconstrain and b<=self.maxb and b>=self.minb):
                                            best=dna[count]
                                            bestrow=row_number
                                   #         totalp=totalp+p
                                   #         elements_count+=1
                                            min_payoff=p
                                            found=True
                                elif (self.cconstrain and c<=self.maxc and c>=self.minc):
                                                best=dna[count]
                                                bestrow=row_number
                                     #           totalp=totalp+p
                                     #           elements_count+=1
                                                min_payoff=p
                                                found=True
                                elif (self.dconstrain and d<=self.maxd and d>=self.mind):
                                                    best=dna[count]
                                                    bestrow=row_number
                                     #               totalp=totalp+p
                                     #               elements_count+=1
                                                    min_payoff=p
                                                    found=True
                                elif (self.econstrain and e<=self.maxe and e>=self.mine):
                                                        best=dna[count]
                                                        bestrow=row_number
                                     #                   totalp=totalp+p
                                     #                   elements_count+=1
                                                        min_payoff=p
                                                        found=True
                                elif (self.fconstrain and f<=self.maxf and f>=self.minf):
                                                            best=dna[count]
                                                            bestrow=row_number
                                      #                      totalp=totalp+p
                                      #                      elements_count+=1
                                                            min_payoff=p
                                                            found=True
                                elif (self.gconstrain and g<=self.maxg and g>=self.ming):
                                                                best=dna[count]
                                                                bestrow=row_number
                                       #                         totalp=totalp+p
                                       #                         elements_count+=1
                                                                min_payoff=p
                                                                found=True
                                                            


  

                    else:
                        print("\ncalc fitness direction error.")
                else:
                    pass     # no value present in payoff_file for that binary value in dna
                  #  found=False
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
        if direction=="x":
            return(found,bestrow,best,elements_count,averagep,max_payoff)
        elif direction=="n":
            return(found,bestrow,best,elements_count,averagep,min_payoff)
        else:
            print("direction error..")
            return(False,0,0,0,0)


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
            if row_find_method=="l":
                total_payoff+=self.find_a_payoff(val,payoff_filename)
            elif row_find_method=="s": 
                total_payoff+=self.find_a_payoff2(val,payoff_filename)
            else:
                print("row find method error.")
                sys.exit()
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
     #   print("\ntotal payoff=",total_payoff)
        for elem in dna:
            val=int(dna[count],2)
            if row_find_method=="l":
                p=self.find_a_payoff(val,payoff_filename)
            elif row_find_method=="s":   
                p=self.find_a_payoff2(val,payoff_filename)
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








    def genetic_algorithm(self):

        length=15   #16   # 16 length of dna bit strings
        max_payoff=-10000
        min_cost=10000
        starting_population=600   #256   #256 (16 bits), 64 (13 bits), 32 (10 bits)   #16 for 8 bits #4 for 5 bits
        best=""
        returnedpayoff=0
        gen=0
        epoch_length=100
        extinction_events=2
        scaling_factor=10000  # scaling figure is the last.  this multiplies the payoff up so that diversity is not lost on the wheel when probs are rounded
        extinctions=0
        mutation_count=0
        mutations=0
        mutation_rate=500   # mutate 1 bit in every 1000.  but the mutation is random 0 or 1 so we need to double the try to mutate rate




        outfile.write("\n\nGenetic algorithm. By Anthony Paech\n")
        outfile.write("===================================\n")
        outfile.write(str(datetime.datetime.now())+"\n")
        outfile.write("Platform: "+platform.machine()+"\n:"+platform.platform()+"\n\n")
        outfile.write("\nPayoff/cost environment file is: "+payoff_filename+" and has "+str(total_rows)+" rows.\n\n")

        outfile.write("Bit Length= "+str(length)+" -> Max CSV datafile rows available is: "+str(2**length)+"\n")
        outfile.write("\n\n")
        print("\n")
        if self.pconstrain:
            outfile.write(str(self.minp)+" <= payoff/cost <= "+str(self.maxp)+"\n")
            print(str(self.minp)+" <= payoff/cost <= "+str(self.maxp)+"\n")

        if self.aconstrain:
            outfile.write(str(self.mina)+" <= a <= "+str(self.maxa)+"\n")
            print(str(self.mina)+" <= a <= "+str(self.maxa)+"\n")

        if self.bconstrain:
            outfile.write(str(self.minb)+" <= b <= "+str(self.maxb)+"\n")
            print(str(self.minb)+" <= b <= "+str(self.maxb)+"\n")
            
        if xgene.cconstrain:
            outfile.write(str(self.minc)+" <= c <= "+str(self.maxc)+"\n")
            print(str(self.minc)+" <= c <= "+str(self.maxc)+"\n")
            
        if self.dconstrain:
            outfile.write(str(self.mind)+" <= d <= "+str(self.maxd)+"\n")
            print(str(self.mind)+" <= d <= "+str(self.maxd)+"\n")

        if self.econstrain:
            outfile.write(str(self.mine)+" <= e <= "+str(self.maxe)+"\n")
            print(str(self.mine)+" <= e <= "+str(self.maxe)+"\n")

        if self.fconstrain:
            outfile.write(str(self.minf)+" <= f <= "+str(self.maxf)+"\n")
            print(str(self.minf)+" <= f <= "+str(self.maxf)+"\n")

        if self.gconstrain:
            outfile.write(str(self.ming)+" <= g <= "+str(self.maxg)+"\n\n\n")
            print(str(self.ming)+" <= g <= "+str(self.maxg)+"\n\n\n")


            
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



     
        clock_start=time.process_time()


        #payoff=generate_payoff_environment_2d(1,5,6,13)
        #print(payoff)
        while extinctions<=extinction_events:

     
            generation_number=1
            print("Generating",starting_population,"unique elements of random DNA of length:",length,". Please wait....")
            outfile.write("Generating "+str(starting_population)+" unique elements of random DNA of length: "+str(length)+". Please wait....\n")

            dna=self.generate_dna(length,starting_population)
            len_sel=len(dna)
            #print(dna)
            total_genes=0
            gene_pool_size=0
            mutations=0
            mutation_count=0
    
            print("Number of epochs:",extinctions+1,"Running for",epoch_length,"generations.")
            outfile.write("Number of epochs: "+str(extinctions+1)+" Running for  "+str(epoch_length)+" generations.\n")

            while generation_number<=epoch_length:
                print("Epochs:",extinctions+1,"Generation progress: [%d%%]" % (generation_number/epoch_length*100) ,"diversity (len_sel)=",len_sel,"Tot gene bits:%d" % total_genes,"Tot Mutations:%d    "  % (mutations), end='\r', flush=True)

                 #   clock_start=time.process_time()

                found,bestrow,fittest,elements,averagep,returned_payoff=self.calc_fitness(dna,direction)
                 #   print("fittest=",fittest," returned=",returned_payoff)
                 #   clock_end=time.process_time()
                 #   duration_clock=clock_end-clock_start
                 #   print("calc fitness - Clock: duration_clock =", duration_clock)

                    #jump=xgene.pconstraint and (returned_payoff<xgene.minp or returned_payoff>xgene.maxp)   # if the payoff is constrainted
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
                                       axis[col]=self.return_a_row(int(best,2),payoff_filename).split(",")[col]
                                    elif row_find_method=="s":   
                                       axis[col]=self.return_a_row2(int(best,2),payoff_filename).split(",")[col]
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
                                        axis[col]=self.return_a_row(int(best,2),payoff_filename).split(",")[col]
                                    elif row_find_method=="s":
                                        axis[col]=self.return_a_row2(int(best,2),payoff_filename).split(",")[col]
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
   
                wheel=self.calc_mating_probabilities(dna,direction,scaling_factor)  # scaling figure is the last.  this multiplies the payoff up so that divsity is not lost on the wheel when probs are rounded
                if len(wheel)==0:
                    print("wheel empty")
                    sys.exit
            #print(wheel)


                mates,len_sel=self.spin_the_mating_wheel(wheel,dna,starting_population)  # sel_len is the size of the uniqeu gene pool to select from in the wheel

         
                crossed=self.crossover(mates,length)


                dna,mutation_count,gene_pool_size=self.mutate(dna,crossed,length,mutation_rate)   # 1000 means mutation 1 in a 1000 bits processed

 
        
                total_genes=total_genes+gene_pool_size
                mutations=mutations+mutation_count
                generation_number+=1

            extinctions+=1
            starting_population=int(round(starting_population/2))    #  increase the starting population between the different epochs to test the results
            clock_end=time.process_time()
            duration_clock=clock_end-clock_start
            print("\n\nFinished - Clock: duration_clock =", duration_clock)
            outfile.write("\nFinished - Clock: duration_clock ="+str(duration_clock)+"\n")
          
            print("")







###############################################





length=15   #16   # 16 length of dna bit strings
max_payoff=-10000
min_cost=10000
starting_population=500   #256   #256 (16 bits), 64 (13 bits), 32 (10 bits)   #16 for 8 bits #4 for 5 bits
best=""
returnedpayoff=0
gen=0
epoch_length=100
extinction_events=2
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
print("Platform:",platform.machine(),"\n:",platform.platform())
#print("\n:",platform.processor(),"\n:",platform.version(),"\n:",platform.uname())
print("Bit Length=",length,"-> Max CSV datafile rows available is:",2**length)
print("\nTheoretical max no of rows for the CSV datafile file:",payoff_filename,"is:",sys.maxsize)

#print(platform.system().lower()[:7])   #=="windows":

if platform.system().lower()[:7]=="windows":
    extra_EOL_char=1
else:
    extra_EOL_char=0

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
print("\nPayoff/cost environment file is:",payoff_filename,"and has",total_rows,"rows.")

row_find_method="l"
#while row_find_method!="l" and row_find_method!="s":
#    row_find_method=input("\nUse (l)ine cache {fast but memory intensive.  Good for bits<=23} or \n(s)eek {slow but memory frugal. Good for bits >=24}?")
    
"""
answer=""
while answer!="y" and answer!="n":
    answer=input("Create payoff env? (y/n)")
if answer=="y":
    print("Creating payoff/cost environment....file:",payoff_filename)
    clock_start=time.process_time()

    total_rows=generate_payoff_environment_7d_file(linewidth,0,2**1,0,2**1,0,2**1,0,2**1,0,2**1,0,2**6,0,2**4,payoff_filename)  
    clock_end=time.process_time()
    duration_clock=clock_end-clock_start
    print("generate payoff/cost environment - Clock: duration_clock =", duration_clock,"seconds.")
    print("Payoff/cost environment file is:",payoff_filename,"and has",total_rows,"rows.")
"""
    
outfile=open("outfile.txt","a")


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
    xgene.maxp=0
    xgene.pconstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain payoff/cost? (y/n)")
        if con=="y":
            xgene.maxp=int(input("Maximum payoff/cost?"))
            xgene.minp=xgene.maxp+1
            while xgene.minp>xgene.maxp:
                xgene.minp=int(input("Minimum payoff/cost?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.minp,"<= payoff/cost <=",xgene.maxp)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.pconstrain=True
        else:
            correct="y"    

#################################### constrain a?

    con=""
    xgene.maxa=0
    xgene.aconstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain column a? (y/n)")
        if con=="y":
            xgene.maxa=int(input("Maximum a?"))
            xgene.mina=xgene.maxa+1
            while xgene.mina>xgene.maxa:
                xgene.mina=int(input("Minimum a?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.mina,"<= a <=",xgene.maxa)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.aconstrain=True
        else:
            correct="y"    

######################################  constrain b?


    con=""    
    xgene.maxb=0
    xgene.bconstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain column b? (y/n)")
        if con=="y":
            xgene.maxb=int(input("Maximum b?"))
            xgene.minb=xgene.maxb+1
            while xgene.minb>xgene.maxb:
                xgene.minb=int(input("Minimum b?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.minb,"<= b <=",xgene.maxb)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.bconstrain=True
        else:
            correct="y"


###########################################  constrain c?
        
    con=""
    xgene.maxc=0
    xgene.cconstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain column c? (y/n)")
        if con=="y":
            xgene.maxc=int(input("Maximum c?"))
            xgene.minc=xgene.maxc+1
            while xgene.minc>xgene.maxc:
                xgene.minc=int(input("Minimum c?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.minc,"<= c <=",xgene.maxc)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.cconstrain=True
        else:
            correct="y"    



##############################################  constrain d?
    con=""
    xgene.maxd=0
    xgene.dconstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain column d? (y/n)")
        if con=="y":
            xgene.maxd=int(input("Maximum d?"))
            xgene.mind=xgene.maxd+1
            while xgene.mind>xgene.maxd:
                xgene.mind=int(input("Minimum d?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.mind,"<= d <=",xgene.maxd)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.dconstrain=True
        else:
            correct="y"    


##############################################  constrain e?
    con=""
    xgene.maxe=0
    xgene.econstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain column e? (y/n)")
        if con=="y":
            xgene.maxe=int(input("Maximum e?"))
            xgene.mine=xgene.maxe+1
            while xgene.mine>xgene.maxe:
                xgene.mine=int(input("Minimum e?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.mine,"<= e <=",xgene.maxe)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.econstrain=True
        else:
            correct="y"


##############################################  constrain f?
    con=""
    xgene.maxf=0
    xgene.fconstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain column f? (y/n)")
        if con=="y":
            xgene.maxf=int(input("Maximum f?"))
            xgene.minf=xgene.maxf+1
            while xgene.minf>xgene.maxf:
                xgene.minf=int(input("Minimum f?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.minf,"<= f <=",xgene.maxf)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.fconstrain=True
        else:
            correct="y"


##############################################  constrain g?
    con=""
    xgene.maxg=0
    xgene.gconstrain=False
    correct="n"
    while correct=="n":
        while con!="y" and con!="n":
            con=input("Constrain column g? (y/n)")
        if con=="y":
            xgene.maxg=int(input("Maximum g?"))
            xgene.ming=xgene.maxg+1
            while xgene.ming>xgene.maxg:
                xgene.ming=int(input("Minimum g?"))
            correct=""
            while correct!="y" and correct!="n":
                print(xgene.ming,"<= g <=",xgene.maxg)
                correct=input("Correct? (y/n)")
            if correct=="y":
                xgene.gconstrain=True
        else:
            correct="y"


###############################################################



    xgene.genetic_algorithm()

             

########################################################################


#    def genetic_algorithm(max_or_min,pconstrain,minp,maxp,\
#                          aconstrain,mina,maxa,\
#                          bconstrain,minb,maxb,\
#                          cconstrain,minc,maxc,\
#                          dconstrain,mind,maxd,\
#                          econstrain,mine,maxe,\
#                          fconstrain,minf,maxf
            


else:
    for a in range(0,2**1):
        direction="x"
        xgene.pconstrain=False
        xgene.minp=0
        xgene.maxp=0
        xgene.aconstrain=True
        xgene.mina=a
        xgene.maxa=a
        xgene.bconstrain=False
        xgene.minb=0
        xgene.maxb=0
        xgene.cconstrain=False
        xgene.minc=0
        xgene.maxc=0
        xgene.dconstrain=False
        xgene.mind=0
        xgene.maxd=0
        xgene.econstrain=False
        xgene.mine=0
        xgene.maxe=0
        xgene.fconstrain=False
        xgene.minf=0
        xgene.maxf=0
        xgene.gconstrain=False
        xgene.ming=0
        xgene.maxg=0




        # maximise results written to outfile.txt
        xgene.genetic_algorithm()
        direction="n"
        #minimise results written to outfile.txt
        xgene.genetic_algorithm()
#

    for b in range(0,2**1):
        direction="x"
        xgene.pconstrain=False
        xgene.minp=0
        xgene.maxp=0
        xgene.aconstrain=False
        xgene.mina=0
        xgene.maxa=0
        xgene.bconstrain=True
        xgene.minb=b
        xgene.maxb=b
        xgene.cconstrain=False
        xgene.minc=0
        xgene.maxc=0
        xgene.dconstrain=False
        xgene.mind=0
        xgene.maxd=0
        xgene.econstrain=False
        xgene.mine=0
        xgene.maxe=0
        xgene.fconstrain=False
        xgene.minf=0
        xgene.maxf=0
        xgene.gconstrain=False
        xgene.ming=0
        xgene.maxg=0




        # maximise results written to outfile.txt
        xgene.genetic_algorithm()
        direction="n"
        #minimise results written to outfile.txt
        xgene.genetic_algorithm()


    for c in range(0,2**1):
        direction="x"
        xgene.pconstrain=False
        xgene.minp=0
        xgene.maxp=0
        xgene.aconstrain=False
        xgene.mina=0
        xgene.maxa=0
        xgene.bconstrain=False
        xgene.minb=0
        xgene.maxb=0
        xgene.cconstrain=True
        xgene.minc=c
        xgene.maxc=c
        xgene.dconstrain=False
        xgene.mind=0
        xgene.maxd=0
        xgene.econstrain=False
        xgene.mine=0
        xgene.maxe=0
        xgene.fconstrain=False
        xgene.minf=0
        xgene.maxf=0
        xgene.gconstrain=False
        xgene.ming=0
        xgene.maxg=0



        # maximise results written to outfile.txt
        xgene.genetic_algorithm()
        direction="n"
        #minimise results written to outfile.txt
        xgene.genetic_algorithm()




    for d in range(0,2**1):
        direction="x"
        xgene.pconstrain=False
        xgene.minp=0
        xgene.maxp=0
        xgene.aconstrain=False
        xgene.mina=0
        xgene.maxa=0
        xgene.bconstrain=False
        xgene.minb=0
        xgene.maxb=0
        xgene.cconstrain=False
        xgene.minc=0
        xgene.maxc=0
        xgene.dconstrain=True
        xgene.mind=d
        xgene.maxd=d
        xgene.econstrain=False
        xgene.mine=0
        xgene.maxe=0
        xgene.fconstrain=False
        xgene.minf=0
        xgene.maxf=0
        xgene.gconstrain=False
        xgene.ming=0
        xgene.maxg=0

        

        # maximise results written to outfile.txt
        xgene.genetic_algorithm()
        direction="n"
        #minimise results written to outfile.txt
        xgene.genetic_algorithm()


    for e in range(0,2**1):
        direction="x"
        xgene.pconstrain=False
        xgene.minp=0
        xgene.maxp=0
        xgene.aconstrain=False
        xgene.mina=0
        xgene.maxa=0
        xgene.bconstrain=False
        xgene.minb=0
        xgene.maxb=0
        xgene.cconstrain=False
        xgene.minc=0
        xgene.maxc=0
        xgene.dconstrain=False
        xgene.mind=0
        xgene.maxd=0
        xgene.econstrain=True
        xgene.mine=e
        xgene.maxe=e
        xgene.fconstrain=False
        xgene.minf=0
        xgene.maxf=0
        xgene.gconstrain=False
        xgene.ming=0
        xgene.maxg=0

        

        # maximise results written to outfile.txt
        xgene.genetic_algorithm()
        direction="n"
        #minimise results written to outfile.txt
        xgene.genetic_algorithm()

    for f in range(11,46):  # temp
        direction="x"
        xgene.pconstrain=False
        xgene.minp=0
        xgene.maxp=0
        xgene.aconstrain=False
        xgene.mina=0
        xgene.maxa=0
        xgene.bconstrain=False
        xgene.minb=0
        xgene.maxb=0
        xgene.cconstrain=False
        xgene.minc=0
        xgene.maxc=0
        xgene.dconstrain=False
        xgene.mind=0
        xgene.maxd=0
        xgene.econstrain=False
        xgene.mine=0
        xgene.maxe=0
        xgene.fconstrain=True
        xgene.minf=f
        xgene.maxf=f
        xgene.gconstrain=False
        xgene.ming=0
        xgene.maxg=0
        

        # maximise results written to outfile.txt
        xgene.genetic_algorithm()
        direction="n"
        #minimise results written to outfile.txt
        xgene.genetic_algorithm()

    for g in range(0,2**4):
        direction="x"
        xgene.pconstrain=False
        xgene.minp=0
        xgene.maxp=0
        xgene.aconstrain=False
        xgene.mina=0
        xgene.maxa=0
        xgene.bconstrain=False
        xgene.minb=0
        xgene.maxb=0
        xgene.cconstrain=False
        xgene.minc=0
        xgene.maxc=0
        xgene.dconstrain=False
        xgene.mind=0
        xgene.maxd=0
        xgene.econstrain=False
        xgene.mine=0
        xgene.maxe=0
        xgene.fconstrain=False
        xgene.minf=0
        xgene.maxf=0
        xgene.gconstrain=True
        xgene.ming=g
        xgene.maxg=g
        

        # maximise results written to outfile.txt
        xgene.genetic_algorithm()
        direction="n"
        #minimise results written to outfile.txt
        xgene.genetic_algorithm()





outfile.close()

