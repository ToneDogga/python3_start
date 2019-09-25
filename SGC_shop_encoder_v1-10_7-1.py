# 
#
# 
# 
#
#
#
#
#!/usr/bin/env python
#
from __future__ import print_function
from __future__ import division


import numpy as np
import pandas as pd
import random
import time
import csv
import math
import linecache
import sys
import platform
import itertools
import datetime
import subprocess as sp
from collections import Counter
from statistics import mean
from multiprocessing import Pool, Process, Lock, freeze_support, Queue, Lock, Manager
from os import getpid
import os
import hashlib
import multiprocessing 
from timeit import default_timer as timer
import pickle
#from difflib import SequenceMatcher


################################################################3





def read_env_file_in(filename):
    population=pd.read_csv(filename)    #,dtype={"condition":np.str,"message":np.str,'match_flag': np.bool,"winners":np.bool})
    return(population)


def normalise(row,col,c,reduction):
    return(int(math.floor(float((row[col[c]]-reduction)))))


def scale(row,col,c,scalingneeded):
    return(int(math.floor(float((row[col[c]]/(2**scalingneeded))))))

def turn_to_binary(row,c,col,bits):
    #return(str(int(str(row[col[c]]),base=2)).zfill(int(bits[c])))
    
    return(str(format(row[col[c]], "0"+str(bits[c])+"b")))
    



##def check_for_nans_in_rows(population):
##    df1 = population[population.isna().any(axis=1)]
##    print("check for nans=",df1)
##    return

def SGC_encoder(params):
    tmp=sp.call(params["cls"],shell=True)  # clear screen 'use 'clear for unix, cls for windows

    f=open(params["reportfile"],"w")
    
    print("\n\nSimple Genetic Classifier Encoder v1.00 algorithm written by Anthony Paech 19/9/19")
    print("===================================================================================\n\n")

    print("Reading in:",params["infile"])

    f.write("\n\nSimple Genetic Classifier Encoder v1.00 algorithm written by Anthony Paech 19/9/19\n")
    f.write("===================================================================================\n\n")

    f.write("Reading in:"+str(params["infile"])+"\n")

    
    print("\n")    
    envdata_in=read_env_file_in(params["infile"])
  #  print(envdata_in.to_string())
  #  print("\n")
    envend=len(envdata_in.columns)
    endlen=len(envdata_in)
 #   print(envdata_in)
#envdata_insize
  #  input("?")
    col=[]
    for c in range(0,len(envdata_in.columns)):
        col.append(str(envdata_in.columns[c]))
        print("column",c,"name:",col[c])   #,"len:",envdata_in.iloc[:,[c]])    #.apply(len))
        f.write("column"+str(c)+" name:"+str(col[c])+"\n")   #,"len:",envdata_in.iloc[:,[c]])    #.apply(len))
    print("\nTypes:")
    f.write("\nTypes:"+str(envdata_in.dtypes)+"\n")
    print(envdata_in.dtypes  )
    print("\nCondition columns: 0-"+str(envend-2))
    print("Messager column=",envend-1)
    f.write("\nCondition columns: 0-"+str(envend-2)+"\n")
    f.write("Messager column="+str(envend-1)+"\n")

    bits=[]
    scaling=[]
    print("\n")
    f.write("\n\n")
    for c in range(0,len(envdata_in.columns)):  
        maxscale=envdata_in[col[c]].max()
        minscale=envdata_in[col[c]].min()
        bitsneeded = math.floor( math.log2(maxscale) + 1 )
        bitschosen=int(input("column"+str(c)+" name:"+str(envdata_in.columns[c])+" max:"+str(maxscale)+" min:"+str(minscale)+" recommended number of bits "+str(bitsneeded)+" ===> Number of bits in encoded column?"))

        if bitschosen<bitsneeded:   # scale the values.
            scalingneeded=bitsneeded-bitschosen
            norm_need=maxscale-minscale-1
            bs=math.floor( math.log2(norm_need) + 1 )
            if minscale>0:
                print("Normalisation is possible on this column.",str(col[c]),". I will reduce all the values by ",minscale,"max would be=",norm_need,"potential bits needed",bs)
                f.write("Normalisation is possible on this column. "+str(col[c])+" I will reduce all the values by "+str(minscale)+" max would be="+str(norm_need)+" potential bits needed"+str(bs)+"\n")

         #   input("?")
                envdata_in[col[c]]=envdata_in.apply(normalise,axis=1,args=[col,c,minscale])   #int(envdata_in[col[c]]/(2*scalingneeded))

            newmaxscale=envdata_in[col[c]].max()
            newminscale=envdata_in[col[c]].min()
            print("new min=",newminscale," newmax=",newmaxscale)   
            print("Scaling needed. col:",str(col[c])," all values divided by",2**scalingneeded," maxvalue in column now=",int(math.floor(float(newmaxscale/(2**scalingneeded))))," minvalue in column now=",int(math.floor(round((newminscale/(2**scalingneeded))))))
            f.write("new min="+str(newminscale)+" newmax="+str(newmaxscale)+"\n")   
            f.write("Scaling needed col:"+str(col[c])+" all values divided by "+str(2**scalingneeded)+" maxvalue in column now="+str(int(math.floor(float(newmaxscale/(2**scalingneeded)))))+" minvalue in column now="+str(int(math.floor(round((newminscale/(2**scalingneeded))))))+"\n")


            #input("?")
            envdata_in[col[c]]=envdata_in.apply(scale,axis=1,args=[col,c,scalingneeded])   #int(envdata_in[col[c]]/(2*scalingneeded))
            newnewmaxscale=envdata_in[col[c]].max()
            newnewminscale=envdata_in[col[c]].min()
            print("new new min=",newnewminscale," new newmax=",newnewmaxscale)
            f.write("new new min="+str(newnewminscale)+" new newmax="+str(newnewmaxscale)+"\n")   

            input("?")
        bits.append(bitschosen)     

    print("\n\ncol=",col,"\nbits=",bits,"\n")
    print(envdata_in)
    f.write("\n\ncol="+str(col)+"\nbits="+str(bits)+"\n\n")
    f.write(envdata_in.to_string())
    f.write("\n\n")

    input("?")


################################################
#    encode to binary.
# create pandas outfile dataframe
    envdata_out_type = np.dtype([(col[0],"S"+str(bits[0])),(col[1],"S"+str(bits[1])),(col[2],"S"+str(bits[2])),(col[3],"S"+str(bits[3])),(col[4],"S"+str(bits[4])),(col[5],"S"+str(bits[5])),(col[6],"S"+str(bits[6])),(col[7],"S"+str(bits[7]))])
    envdata_array=np.zeros(endlen,dtype=envdata_out_type)
    envdata_out=pd.DataFrame({col[0]:envdata_array[col[0]],col[1]:envdata_array[col[1]],col[2]:envdata_array[col[2]],col[3]:envdata_array[col[3]],col[4]:envdata_array[col[4]],col[5]:envdata_array[col[5]],col[6]:envdata_array[col[6]],col[7]:envdata_array[col[7]]})
  #  envdata_out2=pd.DataFrame({"condition":envdata_array["condition"],"message":envdata_array["message"]})
 #   print("\nenvdata_out")
 #   print(envdata_out)

##############################################
    #   Load columns, avoid the index
    

    for c in range(0,len(envdata_out.columns)):
       # print(type(col[c]))
        envdata_out[col[c]]=envdata_in.apply(turn_to_binary,axis=1,args=[c,col,bits])   #f.format(int(str(envdata_in[col[c]])))                  #str(int(s,2)).zfill(bits[c])

  #  envdata_out[col[8]]==envdata_in.apply(turn_to_binary,axis=1,args=[c-1,col,bits])    

    
    envdata_out.to_csv(params["outfile"], sep=',', encoding='utf-8',index=False)
   

    envdata_out2=pd.DataFrame()
  #  envdata_out2=envdata_out.assign(condition =lambda x: (x[col[0]]+x[col[1]]+x[col[2]]+x[col[3]]+x[col[4]]+x[col[5]]+x[col[6]]))
   # envdata_out2["condition"]=envdata_out.apply(lambda x: (x[col[0]]+x[col[1]]+x[col[2]]+x[col[3]]+x[col[4]]+x[col[5]]+x[col[6]]), axis=0)
    envdata_out2["condition"]=envdata_out[col[0]]+envdata_out[col[1]]+envdata_out[col[2]]+envdata_out[col[3]]+envdata_out[col[4]]+envdata_out[col[5]]+envdata_out[col[6]]

   # envend=len(envdata_out.columns)
   
 #   envdata_out2["message"]=envdata_out.apply(lambda x: (x[col[7]]))
    envdata_out2["message"]=envdata_out[col[7]]

    print(envdata_out2)
    
    envdata_out2.to_csv(params["concatfile"],columns=["condition","message"], sep=',', encoding='utf-8',index=False)

    print("Encoded outfile created:",params["outfile"])
    print("Encoded concatenated file created:",params["concatfile"])
    f.write("Encoded outfile created:"+str(params["outfile"])+"\n")
    f.write("Encoded concatenated file created:"+str(params["concatfile"])+"\n")
    f.flush()
    f.close()
    
    return

def main():
 #   freeze_support()
##    if(len(sys.argv) < 2 ) :
##        print("Usage : python MLGA_xxxxx.py epoch_length")
##        sys.exit()

#
    params = dict(
       # no_of_address_bits=2,
       # no_of_data_bits=2**2,   # binary,  in this case 6 input , 1 output
       # min_pop=40,   #3**3,
       # epoch=int(sys.argv[1]),  #1000,
        infile="shopsales31.csv",   #MLGA_environment.csv",
        outfile="shop_SGC_encoded31.csv",   #MLGA_environment.csv",
        concatfile="shop_concat_SGC_encoded31.csv",
        reportfile="SGC_encoded_report.txt")
       # diagnostic_file="MLGA_diagnostic.txt",
      #  results_file="MLGA_results.txt")

    if platform.system().strip().lower()[:7]=="windows":
        params["extra_eol_char"]=1
        params["cls"]="cls"
    else:
        params["extra_eol_char"]=0
        params["cls"]="clear"
#




    SGC_encoder(params)
    
 

if __name__ == '__main__':
    main()

