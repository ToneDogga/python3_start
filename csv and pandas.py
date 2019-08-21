#
#!/usr/bin/env python
#
from __future__ import print_function
from __future__ import division


import numpy as np
import pandas as pd

csv_envir = pd.DataFrame() #creates a new dataframe that's empty, we are not using it with the fileseek method
csv_envir=pd.read_csv("payoff_7d.csv",delimiter=',', header=None)   #, chunksize=chunksize)

print(csv_envir)

input("?")

csv_envir.to_csv("newpayoff_7d.csv",encoding='utf-8', index=False, header=None)   #, chunksize=chunksize)


##
##individual = np.dtype([('fitness','f'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1",allele_len),("chromo2",allele_len),("expressed",allele_len)])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
##poparray = np.zeros(r.pop_size, dtype=individual) 
##population = pd.DataFrame({"fitness":poparray['fitness'],"parentid1":poparray['parentid1'],"xpoint1":poparray["xpoint1"],"parentid2":poparray['parentid2'],"xpoint2":poparray["xpoint2"],"chromo1":poparray["chromo1"],"chromo2":poparray["chromo2"],"expressed":poparray["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': areaprint("\n\n")
###  mates_df = pd.DataFrame(mates, columns=list('ABCD'))
###  mates_df = pd.DataFrame(mates, dtype=individual)   #columns=list('ABCD'))
##    mates_df = pd.DataFrame(mates, columns=["fitness","parentid1","xpoint1","parentid2","xpoint2","chromo1","chromo2","newchromo1","newchromo2","expressed"])
##
##
##
### print("mates df")
### print(mates_df)
##
##
##
### pd.concat([pd.DataFrame(mates[i][0], columns=['chromo1']) for i in range(0,5)], ignore_index=True)
### pd.concat([newpopulation([i], columns=['chromo1']) for i in range(0,5)], ignore_index=True)
###  crossed_population.append(mates_df, ignore_index=True,sort=False)
##
##
###mates_df.loc["xpoint1"]=2
##
##
###   input("?")
##
###delete columns "newchromo1" and "newchromo2"
##
##    mates_df=mates_df.drop(columns=["chromo1","chromo2"])   # delete old chromosomes columns in population
##    mates_df.columns=["fitness","parentid1","xpoint1","parentid2","xpoint2","chromo1","chromo2","expressed"]  # rename newchromos to chromos
##
##    mates_df.index
###  print("mates df drop columns and rename")
