import numpy as np
import pandas as pd

ploidy=2
no_of_alleles=16
pop_size=10

"""
individual3 = np.dtype([('fitness','i8'),('parentid','i8',(ploidy)),('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
population3 = np.zeros(pop_size, dtype=individual3)
#print(population3)
print("\n")
print(population3[9])

#print("idnumber:",population3['idnumber'][0])
print("fitness:",population3['fitness'][0])
print("parentid",population3['parentid'][0])
print("xpoint",population3["xpoint"][0])
print("chromopack:",population3['chromopack'][0])
print("expressed:",population3['expressed'][0])

print("\n\n\n")

#data=pd.Series(population3)
#print("data")
#print(data.values)

#data.index

#$print(data.index)
"""

individual = np.dtype([('fitness','i8'),('parentid1','i8'),("xpoint1","i2"),("parentid2","i8"),("xpoint2","i2"),("chromo1","S16"),("chromo2","S16"),("expressed","S16")])   #,('xpoint','i2',(ploidy)), ('chromopack', 'i1', (ploidy, no_of_alleles)),('expressed','i1',(no_of_alleles))])
population = np.zeros(pop_size, dtype=individual)

#print(population)
print("\n\n")


#pop2 = pd.DataFrame({"population":population})   #,'area': area})
#pop2 = pd.DataFrame({"fitness":population['fitness'],"parentid":population['parentid'],"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': area})
pop = pd.DataFrame({"fitness":population['fitness'],"parentid1":population['parentid1'],"xpoint1":population["xpoint1"],"parentid2":population['parentid2'],"xpoint2":population["xpoint2"],"chromo1":population["chromo1"],"chromo2":population["chromo2"],"expressed":population["expressed"]})  #,"xpoint":population['xpoint'],"chromopack":population['chromopack'],"expressed":population['expressed']})   #,'area': area})
print(pop)



print("\n\n")


pop.index
print(pop)

#print("col=",pop.columns)

print("\n\n")
#print(pop.index)


#list(pop.items())


#print(pop.keys())

#pop.keys()
print(pop["xpoint1"])

print(pop["xpoint1"][2])
  

row=int(input("row?"))
col=input("col?")


print("=",pop[col][row])
print("?",pop.loc[row,col])


"""print("parentid1",pop['parentid1'][row])
print("xpoint1",pop["xpoint1"][row])
print("parentid2",pop['parentid2'][0])
print("xpoint2",pop["xpoint2"][0])
print("chromo1:",pop['chromo1'][0])
print("chromo2:",pop['chromo2'][0])
print("expressed:",pop['expressed'][0])
print("\n\n")
"""

