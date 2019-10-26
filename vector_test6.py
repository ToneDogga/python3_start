import numpy as np
import random
from scipy.spatial import distance



#
# reproduction
# total the fitness of each string and create a % breakdown score of the total for each of the strings in the population
# create a biased roulette probability_table where the % breakdown score is the probability of the probability_table landing on that string
# spin the wheel m times each time yielding a reproduction candidate of the population
# in this way more highly fit strings have more offspring in the next generation.
#
# crossover - PMX partially matched crossover
# the new generation of strings is mated at random
# this means each pair of strings is crossed over at a uniform point in the string
# find two random split points in the string a swap the remaining information over between the mating pairs
#
# mutation
# this is the occasional and small chance SB (1/1000) that an element of a journey list is swapped randomly.
#
# go back to start with the new generation


def create_stops(x,y,no_of_stops):
    stops=np.zeros((1,no_of_stops,2)).astype(int)   # starting point - distance
    endpoint=np.zeros((1,no_of_stops,2)).astype(int)
    for s in range(1,no_of_stops):
        point=np.zeros((1,2)).astype(int)
      #  unique=False
     #   while not unique:
        point[0][0]=random.randint(0,x-1)
        point[0][1]=random.randint(0,y-1)
        if s==1:
            endpoint=point
            # check if point already exists
       #     if stops.size==0:
        #        unique=True
        #    else:
         #       if np.any(stops==point):
             #   for h in stops:
              #      print("h=",h,"poimnt=",point)
               #     input("?")
               #     if (point[0][0]==h) and (point[0][1]==h):
          #          print("point found",point)
          #          unique=False
          #          break
          #      else:
          #          unique=True       
             #   print("h loop finished")       

        stops[0][s-1]=point
    stops[0][s]=endpoint
    #print("create stops=",stops)
    #input("?")
    return(stops)  


def create_starting_genepool(no_of_stops,poolsize):
    #stops=j.no_of_stops   # less one as we need to start and finish in the same place
    gene=[]
    for g in range(poolsize):
        gene.append([])
        gene[g]=[]
        for alleles in range(0,no_of_stops-1):
            unique=False
            c=gene[g]
            while not unique:
                destination=random.randint(0,no_of_stops-2)
                # check if point already exists
                if len(gene[g])==0:
                    unique=True
                else:    
                    for d in c:
               #     print("h=",h,"poimnt=",point)
               #     input("?")
                        if (destination==d):
                   #     print("point found",point,h)
                            unique=False
                            break
                        else:
                            unique=True       
              #  print("gene=",gene)
              #  input("?")
             
            gene[g].append(destination)
        gene[g].append(gene[g][0])   # return to end
    return(np.asarray(gene))       


def distance_table(stops):
    # Path is now a 2D array of coordinates so 3D in total
    # each row is the route of each gene in the poolpool
    # b is an array of the next coordinate shifted by one to calc the distance
    #b=np.roll(stops,-1,axis=1)
    stops=stops[0]
    
    dt = np.around(distance.cdist(stops,stops,'euclidean'),2)
    
    return(dt)


def distance_table_prohibit_paths_start(dt):
#    print("start before flat gp=",current_path_indexes)
#    cpi=np.ndarray.flatten(current_path_indexes)
#    print("start flat cpi=",cpi)
#    print("statr dt before prohibit=\n",dt)
    cpi=np.arange(0,len(dt))
    dt[cpi,cpi]=999   # mark paths it is on so it can't go back

   # dt[current_path_indexes,current_path_indexes]=999   # mark paths it is on so it can't go back
    dt[0,dt.shape[1]-1]=999   # remove the option to go straight to the end which is itself
    dt[dt.shape[0]-1,0]=999   # remove the option to go straight to the end which is itself
  #  print("start dt after prohibit=\n",dt)

    return(dt)  



def distance_table_prohibit_paths_loop(dt,row_index,col_index):
 #   print("loop before flat gp=row=",row_index,"col=",col_index)
    cpi=np.ndarray.flatten(col_index)
    rpi=np.ndarray.flatten(row_index)

  #  print("loop flat rpi=",rpi,"cpi=",cpi)
  #  print("dt before prohibit loop=\n",dt)
  #  dt[cpi,cpi]=999   # mark paths it is on so it can't go back
    dt[rpi,cpi]=999
    dt[cpi,rpi]=999   # mark both sides of the table

   # dt[current_path_indexes,current_path_indexes]=999   # mark paths it is on so it can't go back
   # dt[current_path_indexes,dt.shape[1]-1]=999   # remove the option to go straight to the end which is itself
   # dt[dt.shape[0]-1,current_path_indexes]=999   # remove the option to go straight to the end which is itself
  #  print("dt after prohibit loop=\n",dt)

    return(dt)  



def create_one_greedy_gene(genepool,stops,no_of_stops):
    # use a greedy algorithm to replace any genes lost in the genepool due to duplication
    # algorithm. vectorised with numpy
    # pick a random point
    # find the closest.  connect rinse and repeat
    # remove last return stop
 #   stops=stops[:,:no_of_stops-1,:]
 #   print("stops -1=",stops)
    

    
  # for g in range(0,genes_needed):

    stop_count=0
    start_index=np.unique(np.random.randint(0,no_of_stops-1,1,dtype=int))
 #   start_index=np.random.randint(0,no_of_stops-1,dtype=int)

 #   print("start_index=",start_index)
  #  print("greedy needs:",genes_needed)
 #   no_of_greedy_paths=len(start_index)
    dist=distance_table(stops)
  #  print("dist=",dist)
  #  print("number of greedy paths to calc",no_of_greedy_paths)
  #  greedy_paths=np.zeros((1,len(start_index)),dtype=int)
    greedy_path=np.zeros((1,1),dtype=int)

    greedy_path[0,...]=start_index
  #  print("starting greedy paths=",greedy_paths)
    dist_for_start_only=distance_table_prohibit_paths_start(dist)
    #dist_for_start_only=distance_table_prohibit_paths_loop(dist_for_start_only,start_index,start_index)

    while stop_count<no_of_stops-2:

        # "from" is the rows, "to" is the columns
        # distance table contains only the rows of the start index
        # mark the elements we cant return to
        # that is the current positions and end positions with a 999 so they lose the sort
        #
    
     #   dist_for_start_only=distance_table_prohibit_paths_loop(dist,greedy_paths)  #np.ndarray.flatten(greedy_paths))   # mark orginal distance table no go areas 
     #   dist_for_start_only=distance_table_prohibit_paths_loop(dist,start_index,indexes_to_shortest_dist)  #np.ndarray.flatten(greedy_paths))   # mark orginal distance table no go areas 

        #start_index=greedy_paths[stop_count]
   #     print("start index=",start_index)
        distances=dist_for_start_only[start_index]  # only include "from" rows where it currently is

    #    print("distances=",distances)


       # start_pos=stops[0,start_index,...]
       # print("start_pos=",start_pos,"stop count=",stop_count)
        loop=True
        while loop:    
            distsort=np.sort(distances)

             #   print("distsort=",distsort)
             #   print("distsort[...,1]=",distsort[...,0])  # lowest distances

            distsort_index=np.argsort(dist[start_index])
        
            indexes_to_shortest_dist=distsort_index[...,0]   # indexes to lowest distances
            #if (greedy_paths==indexes_to_shortest_dist).any():
        #    print("greedy path",greedy_paths,"index in greedy paths",indexes_to_shortest_dist)
            if indexes_to_shortest_dist in greedy_path:
                dist_for_start_only=distance_table_prohibit_paths_loop(dist,start_index,indexes_to_shortest_dist)
                loop=True
            else:
                loop=False
                
    #    print("distsort index=",indexes_to_shortest_dist)   # indexes to lowest distances
        dist_for_start_only=distance_table_prohibit_paths_loop(dist,start_index,indexes_to_shortest_dist)  #np.ndarray.flatten(greedy_paths))   # mark orginal distance table no go areas 


        start_index=distsort_index[...,0]
        appendable_index=np.reshape(start_index,(1,-1))
        greedy_path=np.append(greedy_path,appendable_index,axis=0)

        dist=dist_for_start_only
    #    print("final dist at end of loop=\n",dist)
    #    print("greedy paths=\n",greedy_paths)
        stop_count+=1
  #  print("final greedy path=\n",greedy_path)
    greedy_path=np.append(greedy_path,greedy_path[0:1])   #,axis=0)
    greedy_path=greedy_path[np.newaxis,...]
    print("final greedy path=\n",greedy_path,"\ngp=\n",genepool)
    genepool=np.append(genepool,greedy_path,axis=0)
    return(genepool)


 

def create_wheel(probs):
    count=np.arange(len(probs),dtype=int)
#    print("count=",count,"probs=",probs)
    wheel=np.repeat(count,probs)
    return(wheel)

def vpath_distance(pdata):
    # vectorized version for speed   
    stoplist=np.array(pdata[...,0],dtype=int)
    nextlist=np.array(stoplist[1:],dtype=int)
    destinations=np.append(nextlist,[stoplist[0]])
  #  print("stoplist=",stoplist)
  #  print("destinations=",destinations)
  #  print("\npdata=\n",pdata)
    stopdist=pdata[stoplist]
    distances=stopdist[...,2:]
  #  print("\ndistances=\n",distances)
    d=distances[[stoplist],[destinations]]
   # print("\nindexed distances=\n",d)

  #  return(np.sum(distances[[stoplist],[destinations]]))
    return(np.sum(d))

    
#greedy_needs=1
no_of_stops=8
stops=create_stops(80,60,no_of_stops)
#print("stops=",stops)
#stops=np.array([[0,1,2],[2,0,1],[2,1,0],[1,0,2],[0,2,1]])
#stops=np.array([[0,1,2],[1,0,2]])

#no_of_stops+=1
#print("no_of_stops=",no_of_stops)
#data=np.array([[0,(10,20),0,3.2,2.1],[1,(11,30),4,0,7.7],[2,(22,4),5.2,6,0]])
genepool=create_starting_genepool(no_of_stops,8)
#print("len genepool=",len(genepool),genepool)
genepool=create_one_greedy_gene(genepool,stops,no_of_stops)

#print("len genepool=",len(genepool),genepool)

print("genepool=\n",genepool)   #[0:1,...])
                
#genepool=np.append(genepool,gene,axis=1)

#print("genepool after=\n",genepool)

##test=vpath_distance(data)
##print("distance=",test)
##print("\n\n\n")
##test2=vall_distance(stops,data)
##print("wheel weights=",test2)
##
##wheel=create_wheel(test2)
##print("final wheel=",wheel)
