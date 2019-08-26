import pickle
import numpy as np

stop_test=[(1,2),(3,4),(5,6),(7,8)]
no_of_stops=len(stop_test)

##path_screen_type = np.dtype(dict(
##    names=['generation', 'epoch_length', 'bestgeneration', 'best_distance','stop'],
##    formats=[np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))],
##    offsets=[0, 8, 16, 16, 16]
##))

##path_screen_type = np.dtype(dict(
##    names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],
##    formats=[np.int32,np.str,np.bool,np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))],
##    offsets=[5,3,0,0, 8, 8, 16, 0]
##))

path_screen_type = np.dtype(dict(
    names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],
    formats=[np.int32,'|S25',np.bool,np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))]
))



path_screen=np.zeros((1,),dtype=path_screen_type)   # one row only

#path_screen=np.empty([1,],dtype=path_screen_type)
print(path_screen)
input("?")
##path_screen['epoch_length']=[1]  #,2,3]
##print(path_screen)
##input("?")
path_screen['generation'][0]=53   #[1,2,3]
print(path_screen)
input("?")
path_screen['message'][0]="Hello, world!"   #[1,2,3]
print(path_screen)
input("?")
path_screen['redraw'][0]=True   #[1,2,3]
print(path_screen)
input("?")
path_screen['pid'][0]=1234   #[1,2,3]
print(path_screen)
input("?")
path_screen['best_distance'][0]=88.8   #[1,2,3]
print(path_screen)
input("?")
path_screen['bestgeneration'][0]=9   #[1,2,3]
print(path_screen)
input("?")


##for i in range(0,len(genepool[bestjourneyno])):
##    print(i,stops[genepool[bestjourneyno][i]]) # 
##    path_screen["stop"][0][i]=stops[genepool[bestjourneyno][i]]  # 

for i in range(0,no_of_stops):
    path_screen["stop"][0][i]=stop_test[i]  # 


print(path_screen)
input("?")







##
##path_screen['stop'][0]=[5,6]
##print(path_screen)
##input("?")
##path_screen['stop'][1][1]=8
##print(path_screen)
##input("?")
##path_screen['stop'][1][1][1]=7
##print(path_screen)
##input("?")
##
##print(path_screen_type.names)
##print(path_screen_type.fields)
##input("?")
##

x_as_bytes = pickle.dumps(path_screen)
print(x_as_bytes)
print(type(x_as_bytes))

y = pickle.loads(x_as_bytes)
print(y)



print(y["message"][0].decode("utf-8"))
print(y["redraw"][0])
print(y["pid"][0])
print(y["stop"][0])
