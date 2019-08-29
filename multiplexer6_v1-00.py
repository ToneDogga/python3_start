import pickle
import numpy as np
import random

# multiplexer
# 6 signals come into the multiplexer
# first two lines are A are decoded as a unsigned binary integer
# this address values is then used to work out which of the four remaining signels
# on the Data or d lines is to be passed thwough to the mulriplexer output


def generate_data(data):
    i=0
    for i in range(0,len(data)):
        data[i]=random.randint(0,1)
        i+=1

    return(data)

def generate_signal(addresses):
    i=0
    for i in range(0,len(addresses)):
        addresses[i]=random.randint(0,1)
        i+=1
   
    return(addresses)



def multiplexer(addresses,data):
    
    data=generate_data(data)
    print("data=",data)
    
    while True:
        signal=generate_signal(addresses)
        
        print("signal=",signal)
      #  print("data=",data)

        output_address=int("".join(map(str,signal)),2)
        output=data[output_address]

       # print("output address=",output_address)
        print("output=",output)
        print("")
            
    return





no_of_address_bits=8
no_of_data_bits=2**no_of_address_bits
total_inputs=no_of_address_bits+no_of_data_bits
print("Address bits=",no_of_address_bits," Data bits=",no_of_data_bits," Total number of inputs=",total_inputs)

addresses=[0]*no_of_address_bits
data=[0]*no_of_data_bits

#print("before addresses=",addresses)
#print("before data=",data)

multiplexer(addresses,data)
































##
##stop_test=[(1,2),(3,4),(5,6),(7,8)]
##no_of_stops=len(stop_test)
##
##pickle_test=dict(
##    names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],
##    formats=[np.int32,'|S25',np.bool,np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))]
##)
##
##path_screen_type = np.dtype(pickle_test)


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

##path_screen_type = np.dtype(dict(
##    names=["pid","message","redraw","generation", "epoch_length", "bestgeneration", "best_distance","stop"],
##    formats=[np.int32,'|S25',np.bool,np.int32, np.int32, np.int32, np.float64, (np.int32, (no_of_stops,2))]
##))

##
##
##path_screen=np.zeros((1,),dtype=path_screen_type)   # one row only
##
###path_screen=np.empty([1,],dtype=path_screen_type)
##print(path_screen)
##input("?")
####path_screen['epoch_length']=[1]  #,2,3]
####print(path_screen)
####input("?")
##path_##
##for i in range(0,no_of_stops):
##    path_screen["stop"][0][i]=stop_test[i]  # 
##
##
##print(path_screen)
##input("?")
##
##
##
##
##
##
##
##
##x_as_bytes = pickle.dumps(path_screen)
##print(x_as_bytes)
##print(type(x_as_bytes))
##
##y = pickle.loads(x_as_bytes)
##print(y)
##
##
##
##print(y["message"][0].decode("utf-8"))
##print(y["redraw"][0])
##print(y["pid"][0])
##print(y["stop"][0])
##
##y['stop'][0][1]=[55,66]
##y['message'][0]="Hello, world! more"   #[1,2,3]
##
###print(path_scree)
###input("?")
##y['stop'][0][1][0]=8
##print(y)
##print(y["message"][0].decode("utf-8"))
##print(y["redraw"][0])
##print(y["pid"][0])
##print(y["stop"][0])
##
##print(y)
###input("?")
##
###print(y.names)
###print(y.fields)
##input("?")
##
#screen['generation'][0]=53   #[1,2,3]
###print(path_screen)
###input("?")
##path_screen['message'][0]="Hello, world!"   #[1,2,3]
###print(path_screen)
###input("?")
##path_screen['redraw'][0]=True   #[1,2,3]
###print(path_screen)
###input("?")
##path_screen['pid'][0]=1234   #[1,2,3]
###print(path_screen)
###input("?")
##path_screen['best_distance'][0]=88.8   #[1,2,3]
###print(path_screen)
###input("?")
##path_screen['bestgeneration'][0]=9   #[1,2,3]
###print(path_screen)
###input("?")


##for i in range(0,len(genepool[bestjourneyno])):
##    print(i,stops[genepool[bestjourneyno][i]]) # 
##    path_screen["stop"][0][i]=stops[genepool[bestjourneyno][i]]  # 
##
##for i in range(0,no_of_stops):
##    path_screen["stop"][0][i]=stop_test[i]  # 
##
##
##print(path_screen)
##input("?")
##
##
##
##
##
##
##
##
##x_as_bytes = pickle.dumps(path_screen)
##print(x_as_bytes)
##print(type(x_as_bytes))
##
##y = pickle.loads(x_as_bytes)
##print(y)
##
##
##
##print(y["message"][0].decode("utf-8"))
##print(y["redraw"][0])
##print(y["pid"][0])
##print(y["stop"][0])
##
##y['stop'][0][1]=[55,66]
##y['message'][0]="Hello, world! more"   #[1,2,3]
##
###print(path_scree)
###input("?")
##y['stop'][0][1][0]=8
##print(y)
##print(y["message"][0].decode("utf-8"))
##print(y["redraw"][0])
##print(y["pid"][0])
##print(y["stop"][0])
##
##print(y)
###input("?")
##
###print(y.names)
###print(y.fields)
##input("?")
##
