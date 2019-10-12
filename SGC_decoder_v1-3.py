# Simple genetic classifier decoder
#  3/10/19
#
# import a csv to a pandas dataframe
# 
##
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
import pickle
import platform
import subprocess as sp
import sys 


  
def count_file_rows(filename):
    with open(filename,'r') as f:
        return sum(1 for row in f)


def read_csv_file_in(filename):
    environment=pd.read_csv(filename, names=["condition","message"], header=0, dtype={"condition":np.str,"message":np.str})
   # print(environment)
  #  input("?")
    return(environment)



def save_bits_to_file(filename,itemlist):
    with open(filename, 'wb') as fp:
        pickle.dump(itemlist, fp)


def load_bits_from_file(filename):
    if not filename:
        return([])
    else:
        with open (filename, 'rb') as fp:
            itemlist = pickle.load(fp)
        return(itemlist)


def setup(import_file):
    results=read_csv_file_in(import_file)
    #print("results=",results.to_string())

    #condition_bits=len(results["condition"].max)
    condition_bits=results.condition.map(lambda x: len(x)).max()

    print("condition bits=",condition_bits)
    classifier=results.head(1)

    conditionstr=classifier.condition[0]
    #message_bits=len(results["message"].max)
    message_bits=results.message.map(lambda x: len(x)).max()
    print("message bits=",message_bits)
    messagestr=classifier.message[0]
    return(results,conditionstr,condition_bits,messagestr,message_bits)


def get_features(condition_bits):
    features=[]
    fs=0
    incorrect=True
    while incorrect:
        no_of_features=int(input("in the condition, how many features are encoded?"))
        incorrect=(no_of_features<=0 or no_of_features>condition_bits)
        if incorrect:
            print("no of features must be equal to or less than",condition_bits)
    print("name the features")
    for f in range(0,no_of_features):
        fnstr="condition feature name["+str(f)+"]?"
        incorrect=True
        while incorrect:
            fn=input(fnstr)
            incorrect=("?" in fn) or (":" in fn)

        if f==no_of_features-1 or fs>=condition_bits-1:
            fl=condition_bits-fs-1
        else:
            if f==0:
                fs=0
            flstr=0
            fltest=True
            while fltest:
                flstr="feature length?"
                fl=int(input(flstr))
                fltest=(fl<=0 or fl>=condition_bits or fs>condition_bits)
                if fltest:
                    print("feature #",f,"incorrect length.  please reneter.")
            
        features.append((fn,fs))
        fs=fs+fl

    features.append(("?",condition_bits))


    #print(features)
    return(features)

def get_message(message_bits):
    message=[]
    fs=0
    incorrect=True
    while incorrect:
        no_of_features=int(input("in the message, how many features are encoded?"))
        incorrect=(no_of_features<=0 or no_of_features>message_bits)
        if incorrect:
            print("no of features must be equal to or less than",message_bits)
    print("name the features")
    for f in range(0,no_of_features):
        mnstr="message feature name["+str(f)+"]?"
        incorrect=True
        while incorrect:
            mn=input(mnstr)
            incorrect=("?" in mn) or (":" in mn)

        if f==no_of_features-1 or fs>=message_bits-1:
            fl=message_bits-fs-1
        else:
            if f==0:
                fs=0
            flstr=0
            fltest=True
            while fltest:
                flstr="message feature length?"
                fl=int(input(flstr))
                fltest=(fl<=0 or fl>=message_bits or fs>message_bits)
                if fltest:
                    print("feature #",f,"incorrect length.  please reneter.")
            
        message.append((mn,fs))
        fs=fs+fl

    message.append(("?",message_bits))


  #  print(message)
    return(message)



def make_bits(features,conditionstr,condition_bits,message,messagestr,message_bits):
    bits=[]
    bc=0
    count=0    
    for cb in range(0,condition_bits):
        if cb==features[count+1][1]:
            bc=0
            count+=1
        fname=features[count][0]
    #    print("condition[",cb,"]=",conditionstr[cb],fname)
        bits.append((fname,bc))
        bc+=1

    bits.append((":",0))

    bc=0
    count=0    
    for cb in range(0,message_bits):
        if cb==message[count+1][1]:
            bc=0
            count+=1
        fname=message[count][0]
   #     print("message[",cb,"]=",messagestr[cb],fname)
        bits.append((fname,bc))
        bc+=1

    return(bits)


def input_question(results,bits):
    #
    # take the dataframe "results" and the list "bits" and give the user the ability to decode the classifier heirarcy
    # in a meaningful way
    # a classifier is on the form condition_string : message_string
    # there are only three symbols in the dataframe strings
    # "0" = no, "1" = yes and "#"=dont care
    #  each bit for both the condition and the message is named in the bit list
    #
    # a query takes the form
    # featurename1 bit0     (Yes, No, dont care)
    # featurename1 bit1     
    #   .......
    # featurename2 bit0
    #   ......
    #
    
    # it returns an answer string which is of the message strings in heierchy order which match
    # in the form True or False
    #
    print("Condition input.")
    answer=False
    loop=True
    while loop:
        bit_question=[]
        bit_mask1=""
#        bit_mask2=""
#        bit_mask3=""
        cbit_mask1=""   #"condition==\""
#        cbit_mask2="condition==\""
#        cbit_mask3="condition==\""


       
        for elem in bits:
            name=str(elem[0])
            if name!=":":
                #    break
                number=str(elem[1])
           #     bit_question.append(input(name+" bit:"+number+" y/n/#?"))
                notcorrect=True
                while notcorrect:
                    bbinput=input(name+" bit:"+number+" y/n/?")
                    notcorrect=not (bbinput=="y" or bbinput=="n" or bbinput=="?")
                    if notcorrect:
                        print("Please enter either y or n or ?")
                        
                bit_question.append(bbinput)
                if bbinput=="y":
                    bit_mask1=bit_mask1+"1"
 #                   bit_mask2=bit_mask2+"1"
  #                  bit_mask3=bit_mask3+"1"
                elif bbinput=="n":
                    bit_mask1=bit_mask1+"0"
 #                   bit_mask2=bit_mask2+"0"
 #                   bit_mask3=bit_mask3+"0"
                elif bbinput=="?":
                    bit_mask1=bit_mask1+"#"
 #                   bit_mask2=bit_mask2+"1"
 #                   bit_mask3=bit_mask3+"0"
                else:
                    print("bit error")
            else:
                
                print("\nMessage input")
                cbit_mask1=bit_mask1
 #               cbit_mask2=cbit_mask2+bit_mask2+"\""
 #               cbit_mask3=cbit_mask3+bit_mask3+"\""

                bit_mask1=""
 #               bit_mask2="message==\""
 #               bit_mask3="message==\""
               

        loop=not (("?" in bit_question) and (("y" in bit_question) or ("n" in bit_question)))
        if loop:
            print("Question must have at least one '?' and at least one 'y' or 'n', please reenter.")
            
            
   # print("\n\n",bit_question,"mask=",bit_mask)

#  note that the message in the results dataframe does not have #'s in it
 #       bit_mask1=bit_mask1+"\""
 #       bit_mask2=bit_mask2+"\""
 #       bit_mask3=bit_mask3+"\""



    return(cbit_mask1,bit_mask1)



def create_variants(question):
    # take a generic question string and add all possible combinations with the don't care char "#" replaced by either a 0 or 1
    # cquestion
    #
    mlen=len(question)
    if mlen==1:
        question=question+" "
        mlen=2
    hashes=question.count("#")        
    total=3**hashes
  #  print("question=",question,"#=",hashes,"~#=",non_hashes,"total=",total,"\n\n")

    string_check(question,0,mlen,[],total)
    with open ("writefile.txt", 'rb') as fp:
        itemlist = pickle.load(fp)

   # print("mquestion1 set=",itemlist)
    return(itemlist)

def string_check(string,pos,length,r,total):
    # check if string is finished
    r.append(string.strip())    
    if pos>length-1:
        with open("writefile.txt", 'wb') as fp:
            pickle.dump(list(set(r)), fp)
        return
    elif pos==length-1:
        print("\rProgress:{0:.0f}%".format(int(len(list(set(r)))/total*100)),"   #",total,end="\r",flush=True)
        if string[pos]=="#":
            string_check(string[:pos]+"1",pos+1,length,r,total)
            string_check(string[:pos]+"0",pos+1,length,r,total)
            string_check(string[:pos]+"#",pos+1,length,r,total)
        else:
            string_check(string,pos+1,length,r,total)
    else:
        print("\rProgress:{0:.0f}%".format(int(len(list(set(r)))/total*100)),"   #",end="\r",flush=True)
        if string[pos]=="#":
            string_check(string[:pos]+"1"+string[pos+1:],pos+1,length,r,total)
            string_check(string[:pos]+"0"+string[pos+1:],pos+1,length,r,total)
            string_check(string[:pos]+"#"+string[pos+1:],pos+1,length,r,total)
        else:
            string_check(string,pos+1,length,r,total)


    #  c and m question is in the form of a bit_mask string like "##101###" 
    # note that the message part (after the :) is not in the form of #'s it is only 1's or 0's
    #  the answer is in the form of a dataframe of matching conditions in the results dataframe










def main():

    if(len(sys.argv) < 2 ) :
        print("Usage : python SGC_decoder...py import_filename.csv bitlist.dat")
        sys.exit()

      
    if platform.system().strip().lower()[:7]=="windows":
        extra_eol_char="\n"
        cls="cls"
    else:
        extra_eol_char=""
        cls="clear"


#  clear screen
    tmp=sp.call(cls,shell=True)  # clear screen 'use 'clear for unix, cls for windows


   # import_file="metaresults_csv.csv"

    import_file=sys.argv[1]  #name of csv file to import into the results dataframe

    
    features=[]
    message=[]
    fs=0
    ff=0
    print("\n\n\nSimple genetic classifier decoder v1.  Name the bits.")
    print("===========================================\n")
    results,conditionstr,condition_bits,messagestr,message_bits=setup(import_file)
    print("Imported results file name=",import_file)
    #print(len(results))
    print(results.info)
    print("\n")
    

    bits=[]
    if len(sys.argv)==3:
        bits=load_bits_from_file(sys.argv[2])
        if len(bits)!=0:
            print("bits loaded:",bits,":",sys.argv[2])
        else:
            print("no bits loaded.",sys.argv[2])


    print("\n")
    
    if len(bits)==0:
        
       # print("First condition:",conditionstr,"first message:",messagestr)
        savefile=input("Enter decode bits save filename:")
        savefile=savefile+conditionstr+".dat"
        
        features=get_features(condition_bits)
        message=get_message(message_bits)
        bits=make_bits(features,conditionstr,condition_bits,message,messagestr,message_bits)
      #  print(bits)
        
        save_bits_to_file(savefile,bits)

       # input("bits saved")

    #    lbits=load_bits_from_file(savefile)
        print("\n",bits,"\nSaved to",savefile,"\n\n")

    print("Ask questions of the import file dataframe",import_file)
    print("===================================================================\n")
 #   print("bits=",bits)

    loop=True
    while loop:
        cquestion1,mquestion1=input_question(results,bits)
        print("\nCreate condition variants.") 
        cqlist=create_variants(cquestion1)
        print("\nCreate message variants.")
        mqlist=create_variants(mquestion1)
        print("\n\nNumber of unique condition variants to be checked=",len(cqlist),"Number of unique message variants to be checked=",len(mqlist))
        answer=pd.DataFrame(columns=["condition","message"])
        count=0
        total=len(cqlist)*len(mqlist)
        for cq in cqlist:
            for mq in mqlist:
                question_query="condition==\""+cq+"\" & message==\""+mq+"\""


                
                print("\rProgress:{0:.0f}%".format(int(count/total*100)),"     #",total,"len(answer)=",len(answer),"->",question_query,end="\r",flush=True)
 
                #print(question_query)
             #   rq=results.query(question_query)
               # if rq.empty:
                #    print(rq)
                answer=answer.append(results.query(question_query),ignore_index=False,sort=False)
                #print("question=",question_query,"Answer=",answer)
                count+=1
       # print(answer)
        with open("answer_csv.csv", 'a',newline="\n") as f:
        #               answer.sort_index(ascending=True,inplace=True)
        #    answer.to_csv(f, header=True, index=True, columns=["condition","message"])
            print("Question1=",cquestion1,":",mquestion1,"\n\n")
            f.write("Decoding:"+import_file+"\n")
           
            f.write("\n\nQuestion="+cquestion1+":"+mquestion1+"\n\n")
            answer.sort_index(ascending=True,inplace=True)
            answer.to_csv(f, header=True, index=True, columns=["condition","message"])
        print(answer) 
        cont=input("\n\nContinue? (y/n)")
        if cont=="n":
            loop=False

 

if __name__ == '__main__':
    main()





#mydict = {1:["george",16,"test1",True],2:["amber",19,"test2",False]}

#print(list(mydict.keys()))
#print(list(mydict.values()))
#print(list(mydict.items()))

#print(mydict.keys())
#print(mydict.values())
#print(mydict.items())

# sammy.update({'online': False})
##
##testgroups= { "key1":
##       {
##        "1":[
##            { "1c1r1":7,"1c2r1":"1c3r1testf2","1f33":True},
##            { "1f1":7,"1f2":"1testf2","1f3":True},
##            { "1f1":7,"1f2":"1testf2","1f3":True}],
##        "2":[
##            { "2c1r1":5,"2c2r1":"2c3r1testf2","2c3r1":True},
##            { "2f1":8,"2f2":"2testf2","2f3":True},
##            { "2f1":9,"2f2":"2testf2","2f3":True}],
##        }
##                   , "key2":
##        {                    
##         "3":[
##            {"3test":"3test"}],
##         "4":[
##            {"4test":444}] 
##
##        }
##    }
##
##
##
##
##print(testgroups.keys())
##print(testgroups.values())
##print(testgroups.items())

#print(testgroups.index(5))
##
##
##print("key2=",testgroups["key2"])
##print("key2,3=",testgroups["key2"]["3"])
##print("key1, 2=",testgroups["key1"]["2"])
##print("key2,4=",testgroups["key2"]["4"][0])
##
###for item in testgroups["key1"]:
###    print("key1,2?=",item.get("2"))
##
###print(list(testgroups.keys())[list(testgroups.values())])    #.index(5)]) 
##
##
##for item in testgroups["key1"]["1"]:
##    print("key1, 1, 1f2=",item.get("1f2"))
##
##print(list(testgroups["key2"]))
##print(list(testgroups["key2"]["3"]))
####print(list(testgroups["key1"]["2"]))
##print(list(testgroups["key2"]["4"][0]))
##

##
##
##datastore = { "office": {
##    "medical": [
##      { "room-number": 100,
##        "use": "reception",
##        "sq-ft": 50,
##        "price": 75
##      },
##      { "room-number": 101,
##        "use": "waiting",
##        "sq-ft": 250,
##        "price": 75
##      },
##      { "room-number": 102,
##        "use": "examination",
##        "sq-ft": 125,
##        "price": 150
##      },
##      { "room-number": 103,
##        "use": "examination",
##        "sq-ft": 125,
##        "price": 150
##      },
##      { "room-number": 104,
##        "use": "office",
##        "sq-ft": 150,
##        "price": 100
##      }
##    ],
##    "parking": {
##      "location": "premium",
##      "style": "covered",
##      "price": 750
##    }
##  }
##}
##
##print(datastore)
##print(datastore["office"]["parking"])
##
##print(datastore["office"]["medical"][1])
##
##print(datastore["office"].get("law"))  
##spaces = datastore['office']['medical']
##print("spaces=",spaces)
### Here is a method to find and change a value in the database.
##for item in spaces:
##    if item.get('use') == "examination" :
##       item['price'] = 175
##
##for item in datastore['office']['medical']: # This loop shows the change is not only in books, but is also in database
##    if item.get('use') == "examination" :
##        print('The {} rooms now cost {}'.format(item.get("use"), item.get("price")))
##
##people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},
##          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'}}
##
##print(people[1]['name'])
##print(people[1]['age'])
##print(people[1]['sex'])
##
###print(list(mydict.keys())[list(mydict.values()).index(19)]) # Prints george


#  Basically, it separates the dictionary's values in a list, finds the position of the value you have, and gets the key at that position.

#  More about keys() and .values() in Python 3: Python: simplest way to get list of values from dict?





#import nltk
#nltk.download()
#
##from nltk.stem.porter import PorterStemmer
##porter_stemmer=PorterStemmer()
##
##print(porter_stemmer.stem("machines"))
##print(porter_stemmer.stem("learning"))
##
##
##from nltk.stem import WordNetLemmatizer
##lemmatizer=WordNetLemmatizer()
####
####print(lemmatizer.lemmatize("machines"))
####print(lemmatizer.lemmatize("learning"))
####
##from sklearn.feature_extraction.text import CountVectorizer
##from sklearn.datasets import fetch_20newsgroups
##from nltk.corpus import names
###from nltk.corpus import shakespeare as sp
####
##import seaborn as sns
##import matplotlib.pyplot as plt
##import numpy as np    
####
##from nltk.stem import WordNetLemmatizer
##from sklearn.cluster import KMeans
##from sklearn.decomposition import NMF
####
####
####
####
##def letters_only(astr):
##    return(astr.isalpha())
####
####
#####transformed=cv.fit_transform(groups.data)
######print(cv.get_feature_names())
####cv=CountVectorizer(stop_words="english",max_features=500)
##groups=fetch_20newsgroups()
##
##
###print(sp.fileids())
###macbeth = sp.words("macbeth.xml")
###print(shakespeare_macbeth)
###print(sp["target_names"])
##
#print(groups.keys())
####print(groups["target_names"])
#print(groups.target)
####print(np.unique(groups.target))
####print(groups.data[0])
####print(groups.target[0])
####print(groups.target_names[groups.target[0]])
####print(len(groups.data[0]))
####print(len(groups.data[1]))
####sns.distplot(groups.target)
####plt.show()
##
##
##
##
##
##
####sns.distplot(np.log(transformed.toarray().sum(axis=0)))
####plt.xlabel("log count")
####plt.ylabel("frequency")
####plt.title("distribution plot of 500 word sounts")
####plt.show()
##
##cleaned=[]
##
##all_names=set(names.words())
###all_names=set(sp.words("macbeth.xml"))
###print(all_names)
##lemmatizer=WordNetLemmatizer()
##
##
##for post in groups.data:   #all_names:  #groups.data:
##    cleaned.append(" ".join([lemmatizer.lemmatize(word.lower()) for word in post.split() if letters_only(word) and word not in all_names]))
## #   if letters_only(post):
##  #      cleaned.append(" ".join([lemmatizer.lemmatize(post.lower())]))   # if letters_only(post)]))
##
###print(cleaned)
##transformed=cv.fit_transform(cleaned)
###print(cv.get_feature_names())
###km=KMeans(n_clusters=20)
###km.fit(transformed)
##nmf=NMF(n_components=100,random_state=43).fit(transformed)
##for topic_idx, topic in enumerate(nmf.components_):
##    label="{}: ".format(topic_idx)
##    print(label," ".join([cv.get_feature_names()[i] for i in topic.argsort()[:-9:-1]]))
##    
####labels=groups.target
####plt.scatter(labels,km.labels_)
####plt.xlabel("Newsgroup")
####plt.ylabel("Cluster")
####plt.show()
##
