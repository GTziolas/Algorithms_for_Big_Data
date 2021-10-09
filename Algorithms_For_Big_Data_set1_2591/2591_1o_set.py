import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from operator import itemgetter
import timeit

start = timeit.default_timer()

data = np.genfromtxt('ratings_100users.csv', delimiter=',', names=True)

userList = {}
movieMap = {}
movieList = {}
i = 0
for index, values in np.ndenumerate(data):
    
    if values[0] not in userList.keys():
        userList[int(values[0])] = []
    userList[values[0]].append(int(values[1]))
    
    if values[1] not in movieMap.values():
        movieMap[i] = int(values[1])
        i += 1 
        
    if values[1] not in movieList.keys():
        movieList[int(values[1])] = []
    
    movieList[int(values[1])].append(int(values[0]))
    
movie_map = {v: k for k, v in movieMap.items()}
sorted_movieMap = sorted(movie_map.keys())

def create_random_hash_function(p=2**33-355, m=2**32-1):
    a = random.randint(1,p-1)
    b = random.randint(0, p-1)
    return lambda x: 1 + (((a * x + b) % p) % m)

def jaccardSimilarity(movieId1,movieId2):
    s1 = set( movieList[movieId1] )
    s2 = set( movieList[movieId2] )
    result = ( len(s1.intersection(s2)) / len(s1.union(s2)) )
    return (result)
	
def create_random_permutation(K):

    myHashFunction = create_random_hash_function()

    hashList = []             # stores pairs (i , H(i) )...
    randomPermutation = []    # stores the permutation of [1,2,...,K]

    for i in range(0,K):
        j=int(myHashFunction(i))
        hashList.append( (i,j) )
        
    # sort the hashList by second argument of the pairs...
    sortedHashList = sorted( hashList, key=itemgetter(1) )

    for i in range(0,len(sortedHashList)):
        randomPermutation.append(1 + sortedHashList[i][0])
    return randomPermutation


def minHash(n):
    SIG = []
    per = []
    for i in range(0,n): 
        per.append(create_random_permutation(len(userList)))
    
    for i in range(0,n):
        array = []
        for col in movieMap:
            array.append(sys.maxsize)
        SIG.append(array)
           
    for column in movieMap: 
        for row in userList:  
            for movie in userList[row]: 
                if (movie == sorted_movieMap[column]):   #check with movieId order
                    for i in range(0,n):
                        if per[i][row-1] < SIG[i][column]:
                            SIG[i][column] = per[i][row-1]
           
    return SIG      
				
def signatureSimilarity(movieId1,movieId2,n):    
    for value in range(len(sorted_movieMap)):  
        if sorted_movieMap[value] == movieId1 :
            movieId1 = value
        if sorted_movieMap[value] == movieId2 :
            movieId2 = value
            
    sim_count = 0
    for i in range(0,n):
        if (sig[i][movieId1]  == sig[i][movieId2] ):
            sim_count += 1   
        
    return sim_count / n


def LSH(sig,n,bands,rows):
    
    hashFunc = create_random_hash_function(610)
      
    pairs = []
    
    bandSize = int(n / bands)

    
    for b in range(0,n,bandSize):
        bin = {}
        for movieId in range(20):
            mylist = []
            movie = sorted_movieMap[movieId]
            for r  in  range (b,b+rows):
                if sig[r][movie] < 10 :  #if number has 1 digit form 
                    mylist.append("0"+str(sig[r][movieId]))
                else:
                    mylist.append(str(sig[r][movieId]))
                
            subSigNumber = int(''.join(map(str,mylist)))
            hashingResult = int(hashFunc(subSigNumber))
            
            if hashingResult not in bin.keys():
                bin[hashingResult] = []
            else:
                for i in bin[hashingResult]: 
                    tempArray = []
                    tempArray.append(i)
                    tempArray.append(movie)
                    pairs.append(tempArray)
            bin[hashingResult].append(movie)  
            
    return pairs

s = 0.25 
totalPairsSize = 20

sig = minHash(40)

sig = np.asarray(sig)
np.savetxt("SIG.csv", sig, fmt='%i', delimiter=",")


relativeElements = 0
notCalculatedRelativeElements = True
for n in range(5,45,5):
   
    falsePositives = 0
    falseNegatives = 0
    
    for i in range (totalPairsSize):
        for j in range (1,totalPairsSize-i):
            sigSim = signatureSimilarity(sorted_movieMap[i],sorted_movieMap[j+i], n)
            jacSim = jaccardSimilarity(sorted_movieMap[i],sorted_movieMap[j+i])
            
            if (sigSim >= s and jacSim < s):
                falsePositives += 1
            if (sigSim < s and jacSim >= s):
                falseNegatives += 1
            if (notCalculatedRelativeElements): 
                
                if(jaccardSimilarity(sorted_movieMap[i],sorted_movieMap[j+i]) >= s):
                    relativeElements += 1
    
    notCalculatedRelativeElements = False
      
    truePositives = relativeElements - falseNegatives              
    
    
    precision = truePositives / ( truePositives + falsePositives )

    recall = truePositives / ( truePositives + falseNegatives )

    F1 = 2 * recall * precision / ( recall + precision )
    
    
    print("false_positives: ", falsePositives)
    print("false_negatives: ", falseNegatives)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    print()
     
    left = [1, 2, 3] 
       
    values = [precision,recall,F1] 

    tick_label = ['Precision', 'Recall', 'F1'] 
    plt.bar(left, values, tick_label = tick_label, 
            width = 0.8, color = ['blue', 'cyan','green',]) 
    plt.title("n = " + str(n)) 
    #plt.show() 
    plt.savefig('n =' + str(n) + '.png')

s = 0.25
totalPairsSize = 20
n = 40 

testListNumbers = [2,4,5,8,10,20]

for r in testListNumbers:
    b = int(n/r)
    falsePositives = 0
    falseNegatives = 0
    
    similarCouples = LSH(sig,n,b,r)
    
    for i in similarCouples:
            
            sigSim = signatureSimilarity(sorted_movieMap[i[0]],sorted_movieMap[i[1]], n)
            jacSim = jaccardSimilarity(sorted_movieMap[i[0]],sorted_movieMap[i[1]])
            
            if (sigSim < s and jacSim >= s):
                falseNegatives += 1
            if (sigSim >= s and jacSim < s):
                falsePositives += 1
            
    truePositives = relativeElements - falseNegatives              

    precision = truePositives / ( truePositives + falsePositives )

    recall = truePositives / ( truePositives + falseNegatives )

    F1 = 2 * recall * precision / ( recall + precision )
    
    print("false_positives: ", falsePositives)
    print("false_negatives: ", falseNegatives)
    print("( b: ",b,", r: ", r ,")")
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    print()
    
    left = [1, 2, 3] 
    
    values = [precision,recall,F1] 

    tick_label = ['Precision', 'Recall', 'F1'] 

    plt.bar(left, values, tick_label = tick_label, 
            width = 0.8, color = ['blue','red','black']) 

    title = "( r = "+ str(r) + ", b = "+ str(b) + ")"
    
    plt.title(title) 

    #plt.show() 
    
    plt.savefig( title + '.png')
    

stop = timeit.default_timer()
print('The whole process took: {}s'.format(stop-start))        
	







