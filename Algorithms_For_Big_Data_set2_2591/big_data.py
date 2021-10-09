#Giorgos Tziolas, 2591
import sys
import random
import pandas as pd
import numpy as np
import itertools
import keyboard
import ast
from collections import Counter,defaultdict
from operator import itemgetter
import networkx as nx
import matplotlib.pyplot as plt
import timeit

#Time the whole process
start = timeit.default_timer()
subsets=[]
userBaskets = []
s=[]
pairCounters = []
finalMyAprioriList=[]
setOfUsers=[]

def draw_graph(rules, rules_to_show):

    G1 = nx.DiGraph()
    
    color_map=[]
    N=100
    colors = np.random.rand(N)
    #first 10 rules
    strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']
    
    for i in range(rules_to_show):
        G1.add_nodes_from(['R'+str(i)])
        
        for a in rules.iloc[i]['hypothesis']:
            G1.add_nodes_from([a])
            G1.add_edge(a, 'R'+str(i), color = colors[i], weight=2)
            
        for c in rules['conclusion']:
            G1.add_nodes_from([c])
            G1.add_edge('R'+str(i), c, color=colors[i], weight=2)
            
    for node in G1:
        found_a_string=False
        for item in strs:
            if node == item:
                found_a_string=True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')

    edges = G1.edges()
    colors = [G1[u][v]['color'] for u,v in edges]
    weights = [G1[u][v]['weight'] for u,v in edges]

    pos = nx.spring_layout(G1, k=16, scale=1)
    nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            

    for p in pos:  # raise text positions
           pos[p][1] += 0.07
    nx.draw_networkx_labels(G1, pos)
    plt.show()

def presentResults(rules):
    print('Press a letter from the list below to display the corresponding results:')
    print("""===========================================================================
    (a) List ALL discovered rules (press 'a')
    (b) List all rules containing a BAG of movies (press 'b')
    (c) Compare Rules with Confidence (press 'c')
    (cc) Compare Rules with Lift (press 'cc')
    (h) Print the Histograms of CONFIDENCES|LIFTS among discovered rules (press 'h')
    (hh) Print the Scatter plot of CONFIDENCE vs LIFT (press 'hh')
    (m) Show details of a MOVIE (press 'm')
    (r) Show a particular RULE (press 'r')
    (sc) SORT rules by increasing CONFIDENCE (press 'sc')
    (sl) SORT rules by increasing LIFT (press 'sl')
    (v) VISUALIZATION of association rules (press 'v')
    (e) EXIT (press 'e')
    (All letters must be entered without the quotation marks '')
    ===========================================================================""")
    
    letter = input('Press a letter from the list above and ENTER, to display the corresponding results:\n')
    print('You pressed \'{}\''.format(letter))
    while(letter != 'e'):
        
        if (letter.lower() == 'e'):
            print('Exiting...')
            sys.exit()    
        elif (letter.lower() == 'a'):
            print(rules['rule'])
        elif (letter.lower() == 'b'):
            #Hypothesis
            h_input = input('Expecting an input hypothesis like: <number><whitespace><number>... and ENTER \n')
            h = h_input.split()
            for i in range(0, len(h)):
                h[i] = int(h[i])
            #Conclusion
            c_input = int(input('Expecting an input conclusion like: <number> and ENTER \n'))
            rule_count = 1
            for i in range(len(rules)):
                if (any(item in h for item in rules['hypothesis'][i])):
                    if (rules['conclusion'][i] == c_input):
                        print('Rule #{} : {}'.format(rule_count, rules['rule'][i]))
                        rule_count+=1
        
        ####C case
        elif (letter.lower() == 'c'):
            rules_df_confidence = rules['confidence'].value_counts(bins=10) 
            rules_df_confidence.plot()
            plt.xticks(rotation=90)
            plt.title('Confidence - Rules graph')
            plt.xlabel('Confidence')
            plt.ylabel('Rules')
            plt.legend()
            plt.show()
            plt.close()
        
        elif(letter.lower() == 'cc'):
            rules_df_lift = rules['lift'].value_counts(bins=10)
            rules_df_lift.plot()
            plt.xticks(rotation=90)
            plt.title('Lift - Rules graph')
            plt.xlabel('Lift')
            plt.ylabel('Rules')
            plt.legend()
            plt.show()
            plt.close()
        
        elif(letter.lower() == 'h'):
            fig, axes = plt.subplots(1,2)
            
            axes[0].set_title('CONFIDENCES in rules')
            axes[1].set_title('LIFTS in rules')
            
            rules['confidence'].plot.hist(bins=10, alpha=0.7, ax=axes[0])
            rules['lift'].plot.hist(bins=10, alpha=0.6, ax=axes[1], color='r')
            plt.show()
            plt.close()
            
        elif(letter.lower() == 'hh'):
            print(rules[['confidence', 'lift']])
            
            confidence = rules['confidence']
            lift = rules['lift']
            
            plt.title('CONFIDENCE vs LIFT')
            plt.xlabel('Lift')
            plt.ylabel('Confidence')
            plt.scatter(lift, confidence, alpha=0.7, marker='o', color='y')
            plt.show()
            plt.close()
                      
        elif(letter.lower() == 'm'):
            movies_df = pd.read_csv('movies.csv', sep=',')
            mID_int = int(input('Enter a movie ID (integers only): '))
            
            print(movies_df.iloc[mID_int-1])
        
        elif(letter.lower() == 'r'):
            rule_id = int(input('Enter a rule ID (integers only): '))
            print('Rule #{} is: {}'.format(rule_id, rules.iloc[rule_id-1].values[1]))
        
        elif(letter.lower() == 'sc'):
            rules = rules.sort_values(by=['confidence', 'ruleID'], ascending = False)
            print(rules[['confidence', 'ruleID']])
            
        elif(letter.lower() == 'sl'):
            rules = rules.sort_values(by=['lift','ruleID'], ascending = False)
            print(rules[['lift', 'ruleID']])
                
        elif(letter.lower() == 'v'):
            print('TODO V case')
            draw_graph(rules, 10)
        
        else:
            print('No proper input was given. Terminating...')
            sys.exit()
    
        letter = input('Press another letter or press \'e\' to exit\n')
        print('You pressed \'{}\''.format(letter))
    
#Create the baskets file and the usersBaskets_lists and return them  
def createMovieBaskets(file): 
    df = pd.read_csv(file, sep = ',')
    df = df.groupby('userId').movieId.apply(list).sort_values()
    df = df.sort_index()    
    df.to_csv('userBaskets.csv', sep=',')
    f = 'userBaskets.csv'
    df1 = pd.read_csv(f, sep=',')
    lista = np.array(df1['movieId'])
    return f, lista

def readMovies(file):
    df = pd.read_csv(file, sep=',')
    return df

#Create a file like ((pair), (number_of_occurences))
def hashedCountersOfPairs(lista):
    #the list is read as a whole string -> convert to int
    for i in range(0,len(lista)):
        userBaskets.append(ast.literal_eval(lista[i]))
    #find all the individual movie pairs that can be created from user baskets   
    for i in range(0,len(userBaskets)):
        b = findsubsets(userBaskets[i], 2)
        subsets.append(b) 
    #subsets is a list of lists -> convert it to a simple list so that
    #it becomes iterable and the Counter method can be used on it
    #to find the number of occurences of (x,y) pairs
    for sublist in subsets:
        for item in sublist:
            s.append(item)        
    c = Counter(s) 
    
    for x in c:
        key=x
        value = c[key]
        y = key[0], key[1] ,value
        pairCounters.append(y)
    
    with open('array.txt', 'w') as file:
        for line in pairCounters:
            file.write(''.join(str(line))+'\n')
            
    return userBaskets       

#Creating tuples, triples, quadruples etc is the process that takes
#most of the time                     
def findsubsets(s,m):
    return list(itertools.combinations(s,m))

#File locations
userRatings_file = 'ratings_shuffled.csv'
movies_file = 'movies.csv'
userBaskets_file, userBaskets_list = createMovieBaskets(userRatings_file)

#Create movies_df dataframe
movies_df = readMovies(movies_file)
#usr_bskts = hashedCountersOfPairs(userBaskets_list)      


def find_frequent_itemsets(usr_reviews, k_1_itemsets, min_support):
    counts = defaultdict(int)
    #iterate through each user
    for user, reviews in usr_reviews.items():
        #iterate through the previously discovered itemsets
        for itemset in k_1_itemsets:
            #if the previous is subset of the current
            if itemset.issubset(reviews):
                #go through each movie the user has reviewed and create
                #a superset combining the itemset with the new movie 
                #and record that we saw this superset in our counting dict
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])


def myApriori(file):
    all_ratings = pd.read_csv(file, sep=',')
    all_ratings=all_ratings.rename(columns={'userId':'UserID','movieId':'MovieID','rating':'Rating','timestamp':'Datetime'})
    all_ratings['Datetime'] = pd.to_datetime(all_ratings['Datetime'], unit='s')
    all_ratings['Appearances'] = all_ratings['Rating']>=0.0
    ratings = all_ratings
    favorable_ratings = ratings[ratings['Appearances']]

    #frozenset because faster search
    usr_reviews = dict((k, frozenset(v.values))
    for k, v in favorable_ratings.groupby("UserID")["MovieID"])
    
    movie_appearances = ratings[['MovieID', 'Appearances']].groupby('MovieID').sum()
        
    frequent_itemsets={}
    min_frequency = 0.45
    minlen = len(usr_reviews)
    min_support = min_frequency*minlen
    #frozenset because of faster set-based operations and
    #can be userd as keys in counting the dictionary
    
    frequent_itemsets[1] = dict((frozenset((movie_id,)), row['Appearances'])
                                for movie_id, row in movie_appearances.iterrows()
                                if row['Appearances'] >= min_support)
            
    print('Minimum frequency is: {}, sample size is: {}, minimum support is: {}'.format(min_frequency, minlen, int(min_support)))
    for k in range(2, 20):
      # Generate candidates of length k, using the frequent itemsets of length k-1
      # Only store the frequent itemsets
        cur_frequent_itemsets = find_frequent_itemsets(usr_reviews,frequent_itemsets[k-1], min_support)
        if len(cur_frequent_itemsets) == 0:
            print('Did not find any frequent itemsets of length {}. Stopping and creating rules...'.format(k))
            sys.stdout.flush()
            break
        else:
            print('Found {} frequent itemsets of length {}:'.format(len(cur_frequent_itemsets), k))
            sys.stdout.flush()
            frequent_itemsets[k] = cur_frequent_itemsets
            
    return frequent_itemsets,usr_reviews

def reservoirSampling(iterable, n):
    reservoir=[]
    for t,item in enumerate(iterable):
        if t<n:
            reservoir.append(item)
        else:
            m = random.randint(0,t)
            if m<n:
                reservoir[m]=item
    return reservoir

#SampledApriori
def sampling(file):        
    for index, row in file.iterrows():
        if keyboard.is_pressed('y') or keyboard.is_pressed('Y'):
            print('Pressed \'y\', stopping')
            print('Users in list: {}'.format(len(result)))
            break
        else:
            current_user = row['userId']
            if (current_user not in setOfUsers):
                setOfUsers.append(int(current_user))
                numberOfDistinctUsersSoFar = len(setOfUsers)
                ran = random.randint(0,numberOfDistinctUsersSoFar)###
                result = reservoirSampling(setOfUsers,ran)
    print('Users in list: {}'.format(len(result)))        
    return result


def associationRulesCreation(file,resultOfApriori,usr_reviews):
    candidate_rules = []
    for itemset_length, itemset_counts in resultOfApriori.items():
        for itemset in itemset_counts.keys():
            for conclusion in itemset:
                premise = itemset - set((conclusion,))
                candidate_rules.append((premise, conclusion))

    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)
    for user, reviews in usr_reviews.items():
        for candidate_rule in candidate_rules:
            premise, conclusion = candidate_rule
            if premise.issubset(reviews):
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1

    #divide correct count by the total number of times the rule was seen:
    #correct / (correct + incorrect)
    rule_confidence = {candidate_rule:
        (correct_counts[candidate_rule] / float(correct_counts[candidate_rule] +
        incorrect_counts[candidate_rule]))
        for candidate_rule in candidate_rules}
    
    df = pd.read_csv(file, sep=',')
    count = df['movieId'].value_counts()
    count_sorted = count.sort_index()  
    column_names = ['itemset','rule','hypothesis','conclusion','frequency','confidence','lift','interest','ruleID']               
    rules_df = pd.DataFrame(columns = column_names)
    min_confidence=0.5 ###CHANGE AS NEEDED
    sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)
    #Number of max rules to create
    rule_count = 100000
    for index in range(rule_count):
        #check min_confidence
        if ((rule_confidence[(premise, conclusion)]) >= min_confidence):
            #print("Rule #{0}".format(index + 1))
            premise, conclusion = sorted_confidence[index][0]
            
            #Optional prints for each rule found
            #print("Rule: If a user recommends {0} they will also recommend {1}".format(premise, conclusion))
            #print(" - Confidence: {:.6f}".format(rule_confidence[(premise, conclusion)]))
            #print(" - Lift: {:.6f}".format((rule_confidence[(premise, conclusion)]) / ((count_sorted.iloc[conclusion])/len(count_sorted))/100))
            #print(" - Interest: {:.6f}".format(((rule_confidence[(premise, conclusion)])-((count_sorted.iloc[conclusion])/len(count_sorted)))))
            
            rules_df = rules_df.append({'itemset':(premise, conclusion),
                                    'rule':'{} -->{}'.format(list(premise),conclusion),
                                    'hypothesis':premise, 
                                    'conclusion':conclusion,
                                    'frequency':(((count_sorted.iloc[conclusion])/len(count_sorted))),
                                    'confidence':rule_confidence[(premise, conclusion)],
                                    'lift':(((rule_confidence[(premise, conclusion)]) / ((count_sorted.iloc[conclusion])/len(count_sorted))/100)),
                                    'interest':((rule_confidence[(premise, conclusion)])-((count_sorted.iloc[conclusion])/len(count_sorted))),
                                    'ruleID':index+1}, ignore_index=True)
            
    #Write all rules found to an output file to be used for presentation later
    rules_df.to_csv('rules.csv', index=None, sep='\t', encoding='utf-8')
    return rules_df

#REMOVE THESE COMMENTS TO USE SAMPLED APRIORI
# =============================================================================
# ratings_stream = pd.read_csv('ratings_shuffled.csv', sep=',')
# smpl = sampling(ratings_stream)
# print('Users selected from sampling: {}'.format(sorted(smpl)))
# ratings_stream = ratings_stream[ratings_stream['userId'].isin(smpl)]
# ratings_stream.to_csv('sampled.csv', sep=',', index=None)
# a, usr_reviews = myApriori('sampled.csv')
# =============================================================================


a, usr_reviews = myApriori(userRatings_file) # COMMENT OUT THIS LINE TO USE SAMPLED APRIORI
 
b = associationRulesCreation(userRatings_file, a, usr_reviews)

print()
stop = timeit.default_timer()
print('Execution time = {:.3f}s \n'.format(stop-start))
presentResults(b)

























