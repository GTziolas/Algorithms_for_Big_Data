import pandas as pd

df = pd.read_csv('ratings.csv', sep = ',', low_memory=False)

#---BEGIN create userList dictionary---#
# userList=df.groupby('userId')['movieId'].apply(list).reset_index(name='movies')
# userList_dict = userList.to_dict('records')

# for i in range(0,len(userList_dict),1):
#     for k, v in userList_dict[i].items():
#         print (k, ':', v)

# f = open("userList_dict.txt","w")
# f.write( str(userList_dict) )
# f.close()

#userList in a print friendly mode with tabbed columns
userList=df.groupby('userId')['movieId'].apply(list).reset_index(name='movies')
#print(userList)
userList.to_csv('userList.txt', sep='\t', index=False)

#---END---#

#---BEGIN create movieMap ---#
df2 = df.assign(id=(df['movieId']).astype('category').cat.codes+1)
movieMap = df2[['movieId', 'id']]
movieMap_dropped = movieMap.drop_duplicates()
movieMap_sorted = movieMap_dropped.sort_values(by = 'id')
movieMap_final = movieMap_sorted.reindex(columns = ['id', 'movieId'])

#movieMap_dict = movieMap_final.to_dict('records')
movieMap_final = movieMap_final.astype(int)
movieMap_final.to_csv('movieMap.txt', sep='\t', index=False)

#print(movieMap_dict)

#---END---#

#---BEGIN create movieList ---#
movieList = df.groupby('movieId')['userId'].apply(list).reset_index()
movieList.to_csv('movieList.txt', sep='\t', index=False)
#---END---#