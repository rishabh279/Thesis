# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:07:25 2017

@author: rishabh
"""
'''
import pandas as pd
import numpy as np

dataFile='E:/Dissertation/Implementation/Total/ml-100k/u.data'
data=pd.read_csv(dataFile,sep='\t',header=None,names=['userId','itemId','rating','timestamp'])

data.head()

movieInfoFile='E:/Dissertation/Implementation/Total/ml-100k/u.item'
movieInfo=pd.read_csv(movieInfoFile,sep="|",header=None,index_col=False,names=['itemId','title'],usecols=[0,1],encoding="ISO-8859-1")

movieInfo.head()

data=pd.merge(data,movieInfo,on="itemId")

data.head()

userIds=data.head()

userIds2=data[['userId']]

userIds.head()

userIds2.head()

data.loc[0:10,['userId']]

data[data.title=="Toy Story (1995)"].head()

data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])

numUsers=max(data.userId)
numMovies=max(data.itemId)

moviesPerUser=data.userId.value_counts()
usersPerMovie=data.itemId.value_counts()

def favoriteMovies(activeUser,N):
    topMovies=pd.DataFrame.sort_values(
                data[data.userId==activeUser],['rating'],ascending=[0])[:N]
    
    return list(topMovies.title)
    
#print(favoriteMovies(5,3))    

userItemRatingMatrix=pd.pivot_table(data,values='rating',index=['userId'],columns=['itemId'])

userItemRatingMatrix.head()

from scipy.spatial.distance import correlation
def similarity(user1,user2):
    user1=np.array(user1)-np.nanmean(user1)
    
    user2=np.array(user2)-np.nanmean(user2)
    
    commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]

    if len(commonItemIds)==0:
      return 0;
    else:
      user1=np.array([user1[i] for i in commonItemIds])
      user2=np.array([user2[i] for i in commonItemIds])
      return correlation(user1,user2)
#print(userItemRatingMatrix.loc[1])
#print(np.nanmean(userItemRatingMatrix.loc[5]))      
def nearestNeighbourRatings(activeUser,k):
    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=['Similarity'])
    
    for i in userItemRatingMatrix.index:
        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
        
    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
    
    nearestNeighbours=similarityMatrix[:k]
    
    #print(similarityMatrix[:k].userId)  
    
    #print(data.loc[data.userId==434])
    neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
    
    predictItemRatings=pd.DataFrame(index=userItemRatingMatrix.columns,columns=['Ratings'])
    
    for i in userItemRatingMatrix.columns:
        
        predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
        
        for j in neighbourItemRatings.index:
            
            if userItemRatingMatrix.loc[j,i]>0:
                
                predictedRating +=(userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                                   
        predictItemRatings.loc[i,'Rating'] =predictedRating
        
    return predictItemRatings    

def topNRecommendations(activeUser,N):
    
    predictItemRating=nearestNeighbourRatings(activeUser,10)
    moviesAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
    predictItemRating=predictItemRating.drop(moviesAlreadyWatched)
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:N]
                                                
    topRecommendationTitles=(movieInfo.loc[movieInfo.itemId.isin(topRecommendations.index)]) 
    return list(topRecommendationTitles.title)  

activeUser=5
#print(favoriteMovies(activeUser,2),"\nPredicted:-",topNRecommendations(activeUser,4))                                         
data.loc[data.title=='Truth About Cats & Dogs, The (1996)']

#------------------------------------Matrix Factorization---------------------------------------

def matrixFactorization(R,K,steps=10,gamma=0.001,lamda=0.002):    
    
    N=len(R.index)
    M=len(R.columns)
    
    P=pd.DataFrame(np.random.rand(N,K),index=R.index)
    Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
    
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                   
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[i]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
                    
        e=0
        for i in R.index:
            for j in R.columns: 
                if R.loc[i,j]>0:
                    e=e+pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        print(step)
    return P,Q         

(P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02,steps=1)   

activeUser=1
predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]
topRecommendationsTitles=movieInfo.loc[movieInfo.itemId.isin(topRecommendations.index)]
print(list(topRecommendationsTitles.title))                                
'''

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:07:25 2017

@author: rishabh
"""

import pandas as pd
import numpy as np

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataFile='E:/Dissertation/Implementation/Total/ml-100k/u.data'
data=pd.read_csv(dataFile,sep='\t',header=None,names=['userId','itemId','rating','timestamp'])

data.head()

movieInfoFile='E:/Dissertation/Implementation/Total/ml-100k/u.item'
movieInfo=pd.read_csv(movieInfoFile,sep="|",header=None,index_col=False,names=['itemId','title'],usecols=[0,1],encoding="ISO-8859-1")

movieInfo.head()

data=pd.merge(data,movieInfo,on="itemId")

data.head()

userIds=data.head()

userIds2=data[['userId']]

userIds.head()

userIds2.head()

data.loc[0:10,['userId']]

data[data.title=="Toy Story (1995)"].head()

data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])

numUsers=max(data.userId)
numMovies=max(data.itemId)

moviesPerUser=data.userId.value_counts()
usersPerMovie=data.itemId.value_counts()

def favoriteMovies(activeUser,N):
    topMovies=pd.DataFrame.sort_values(
                data[data.userId==activeUser],['rating'],ascending=[0])[:N]
    
    return list(topMovies.title)
    
#print(favoriteMovies(5,3))    

userItemRatingMatrix=pd.pivot_table(data,values='rating',index=['userId'],columns=['itemId'])

userItemRatingMatrix.head()
print(np.array(userItemRatingMatrix.loc[5]))
print(np.nanmean(userItemRatingMatrix.loc[5]))
from scipy.spatial.distance import correlation
def similarity(user1,user2):
    user1=np.array(user1)-np.nanmean(user1)
    
    user2=np.array(user2)-np.nanmean(user2)
    
    commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]

    if len(commonItemIds)==0:
      return 0;
    else:
      user1=np.array([user1[i] for i in commonItemIds])
      user2=np.array([user2[i] for i in commonItemIds])
      return correlation(user1,user2)
      
def nearestNeighbourRatings(activeUser,k):
    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=['Similarity'])
    
    for i in userItemRatingMatrix.index:
        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
        
    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
    
    nearestNeighbours=similarityMatrix[:k]
    #similarityMatrix[:k].to_csv('C:/Users/rishabh/Desktop/Verification/myfile.csv')
    print(nearestNeighbours)  

    neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
    
    predictItemRatings=pd.DataFrame(index=userItemRatingMatrix.columns,columns=['Ratings'])
    
    for i in userItemRatingMatrix.columns:
        
        predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
        
        for j in neighbourItemRatings.index:
            
            if userItemRatingMatrix.loc[j,i]>0:
                
                predictedRating +=(userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                                   
        predictItemRatings.loc[i,'Rating'] =predictedRating
        
    return predictItemRatings    

def topNRecommendations(activeUser,N):
    
    predictItemRating=nearestNeighbourRatings(activeUser,10)
    moviesAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]
                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)
    predictItemRating=predictItemRating.drop(moviesAlreadyWatched)
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:N]
                                                
    topRecommendationTitles=(movieInfo.loc[movieInfo.itemId.isin(topRecommendations.index)]) 
    
    #print(list(topRecommendationTitles.columns.values))
    index=list(topRecommendationTitles.itemId)
    for i in index:
        print(data.loc[data.itemId==i])
        #data.loc[data.itemId==i].to_csv('C:/Users/rishabh/Desktop/Verification/myfile'+str(i)+'.csv')
    #print(data.loc[data.itemId==111])
    #print(topRecommendationTitles.itemId)
    #topRecommendationTitles.title.to_csv('C:/Users/rishabh/Desktop/Verification/myfile2.csv')
    return list(topRecommendationTitles.title)  

activeUser=5
#print(favoriteMovies(activeUser,2),"\nPredicted:-",topNRecommendations(activeUser,4))                                         
#data.loc[data.title=='Truth About Cats & Dogs, The (1996)']

def matrixFactorization(R,K,steps=10,gamma=0.001,lamda=0.002):    
    
    N=len(R.index)
    M=len(R.columns)
    
    P=pd.DataFrame(np.random.rand(N,K),index=R.index)
    Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
    
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                   
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[i]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
                    
        e=0
        for i in R.index:
            for j in R.columns: 
                if R.loc[i,j]>0:
                    e=e+pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        print(step)
    return P,Q         

(P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02,steps=100)   

activeUser=5
predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]
topRecommendationsTitles=movieInfo.loc[movieInfo.itemId.isin(topRecommendations.index)]
print(list(topRecommendationsTitles.title))                                
