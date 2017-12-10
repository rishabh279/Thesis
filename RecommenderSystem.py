# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:07:25 2017

@author: rishabh
"""

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
    
print(favoriteMovies(5,3))    

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
      
def nearestNeighbourRatings(activeUser,k):
    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=['Similarity'])
    
    for i in userItemRatingMatrix.index:
        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],
                                          userItemRatingMatrix.loc[i])
        
    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
    
    nearestNeighbours=similarityMatrix[:k]

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
print(favoriteMovies(activeUser,5),"\n",topNRecommendations(activeUser,3))                                         
        