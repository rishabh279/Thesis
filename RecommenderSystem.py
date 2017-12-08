# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:07:25 2017

@author: rishabh
"""

import pandas as pd
import numpy as np

dataFile='E:/Dissertation/Implementation/Total/u.data'
data=pd.read_csv(dataFile,sep='\t',header=None,names=['userId','itemId','rating','timestamp'])

data.head()

movieInfoFile='E:/Dissertation/Implementation/Total/u.item'
movieInfo=pd.read_csv(movieInfoFile,sep="|",header=None,index_col=False,names=['itemId','title'],usecols=[0,1],encoding="ISO-8859-1")

movieInfo.head()

