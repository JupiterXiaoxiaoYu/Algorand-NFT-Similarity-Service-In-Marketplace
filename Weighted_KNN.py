import numpy as np
from sklearn.preprocessing import LabelEncoder
dep_var = 'salesMicroAlgoAmount'

def gaussian(dist, sigma = 10.0):
    """ Input a distance and return it`s weight"""
    weight = np.exp(-dist**2/(2*sigma**2))
    return weight
 
def preprocess(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    x = (x - np.mean(x)) / np.std(x)
    return x

def MaxMinNormalization(x):
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def weighted_regression(input, dataSet, k):
    k = int(k)
    data = dataSet
    for col in list(dataSet.columns):
        if col not in list(input.columns):
            dataSet = dataSet.drop([col],axis=1)
    preprocess(dataSet)
    preprocess(input)
    dataSize = dataSet.shape[0]
    diff = np.tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff**2
    squareDist = sqdiff.sum(axis=1)
    dist = squareDist**0.5
    sortedDistIndex = np.argsort(dist)
    # print(sortedDistIndex)
    dist = [1/d for d in dist]
    dist = MaxMinNormalization((dist))
 
    asset_ids = []
    # weights = [(-(2/k**2)*(x-1)+2/k -(2/k**2)*x + 2/k)/2 for x in range(1,k+1) ]
    weights = []
    prices = []
    i = 0
    predict_price= 0
    while i<k:
        index = sortedDistIndex[i]
        id = data.iloc[index]['asset_id']
        if id in asset_ids:
            k+=1
            i+=1
            continue
        weight = dist[index]
        price = data.iloc[index][dep_var]
        prices.append(price)
        asset_ids.append(id)
        weights.append(weight)
        i+=1

    times = 1/sum(weights)
    for i in range(len(prices)):
        predict_price+= prices[i] * weights[i] * times

    # print(np.array(MaxMinNormalization(ZscoreNormalization(weights))))
        #print(index, dist[index],weight)
    return asset_ids, round(predict_price/1000000,2), weights
