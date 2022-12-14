﻿---
author: Jupiter Yu (@JupiterXiaoxiaoYu)
---

# Algorand-NFT-Similarity-Service-In-Marketplace

## Overview
In this project, I created an NFT Similarity Service In Marketplace targeting NFT on Algorand (the current API is for AlgoSeas PIRATEs), it contains:
1. Two databases, one for listings data and the other for sales data
2. Clustering algorithm which gets N similar NFT from databases
3. Machine Learning algorithm which predicts the price of query NFT through all historical sales data
4. A primary service endpoint that handles other services mentioned above

## Goals
1. Constantly update the data in the marketplace and store them for usage
2. Build a similarity service that given a query NFT asset and a number N, returns N most similar NFT data to the query NFT.
3. The similarity service should support and use both sales and active listing data, which means N active listings and N sales data should be returned for a single query. 
4. The similarity service should also return a predicted price based on similarity NFT data or historical sales data.

## Demo Query
Here is the demo query, you can enter the query NFT's asset id as well as a number of similar NFTs you wanna compare, the similarity service will return the result for you.
```
======Data Updated======
Please enter the asset id: 916123298
Please enter the number of similar NFT you want: 5

===============Query asset id 916123298 for 5 similar sales and listings================
The predicted price of query asset through all sales data (RandomForest):  [3.54622636] Algo
Got 5 similar assets' ids of previous sales:
['848209127', '706062256', '879144220', '772011509', '755306048']
Predicted price of query asset through these 5 sales data (Weighted_KNN):  1.6 Algo
See Following Details (Sales Information):
╒════╤════════════╤═════════════════════╤══════════════════════════╤════════════════════╕
│    │  Asset_ID  │  Sales_Algo_Amount  │   Market_Activity_Date   │  Relative_Weights  │
╞════╪════════════╪═════════════════════╪══════════════════════════╪════════════════════╡
│ 0  │ 848209127  │      2.5 Algo       │ 2022-08-23T21:46:01.000Z │         1          │
├────┼────────────┼─────────────────────┼──────────────────────────┼────────────────────┤
│ 1  │ 706062256  │      0.4 Algo       │ 2022-05-04T17:45:26.000Z │      0.901076      │
├────┼────────────┼─────────────────────┼──────────────────────────┼────────────────────┤
│ 2  │ 879144220  │      2.0 Algo       │ 2022-10-31T03:37:43.000Z │      0.894343      │
├────┼────────────┼─────────────────────┼──────────────────────────┼────────────────────┤
│ 3  │ 772011509  │      6.9 Algo       │ 2022-07-28T23:56:44.000Z │      0.891781      │
├────┼────────────┼─────────────────────┼──────────────────────────┼────────────────────┤
│ 4  │ 772011509  │      2.75 Algo      │ 2022-07-24T05:33:56.000Z │      0.891781      │
├────┼────────────┼─────────────────────┼──────────────────────────┼────────────────────┤
│ 5  │ 772011509  │      2.5 Algo       │ 2022-07-08T06:47:03.000Z │      0.891781      │
├────┼────────────┼─────────────────────┼──────────────────────────┼────────────────────┤
│ 6  │ 772011509  │      0.25 Algo      │ 2022-06-10T09:00:24.000Z │      0.891781      │
├────┼────────────┼─────────────────────┼──────────────────────────┼────────────────────┤
│ 7  │ 755306048  │      0.5 Algo       │ 2022-06-07T06:21:50.000Z │      0.876381      │
╘════╧════════════╧═════════════════════╧══════════════════════════╧════════════════════╛
Got 5 similar assets' ids of active listings:
['797926329', '698945727', '747920460', '798087604', '879144920']
Predicted price of query asset through these 5 listings data (Weighted_KNN):  20.59 Algo
See Following Details: (Listings Information)
╒════╤════════════╤═══════════════════════╤══════════════════════════╤════════════════════╕
│    │  Asset_ID  │  Listing_Algo_Amount  │   Market_Activity_Date   │  Relative_Weights  │
╞════╪════════════╪═══════════════════════╪══════════════════════════╪════════════════════╡
│ 0  │ 797926329  │       33.0 Algo       │ 2022-09-05T13:15:41.000Z │         1          │
├────┼────────────┼───────────────────────┼──────────────────────────┼────────────────────┤
│ 1  │ 698945727  │       50.0 Algo       │ 2022-05-01T19:06:23.000Z │      0.930642      │
├────┼────────────┼───────────────────────┼──────────────────────────┼────────────────────┤
│ 2  │ 747920460  │       10.0 Algo       │ 2022-07-22T08:05:02.000Z │      0.913222      │
├────┼────────────┼───────────────────────┼──────────────────────────┼────────────────────┤
│ 3  │ 798087604  │       2.0 Algo        │ 2022-10-21T18:33:17.000Z │      0.832746      │
├────┼────────────┼───────────────────────┼──────────────────────────┼────────────────────┤
│ 4  │ 879144920  │       2.5 Algo        │ 2022-10-31T03:03:29.000Z │      0.809556      │
╘════╧════════════╧═══════════════════════╧══════════════════════════╧════════════════════╛
===========================================================================================
```

## Configuration
After clone this repo to your side, you may need some extra steps to run this implementation

### MongoDB setup
Please follow the instruction and configure MongoDB:
https://www.w3schools.com/python/python_mongodb_getstarted.asp

MongoDB Compass is suggested to install as it provides nice visualization of data

### Install dependencies
Install all the required dependencies of python （my python version is 3.9):
pip install -r requirement.txt

## Usage

### Data collection
In order to collect data for usage, you should first run `NFTdata_collection.py`, this will store all of the active listing data and all historical sales data to the databases through the `init_data()` method, this is a one-time setup and it takes some time to fetch and store all of the data (about 5000 items). If you don't stop the process, the data will continue to update every minute. 

### Model Training and Predictions
In this implementation, two algorithms are provided to catch the similarities of NFT metadata better: 

1. K-Nearest Neighbor (KNN): this algorithm is used to find data of N most similar NFTs given an NFT, it also provides the predicted price of the query NFT from the weighted prices of returned N similar NFTs. 

#### Relative weights
In KNN, the weights of returned similar NFT are computed using Euclid Distance. KNN computes all distances between the query NFT and other NFT data, After that, the distances are normalized, and the NFTs corresponding to N smallest distances are selected to be the N most similar NFT. The reciprocals of those selected distances are relative weights that describe the similarity regarding the specific similar NFT and the query NFT compared to the similarity of query NFT of all other NFTs. The smaller the distance, the bigger the reciprocals, i.e. the relative weights are bigger.

#### Price prediction
The N relative weights describe how similar the N NFT is to the query NFT, it also describes how much should those N NFTs contributes to the prediction of the price of the query NFT. To predict the price, the sum of the N relative weights is scaled to 1, thus every relative weight is scaled to 0~1. The predicted price is computed by multiplying each relative weight by the corresponding NFT price and then summing those weighted prices.

2. Random Forest Regression (RFR): this algorithm aims to improve price prediction through all historical sales data. To see the details of RFR, please refer to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">sklearn</a>. The parameters of this RFR model have been selected by <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a> to improve the performance of the model. The properties for training are also selected based on feature importance analysis (see <a href="./importance.png">figure 1</a>). In addition, The model is actively training when the sales data is updated, this ensures it fits the need of the ever-changing NFT market. 

![figure 1](./importance.png)

### Query
To send a query, you need to run `NFTsimilarity_service.py`. Every time you start this service, it will init and train the RFR model using the latest data. After the initialization, you can send the query by inputting the asset id and the number of similar NFTs you wanna the service to return, you will get: 

According to sales data:
1. A predicted price of the query NFT by RFR based on all historical sales data
2. The asset ids of N similar sold NFTs 
3. A predicted price of the query NFT by KNN based on the weights of N similar sold NFT
4. A detailed table contains all the market activities of these N similar sold NFTs, each row including the asset id, Algo amount of the sale, date of the sale, and the relative weight of the asset.

According to active listings data:
1. The asset ids of N similar active listing NFTs 
2. A predicted price of the query NFT by KNN based on the weights of N similar active listing NFT
3. A detailed table contains all the market activities of these N similar active listing NFTs, each row including the asset id, Algo amount of the sale, date of the listing, and the relative weight of the asset.

## Demo Vedio
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/5CiFzqUpkKM/0.jpg)](https://www.youtube.com/watch?v=5CiFzqUpkKM)

## Help 
Please contact jupiterxiaoxiaoyu@gmail.com for more help.
