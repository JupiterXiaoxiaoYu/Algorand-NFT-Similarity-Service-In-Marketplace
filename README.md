---
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

2. Random Forest Regression (RFR): this algorithm aims to improve price prediction through all historical sales data. To see the details of RFR, please refer to <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">sklearn</a>. The parameters of this RFR model have been selected by <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a> to improve the performance of the model. The properties for training are also selected based on feature importance analysis (see figure 1(./importance.png)). In addition, The model is actively training when the sales data is updated, this ensures it fits the need of the ever-changing NFT market. 

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
![figure 1](./importance.png)


## Help 
Please contact jupiterxiaoxiaoyu@gmail.com for more help.