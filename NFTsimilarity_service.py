
import time
import pandas as pd
import NFTdata_collection
import Weighted_KNN
import RandomForest
import asyncio

salesdata = NFTdata_collection.export_csv_for_ml()

dep_var = 'salesMicroAlgoAmount'

asset_data = []
    
def confirm_asset_id():
    asset_id = input('Please enter the asset id: ')
    n = input('Please enter the number of similar NFT you want: ')
    try:
        asset_data = NFTdata_collection.get_main_nft_data(asset_id)
        return asset_data, asset_id, n
    except:
        print('Wrong asset id or number')


async def get_n_nearest_asset(n, asset_data):
    sales_dataset = NFTdata_collection.export_csv_for_ml()
    listing_dataset = NFTdata_collection.expory_csv_for_listing()
    if listing_dataset.shape[1] != 31:
        print('Active listing data is updating, please wait')
    while listing_dataset.shape[1] != 31:
        listing_dataset = NFTdata_collection.expory_csv_for_listing()
    sales_assets, knn_predicted_price_sales, sales_weights = Weighted_KNN.weighted_regression(asset_data,sales_dataset,n)
    listing_assets, knn_predicted_price_listing, listing_weights = Weighted_KNN.weighted_regression(asset_data,listing_dataset,n)
    print(f'Got {n} similar assets\' ids of previous sales:\n{sales_assets}')
    print(f'Predicted price of query asset through these {n} sales data (Weighted_KNN):  {knn_predicted_price_sales} Algo')
    # RF_predicted = RandomForest.predict_price_accurate_from_previous_sales(asset_data)
    # print(f'Predicted price of query asset through all sales data (RandomForest):  {RF_predicted} Algo')
    dic_sales=dict(zip(sales_assets,sales_weights))
    print(f'See Following Details (Sales Information):')
    NFTdata_collection.get_n_from_sales(dic_sales)
    print(f'Got {n} similar assets\' ids of active listings:\n{listing_assets}')
    print(f'Predicted price of query asset through these {n} listings data (Weighted_KNN):  {knn_predicted_price_listing} Algo')
    dic_listing=dict(zip(listing_assets,listing_weights))
    print(f'See Following Details: (Listings Information)')
    NFTdata_collection.get_n_from_listing(dic_listing)


def initialization():
    RandomForest.all_train(NFTdata_collection.export_csv_for_ml())
    NFTdata_collection.updates_data()
    

if __name__ == "__main__":
    # RandomForest.all_train()
    # initialization()

    while True:
        # print(asset_data)
        try:
            asset_data, asset_id, number = confirm_asset_id()
            print(f'===============Query asset id {asset_id} for {number} similar sales and listings================')
            RandomForest.predict_price_accurate_from_previous_sales(asset_data)
            loop = asyncio.get_event_loop()
            loop.run_until_complete(get_n_nearest_asset(number, asset_data))
            print('===========================================================================================')
        except Exception as e:
            print(f'Try Again: {e}')

    