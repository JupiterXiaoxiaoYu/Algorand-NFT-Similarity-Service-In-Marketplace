import requests
import time
import json
import base64
import os,inspect
import asyncio
import pandas as pd 
import RandomForest
from tabulate import tabulate
from settings import Settings
from pymongo import MongoClient
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)

settings = Settings('active_listing')
myindexer = settings.get_indexer()

Active_Pirate_Listings_API = 'https://d3ohz23ah7.execute-api.us-west-2.amazonaws.com/prod/marketplace/listings?type=listing&sortBy=price&sortAscending=true&collectionName=AlgoSeas Pirates&limit={}'
Recent_Pirate_Sales_API = 'https://d3ohz23ah7.execute-api.us-west-2.amazonaws.com/prod/marketplace/sales?collectionName=AlgoSeas Pirates&sortBy=time&sortAscending=false&limit={}'
Sales_Asset_Metadata_API = 'https://algoindexer.algoexplorerapi.io/v2/assets/{}'

mongoClient = MongoClient() 
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client['algoseas_pirate']
listdata = db["listdata"]
salesdata = db['salesdata']

async def listing_data_async(asset):
    item = {'name': asset['assetInformation']['nName']}
    item['asset_id'] = asset['assetInformation']['SK']
    for k, v in sorted(asset['assetInformation']['nProps']['properties'].items()):
        if v == False:
            continue
        item[k] = v
    item['market_activity_date'] = asset['marketActivity']['creationDate']
    item['salesMicroAlgoAmount'] = asset['marketActivity']['listedAlgoAmount']
    try:
        # listdata.insert_one(item)
        listdata.update_one({"name":item["name"], 'market_activity_date': item['market_activity_date']},{'$set':item},True)
    except Exception as e:
        print(e, 'DB error')


async def get_listing_data(number):
    max_try = 2
    for tries in range(max_try):
        try:
            data = requests.get(Active_Pirate_Listings_API.format(number))
            content = data.json()
            tasks = []
            print(f'adding {len(content)} listing assets')
            for asset in content:
                tasks.append(listing_data_async(asset))
            await asyncio.gather(*tasks)
            break
        except Exception as e:
            print(e)
            if tries < (max_try - 1):
                time.sleep(2)
                continue
            else:
                print("crawl_error", "501")


def sales_data_async(asset, active_learning):
    item = {'name': asset['assetInformation']['nName']}
    item['asset_id'] = asset['assetInformation']['SK']

    asset_id = item['asset_id'] 
    last_config_tnx = myindexer.search_asset_transactions(asset_id,txn_type='acfg')['transactions'][-1]
    if 'note' in last_config_tnx:
        json_string = base64.b64decode(last_config_tnx['note']).decode('utf-8')
        json_data_as_dict = json.loads(json_string)
        for k, v in sorted(json_data_as_dict['properties'].items()):
            item[k] = v
        item['market_activity_date'] = asset['marketActivity']['creationDate']
        item['salesMicroAlgoAmount'] = asset['marketActivity']['algoAmount']
        try:
            if [i for i in salesdata.find(item)] == []:
                print(f"ASA ID {asset_id}: metadata found - adding.")
                active_learning.append(item)
                salesdata.update_one({"name":item["name"], 'market_activity_date': item['market_activity_date']},{'$set':item},True)
                return False
            else:
                return True
        except Exception:
            print(Exception, 'DB error')
    else:
        print(f"ASA ID {asset_id}: no metadata found.")


def get_sales_data(number):
    max_try = 2
    for tries in range(max_try):
        try:
            data = requests.get(Recent_Pirate_Sales_API.format(number))
            content = data.json()
            # tasks = []
            active_learning = []
            print(f'adding {len(content)} sold assets')
            for asset in content:
                flag = sales_data_async(asset, active_learning)
                if flag == True:
                    print('======Data Updated======')
                    break
            # await asyncio.gather(*tasks)
            if active_learning!=[]:
                RandomForest.active_learning(pd.DataFrame(active_learning))
            break
        except Exception as e:
            print(e)
            if tries < (max_try - 1):
                time.sleep(2)
                continue
            else:
                print("crawl_error", "501")

async def init_sales_data(number):
    max_try = 2
    for tries in range(max_try):
        try:
            data = requests.get(Recent_Pirate_Sales_API.format(number))
            content = data.json()
            tasks = []
            active_learning = []
            print(f'adding {len(content)} sold assets')
            for asset in content:
                tasks.append(sales_data_async(asset, active_learning))
            await asyncio.gather(*tasks)
            RandomForest.active_learning(pd.DataFrame(active_learning))
            break
        except Exception as e:
            print(e)
            if tries < (max_try - 1):
                time.sleep(2)
                continue
            else:
                print("crawl_error", "501")

def export_csv_for_ml():
    sales_data_csv = pd.DataFrame(list(salesdata.find()))
    # print(sales_data_csv.head(), len(sales_data_csv))
    # sales_data_csv.to_csv('salesdata.csv', encoding='utf-8')
    return sales_data_csv

def expory_csv_for_listing():
    sales_data_csv = pd.DataFrame(list(listdata.find()))
    # print(sales_data_csv.head(), len(sales_data_csv))
    # sales_data_csv.to_csv('salesdata.csv', encoding='utf-8')
    return sales_data_csv

def get_main_nft_data(asset_id):
    # info = myindexer.asset_info(asset_id) 
    # print(info)
    asset = {}
    # asset['name'] = info['name']
    last_config_tnx = myindexer.search_asset_transactions(asset_id,txn_type='acfg')['transactions'][-1]
    if 'note' in last_config_tnx:
        print(f"ASA ID {asset_id}: metadata found.\n")
        json_string = base64.b64decode(last_config_tnx['note']).decode('utf-8')
        json_data_as_dict = json.loads(json_string)
        for k, v in sorted(json_data_as_dict['properties'].items()):
            asset[k] = v
    return pd.DataFrame([asset])

def get_n_from_sales(asset_ids):
    d = []
    for asset in asset_ids.keys():
        d = d+ list(salesdata.find({'asset_id':asset}))
    df = pd.DataFrame(d)[['asset_id', 'salesMicroAlgoAmount', 'market_activity_date']]
    df['salesMicroAlgoAmount'] = df.apply(lambda x:str(x['salesMicroAlgoAmount']/1000000)+ ' Algo', axis=1)
    df['Relative_Weights'] = df.apply(lambda x: asset_ids[x['asset_id']], axis=1)
    df.rename(columns={'salesMicroAlgoAmount':'Sales_Algo_Amount', 'asset_id': 'Asset_ID', 'market_activity_date':'Market_Activity_Date'},inplace=True) 
    print(tabulate(df, headers='keys', tablefmt='fancy_grid',stralign='center', numalign='center'))


def get_n_from_listing(asset_ids):
    d = []
    for asset in asset_ids.keys():
        d = d+ list(listdata.find({'asset_id':asset}))
    df = pd.DataFrame(d)[['asset_id', 'salesMicroAlgoAmount', 'market_activity_date']]
    df['salesMicroAlgoAmount'] = df.apply(lambda x:str(x['salesMicroAlgoAmount']/1000000)+ ' Algo', axis=1)
    df['Relative_Weights'] = df.apply(lambda x: asset_ids[x['asset_id']], axis=1)
    df.rename(columns={'salesMicroAlgoAmount':'Listing_Algo_Amount', 'asset_id': 'Asset_ID', 'market_activity_date':'Market_Activity_Date'},inplace=True) 
    print(tabulate(df, headers='keys', tablefmt='fancy_grid',stralign='center',numalign='center'))
    
def updates_data():
    listdata.drop()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_listing_data(500))
    get_sales_data(500)
    # export_csv_for_ml()
    # get_main_nft_data(913579356)

def init_data():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_listing_data(500))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_sales_data(10000))

    
if __name__ == "__main__":
    if list(listdata.find())==[] and list(salesdata.find())==[]:
        print('=======start data initialization=======')
        init_data()
    t0 = time.time()
    while True:
        t1 = time.time()
        if (t1-t0) >= 60.0:
            updates_data()
            t0 = t1 


