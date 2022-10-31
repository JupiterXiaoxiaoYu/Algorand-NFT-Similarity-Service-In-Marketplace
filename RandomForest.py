from operator import mod
import pandas as pd
from sklearn.preprocessing import LabelEncoder as LEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split


class LabelEncoder(LEncoder):
 
    def fit(self, y):
        """
        This will fit the encoder for all the unique values
        and introduce unknown value
        :param y: A list of string
        :return: self
        """
        return super(LabelEncoder, self).fit(list(y) + ['Unknown'])
 
    def transform(self, y):
        """
        This will transform the y to id list where the new values
        get assigned to Unknown class
        :param y:
        :return: array-like of shape [n_samples]
        """
        new_y = ['Unknown' if x not in set(self.classes_) else x for x in y]
        return super(LabelEncoder, self).transform(new_y)


dep_var = 'salesMicroAlgoAmount'

columns = ['Back Item', 'Background Accent', 'Body', 'Footwear', 'Hat', 'Left Arm',
       'Necklace', 'Pants', 'Right Arm', 'Scenery', 'Shirts', 'Tattoo',
       'combat', 'constitution', 'luck', 'plunder', 'Face', 'Head', 'Overcoat',
       'Shirt', 'Facial Hair', 'Pet', 'Background', 'Back Hand', 'Front Hand',
       'Hip Item']

def preprocess(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit(df[col])
            df[col] = le.transform(df[col])
            joblib.dump(le, f'./labelencoders/{col}.pkl')
    return df

def preprocess_input(input_df):
    df = pd.concat([pd.DataFrame(input_df), pd.DataFrame(columns=columns)])
    df[['combat', 'constitution', 'luck', 'plunder']] = df[['combat', 'constitution', 'luck', 'plunder']].astype('int')
    for col in df.columns:
        if df[col].dtype == 'object':
            le = joblib.load(f'./labelencoders/{col}.pkl')
            df[col] = le.transform(df[col])
    # print(df.columns)
    return df
    
    

def all_train(df):
    print('======Initiate RandomForest Model======')
    df = df.drop(['_id', 'market_activity_date', 'name', 'asset_id'],axis=1)
    df = preprocess(df)
    X = df.drop([dep_var],axis=1).values
    y = df[dep_var].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    # print(columns)
    regr = RandomForestRegressor(max_depth=6, n_estimators=100, min_samples_leaf=1, min_samples_split=2, random_state=0, max_samples=0.92, max_features=10)
    regr.fit(X_train,y_train)
    y_predicted = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_predicted)
    regr.fit(X,y)
    joblib.dump(regr, 'model.pkl')

def active_learning(new_data):
    model = joblib.load('model.pkl')
    train_y = new_data[dep_var]
    train_xs = new_data.drop([dep_var],axis=1)
    model.fit(train_xs, train_y)
    joblib.dump(model, 'model.pkl')

def predict_price_accurate_from_previous_sales(asset_data):
    asset_data = preprocess_input(asset_data)
    # print(asset_data.columns)
    model = joblib.load('model.pkl')
    # print(asset_data.values) 
    print(f'The predicted price of query asset through all sales data (RandomForest):  {model.predict(asset_data.values)/1000000} Algo')
    # print(f'The predicted price of this asset from all sales data is {model.predict(asset_data.values)/100000} algo')
    return model.predict(asset_data.values)/1000000