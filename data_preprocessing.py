import pandas as pd
from sklearn import preprocessing

def preprocess_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train['Date'] = pd.to_datetime(train['Date'])
    train['Date'] = train['Date'].dt.strftime("%m%d")
    test['Date'] = pd.to_datetime(test['Date'])
    test['Date'] = test['Date'].dt.strftime("%m%d")
    train = train.fillna('NA')
    test = test.fillna('NA')

    return train, test
