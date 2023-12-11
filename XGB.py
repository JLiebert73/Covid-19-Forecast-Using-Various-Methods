import numpy as np
from sklearn import preprocessing
from xgboost import XGBRegressor
from data_preprocessing import preprocess_data

def train_model(train, test, country_list):
    sub = []

    for country in country_list:
        province_list = train.loc[train['Country_Region'] == country].Province_State.unique()
        for province in province_list:
            X_train = train.loc[(train['Country_Region'] == country) & (train['Province_State'] == province), ['Date']].astype('int')
            Y_train_c = train.loc[(train['Country_Region'] == country) & (train['Province_State'] == province), ['ConfirmedCases']]
            Y_train_f = train.loc[(train['Country_Region'] == country) & (train['Province_State'] == province), ['Fatalities']]
            X_test = test.loc[(test['Country_Region'] == country) & (test['Province_State'] == province), ['Date']].astype('int')
            X_forecastId = test.loc[(test['Country_Region'] == country) & (test['Province_State'] == province), ['ForecastId']]
            X_forecastId = X_forecastId.values.tolist()
            X_forecastId = [v[0] for v in X_forecastId]
            model_c = XGBRegressor(n_estimators=1000)
            model_c.fit(X_train, Y_train_c)
            Y_pred_c = model_c.predict(X_test)
            model_f = XGBRegressor(n_estimators=1000)
            model_f.fit(X_train, Y_train_f)
            Y_pred_f = model_f.predict(X_test)
            for j in range(len(Y_pred_c)):
                dic = {'ForecastId': X_forecastId[j], 'ConfirmedCases': Y_pred_c[j], 'Fatalities': Y_pred_f[j]}
                sub.append(dic)

    return sub
