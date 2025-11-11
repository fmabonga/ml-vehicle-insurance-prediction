import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle 

df = pd.read_csv('car_insurance_premium_dataset.csv')
df.columns = df.columns.str.replace(" ","_").str.lower()
df.rename(columns={'insurance_premium_($)': 'insurance_premium'}, inplace=True)
df.rename(columns={'annual_mileage_(x1000_km)': 'annual_mileage'}, inplace=True)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)

y_full_train = df_full_train.insurance_premium
y_test = df_test.insurance_premium

del df_full_train["insurance_premium"]
del df_test["insurance_premium"]

train_dicts = df_full_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
features = list(dv.get_feature_names_out())
X_test = dv.transform(test_dicts)



dtrain = xgb.DMatrix(X_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)


xgb_params = {
    'eta': 0.2, 
    'min_child_weight': 1,
    'max_depth': 2,
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}
model = xgb.train(xgb_params, dtrain, num_boost_round=600,
                  verbose_eval=5,
                 )
y_pred = model.predict(dtest)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('rmse = %.3f' % rmse)

with open('Insurance-premium-prediction-model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print('output saved to Insurance-premium-prediction-model.bin')