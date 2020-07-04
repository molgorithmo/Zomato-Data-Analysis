import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

met_df = pd.read_csv('../datasets/cleaned_dataset.csv', index_col=0)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

val = le.fit_transform(met_df['rest_type']).reshape(-1,1)

res = met_df['rest_type'].values
res = ohe.fit_transform(val)
res = pd.DataFrame(res)
new = pd.concat([met_df,res], axis=1)
new.dropna(inplace=True)
new.drop('rest_type', axis=1, inplace=True)
X = new.drop('cost_for_two', axis=1)
y = new['cost_for_two']

X_train = X[:round(0.7*len(X))]
X_valid = X[round(0.7*len(X))+1 : -round(0.15*len(X))]
X_test = X[-round(0.15*len(X)):]
#y
y_train = y[:round(0.7*len(y))]
y_valid = y[round(0.7*len(y))+1 : -round(0.15*len(y))]
y_test = y[-round(0.15*len(y)):]

from sklearn.ensemble import RandomForestRegressor
print("modelling.....")
rf_100 = RandomForestRegressor(n_estimators=50, criterion='mae')
rf_100.fit(X_train, y_train)
print("modelling done.....")

pickle.dump(rf_100, open('model.pkl','wb'))
pickle.dump(le, open('label.pkl','wb'))
pickle.dump(ohe, open('hot.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
le = pickle.load(open('label.pkl','rb'))
ohe = pickle.load(open('hot.pkl','rb'))

node = [[1,1,340,240,10,1, 'Casual Dining', 3.8, 4.0, 4.3]]
node = pd.DataFrame(node)
node.columns = met_df.drop('cost_for_two',axis=1).columns
val = le.transform(node['rest_type']).reshape(-1,1)
res = node['rest_type'].values
res = ohe.transform(val)
res = pd.DataFrame(res)
new = pd.concat([node,res], axis=1)
new.dropna(inplace=True)
new.drop('rest_type', axis=1, inplace=True)
print(model.predict(new))