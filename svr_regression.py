import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Car_Purchasing_Data.csv',encoding='Latin-1')
X=data.iloc[:,3:8].values
y=data.iloc[:,8:9].values


#feature scaling
from sklearn.preprocessing import  StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

#svr regression
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#predicting values
y_pred=regressor.predict(X)
y_pred=sc_y.inverse_transform(y_pred)
