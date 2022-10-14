import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

data = {'x1': [40,45,38,50,48,55,53,55,58,40,55,48,45,55,60,60,60,65,50,58],
        'x2': [25,20,30,30,28,30,34,36,32,34,38,28,30,36,34,38,42,38,34,38],
        'y': [1,2,1,3,2,3,3,4,4,3,5,3,3,2,4,5,5,5,4,3]      
        }

#we are finding intercept for multilinear regression y = B0 + B1X1 + B2X2 we are finding B0, B1, B2

df = pd.DataFrame(data)

x = df[['x1','x2']]
y = df['y']

#method 1
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_[1])

# method 2 using statsmodel
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)