import pandas as pd  
import numpy as np 

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score 

data = pd.read_csv("Mall_Customers.csv")
#Data pre-processing 
data = data.drop('CustomerID',axis=1)
#encode gender
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])

#define features 
x = data[['Gender','Age','Annual Income (k$)']]
y = data['Spending Score (1-100)']

#split into train/test 
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#train linear regression model 
model = LinearRegression()
model.fit(x_train,y_train)

#predication 
y_pred =model.predict(x_test)

#evaluate
print("mean squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test,y_pred))

#VISUAL 
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color= 'blue', edgecolor = 'k' )
plt.xlabel("actual spending score")
plt.ylabel("predicated spending score")
plt.title("actual vs predicated spendig score")
plt.grid(True)
plt.show()

#bar graph 
features = x.columns
coefficents = model.coef_
plt.figure (figsize=(7,5))
plt.bar(features,coefficents, 
color =['red','green' 'pink'],
edgecolor= 'black')
plt.xlabel("features")
plt.ylabel("coefficent value")
plt.title("features importance (effect on spending score)")
plt.grid(axis='y')
plt.show()