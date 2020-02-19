#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# import tensorflow as tf


#Read data and define data frame (df=train)
df = pd.read_csv("iris.data")
df.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
df['class'].replace(to_replace=['Iris-setosa', 'Iris-versicolor','Iris-virginica'], value=[0, 1, 2])

#Plot SepalWidth and SepalLength
plt.scatter(df['sepal_length'][:50], df['sepal_width'][:50], label ='Iris-setosa')
plt.scatter(df['sepal_length'][51:100], df['sepal_width'][51:100], label ='Iris-versicolor')
plt.scatter(df['sepal_length'][101:], df['sepal_width'][101:], label ='Iris-virginica')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend(loc='best')
#plt.show()

#Lets define variables X and y
X = df.drop(columns=['class'], axis=1).values
y = df['class'].values

#Test Train Split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#Training and predicting
logmodel = LogisticRegression(max_iter=150)
logmodel.fit(X_train, Y_train)
predictions = logmodel.predict(X_test)

#Lets see precision
print(classification_report(Y_test, predictions))


