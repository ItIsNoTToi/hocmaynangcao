import csv
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from deap import base, creator, tools, algorithms
import random

scaler = MinMaxScaler()

df = pd.read_csv('kanpur_clean.csv')

X = df.drop(['tempC'], axis = 1)
y = df['tempC']

print(X)
print(y)

X_scaler = scaler.fit_transform(X=X)

X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_train)
print(y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))