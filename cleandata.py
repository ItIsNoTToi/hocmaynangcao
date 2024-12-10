import csv
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_csv('kanpur.csv')

# kiem tra du lieu null
print(df.isnull().sum())

# chuyen doi du lieu ngay thang nam theo mua
df['date_time'] = pd.to_datetime(df['date_time'])
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

# chuyen doi du lieu gio phut
df['moonrise'] = df['moonrise'].replace('No moonrise', '00:00 AM')
df['moonrise_hour'] = pd.to_datetime(df['moonrise'], format='%I:%M %p',errors='coerce').dt.hour
df['sunrise_hour'] = pd.to_datetime(df['sunrise'], format='%I:%M %p').dt.hour

df['moonset'] = df['moonset'].replace('No moonset', '00:00 AM')
df['moonset_hour'] = pd.to_datetime(df['moonset'], format='%I:%M %p',errors='coerce').dt.hour
df['sunset_hour'] = pd.to_datetime(df['sunset'], format='%I:%M %p').dt.hour

def classify_time(hour):
    if hour < 0 or hour >= 24:
        return 'invalid'
    if 6 <= hour <= 12:
        return 'sang'
    elif 13 <= hour <= 18:
        return 'ngay'
    elif 19 <= hour <= 21: 
        return 'toi'
    else: 
        return 'khuya'
    
df['moonrise_time_class'] = df['moonrise_hour'].apply(classify_time)
df['sunrise_time_class'] = df['sunrise_hour'].apply(classify_time)
    
df['moonset_time_class'] = df['moonset_hour'].apply(classify_time)
df['sunset_time_class'] = df['sunset_hour'].apply(classify_time)


# chuan hoa du lieu ve [0-1]
scaler = MinMaxScaler()
df[['maxtempC', 'mintempC', 'humidity', 'windspeedKmph']] = scaler.fit_transform(df[['maxtempC', 'mintempC', 'humidity', 'windspeedKmph']])


# One-Hot Encoding: ma hoa du lieu phan loai
#df = pd.get_dummies(df, columns=['season'], drop_first=True)

df = df.drop(columns=['date_time', 'month','year', 'day', 'hour', 'moonrise', 'sunrise', 'moonrise_hour', 'sunrise_hour', 'moonset', 'sunset','moonset_hour', 'sunset_hour'])

df = df.sample(n=6000, random_state=42) 

label_encoder = LabelEncoder()
df['season'] = label_encoder.fit_transform(df['season'])
df['moonrise_time_class'] = label_encoder.fit_transform(df['moonrise_time_class']) 
df['sunrise_time_class'] = label_encoder.fit_transform(df['sunrise_time_class'])
df['moonset_time_class'] = label_encoder.fit_transform(df['moonset_time_class'])
df['sunset_time_class'] = label_encoder.fit_transform(df['sunset_time_class'])
print(df)

# Lưu DataFrame vào file CSV
df.to_csv('kanpur_clean.csv', index=False) 
