# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import datetime
import time
import pytz

# import necessary model to use
from sklearn.ensemble import RandomForestRegressor

# the error metric, use c-stat, one way to score our model's performance
from sklearn.metrics import roc_auc_score

# encoding
from sklearn import preprocessing

from sklearn import cross_validation
from sklearn import neighbors, metrics, datasets


df = pd.read_csv('preprocessed_data.csv',header=None)
df = df.reset_index(drop=True)

header = list(df.columns.values)

print header

df = df.drop(header[0])
df = df.drop(header[1])
df = df.drop(header[12])
df = df.drop(header[36])


def randomf_c(df,header):
    y = df.pop(header[-1])

    print y

    le = preprocessing.LabelEncoder()
    le.fit(['sitting','standing','laying_down','walking']) # have to have all of them
    # le.fit(['sitting','standing'])

    new_y = le.transform(y)
    print set(new_y)

    # print le.inverse_transform(new_y)

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(df,new_y,test_size=0.1,random_state=0)

    # model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42)
    # model.fit(X_train,y_train)

    # y_pred = model.predict(X_test)
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    
    model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42)
    model.fit(df,new_y)
    y_oob = model.oob_prediction_
    score = roc_auc_score(new_y,y_oob)

    return score


print randomf_c(df,header)


# p = Prediction()
# print p.randomf_c(p.preprocess())