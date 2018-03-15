# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv 
import datetime
import time
import pytz


def to_date(timestamp):
    pst_tz = pytz.timezone('US/Pacific')
    return datetime.datetime.fromtimestamp(timestamp, pst_tz)

def get_hours(data):
    return data.hour

class Prediction:
    def __init__(self):
        self.alldata = {}

    def preprocessing(self):
        path = 'first_data/data/'

        for filename in os.listdir(path):
            print filename
            df = pd.read_csv(path+filename, names = ['time','a','b','c','d','label'])
            print df
            df['time'] = df['time'].apply(lambda x: get_hours(to_date(float(x)/1000.0)))
            print df

