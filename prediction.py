# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv 
import datetime
import time
import pytz
import math


def to_date(timestamp):
    pst_tz = pytz.timezone('US/Pacific')
    return datetime.datetime.fromtimestamp(timestamp, pst_tz)

def get_hours(data):
    return data.hour

class Prediction:
    def __init__(self):
        self.alldata = []

    def preprocess(self): 
        path = 'first_data/'
        dire = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
        data_all = []
        print dire
        for subdir in dire:
            for filename in os.listdir(path+subdir):
                print filename
                df = pd.read_csv(path+subdir+'/'+filename, header = None)
                df.columns = df.columns.astype(str)
                df.columns.values[0] = 'time'
                df.columns.values[-1] = 'label'
                df.columns.values[-2] = 'type'
                # df['time'] = df['time'].apply(lambda x: to_date(float(x)/1000.0))
                # print df
                data_all.append(df)

            print len(data_all)
        self.build_data(data_all,3000)

    def build_data(self, data_all, threshold):
        initial_time = 0
        df_final = data_all[0]
        df_final = df_final.drop(['label', 'type'], axis=1)
        for i in range(len(data_all)-1):
            if i != 0:
                df_new = data_all[i]
                df_new = df_new.drop(['label', 'type'], axis=1)
                # if you want to keep the values that are matched in time during the mergy, delete how = "left"
                df_final = pd.merge(df_final, df_new, how="left", on="time")
                # df_final_witout_nan = pd.merge(df_final, df_new, on="time")
        print df_final
        # print df_final_witout_nan











