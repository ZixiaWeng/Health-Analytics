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
        self.data = self.preprocess()

    def preprocess(self): 
        path = 'first_data/'
        dire = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path,dI))]
        data_all = []
        data_dir = []
        final_data_frame_folder = []
        final_data_frame = pd.DataFrame()
        print dire
        for subdir in dire:
            data_all = []
            print '------------- subdriectory name:', subdir,'---------------'
            for filename in os.listdir(path+subdir):
                print filename
                df = pd.read_csv(path+subdir+'/'+filename, header = None)
                df.columns = df.columns.astype(str)
                df.columns.values[0] = 'time'
                df.columns.values[-1] = 'label'
                df.columns.values[-2] = 'type'
                # df['time'] = df['time'].apply(lambda x: to_date(float(x)/1000.0))
                data_all.append(df)
            print len(data_all)
            data_dir.append(data_all)
        print 'how many data folder in training data:',len(data_dir)
        high_dim_data = pd.DataFrame()
        for data in data_dir:
            high_dim_data, label = self.merge_data(data,3000)
            # high_dim_data.to_csv('first_data/preprocessed_data.csv', sep=',')
            high_dim_data_complex = self.expand_dim(high_dim_data, label)
            # print high_dim_data_complex
            final_data_frame_folder.append(high_dim_data_complex)
        final_data_frame = self.combine_data(final_data_frame_folder)
        return final_data_frame
        # print len(final_data_frame)

    def merge_data(self, data_all, threshold):
        label = data_all[0]['label'][0]
        df_final = data_all[0]
        df_final = df_final.drop(['label', 'type'], axis=1)
        for i in range(len(data_all)):
            if i != 0 and i != len(data_all)-1:
                df_new = data_all[i]
                df_new = df_new.drop(['label', 'type'], axis=1)
                # if you want to keep the values that are matched in time during the mergy, delete how = "left"
                df_final = pd.merge(df_final, df_new, on="time")
                # df_final_witout_nan = pd.merge(df_final, how="left" df_new, on="time")
                # print df_final
            elif i == len(data_all)-1:
                df_new = data_all[i]
                df_new = df_new.drop(['type', 'label'], axis=1)
                df_final = pd.merge(df_final, df_new, on="time") 
        
        # Get first 1/10 data from all data
        lenth = len(df_final.index)
        df_final = df_final.head(lenth/10)
        # distributed datas averagily by 1/5 in all data
        # for i in range(lenth/10):
        #     if i%2 != 0:
        #        df_final = df_final.drop([i])
        df_final = df_final.reset_index(drop=True)
            # print df_final
        return df_final, label

    def expand_dim(self, high_dim_data, label):
        new_dim_data = pd.DataFrame(columns = range(25))
        for i in range(len(high_dim_data.index)):
            if i == 0:
                iterator = np.array(high_dim_data.iloc[0])
            else:
                # print np.array(high_dim_data.iloc[i])
                diff_between_rows = list(np.array(high_dim_data.iloc[i]).astype(float) - iterator.astype(float))
                # print diff_between_rows
                diff_between_rows.append(str(label))
                new_dim_data.loc[i-1] = diff_between_rows
                # new_dim_data.append(diff_between_rows)
                iterator = np.array(high_dim_data.iloc[i])
        combined_data = pd.concat([high_dim_data[:-1], new_dim_data], axis=1, join_axes=[high_dim_data[:-1].index])
        combined_data = combined_data.reset_index(drop=True)
        # print combined_data
        return combined_data

    def combine_data(self, final_data_frame_folder):
        result = pd.concat(final_data_frame_folder)
        print result.shape, result
        result.to_csv('preprocessed_data.csv', sep=',')
        return result
        # print type(df1), df1.shape

    # def get_fft(self):










