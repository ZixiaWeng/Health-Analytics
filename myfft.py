# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv 
import datetime
import time
import pytz
import pylab as pl


def to_date(timestamp):
    pst_tz = pytz.timezone('US/Pacific')
    return datetime.datetime.fromtimestamp(timestamp, pst_tz)

def get_hours(data):
    return data.hour

class Prediction:
    def __init__(self):
        self.alldata = {}

    def preprocess(self):
        path = 'first_data/data_lay/'

        for filename in os.listdir(path):
            # df = pd.DataFrame() 
            df = pd.read_csv(path+filename, header = None)
            # print df
            df.columns = ['time','a','b','c','d','label']
            # print to_date(float(1521059843377)/1000.0)
            df['time'] = df['time'].apply(lambda x: get_hours(to_date(float(x)/1000.0)))
            # print df['time']
            col = df.a
            print col

            # newRows = []
            # for i in range(200):
            #     newRows.append(col.ix[i*10])

            # print newRows
            rows = col.ix[:199]
            print rows

            self.myfft(rows)
            
            # rows = newRows

            # print df
            # with open(path+filename) as csvfile:
            #   line = f.readline()

            # sampling_rate = 8000
            # sampling_rate = len(rows)
            # print sampling_rate
            # fft_size = 16
            # # t = np.arange(0, 1.0, 1.0/sampling_rate)
            # t = np.arange(0, len(rows), 1.0) 
            # print t.size


            # # x = np.sin(2*np.pi*156.25*t)  + 2*np.sin(2*np.pi*234.375*t)
            # x = rows
            # # print x.size
            # xs = x[:fft_size]
            # xf = np.fft.rfft(xs)/fft_size
            # freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
            # xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
            # pl.figure(figsize=(8,4))
            # pl.subplot(211)
            # pl.plot(t[:fft_size], xs)
            # pl.xlabel(u"时间(秒)")
            # pl.title(u"156.25Hz和234.375Hz的波形和频谱")
            # pl.subplot(212)
            # pl.plot(freqs, xfp)
            # pl.xlabel(u"频率(Hz)")
            # pl.subplots_adjust(hspace=0.4)
            # pl.show()


    def myfft(self, data):
        sampling_rate = len(data)
        print sampling_rate
        fft_size = 16
        # t = np.arange(0, 1.0, 1.0/sampling_rate)
        t = np.arange(0, len(data), 1.0) 
        print t.size


        # x = np.sin(2*np.pi*156.25*t)  + 2*np.sin(2*np.pi*234.375*t)
        x = data
        # print x.size
        xs = x[:fft_size]
        xf = np.fft.rfft(xs)/fft_size
        freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
        xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        pl.figure(figsize=(8,4))
        pl.subplot(211)
        pl.plot(t[:fft_size], xs)
        pl.xlabel(u"时间(秒)")
        pl.title(u"时域波形和频谱")
        pl.subplot(212)
        pl.plot(freqs, xfp)
        pl.xlabel(u"频率(Hz)")
        pl.subplots_adjust(hspace=0.4)
        pl.show()

if __name__ == '__main__':
    p = Prediction()
    p.preprocess()

  
