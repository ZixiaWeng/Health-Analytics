import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Prediction:
    def __init__(self):
    	self.alldata = {}

    def preprocessing(self):
    	path = 'first_data/data/'
        for filename in os.listdir(path):
            print filename