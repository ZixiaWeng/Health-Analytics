import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import csv


class Logistic_Regression:
    def __init__(self):
        # global data
        self.ft = pd.read_csv("./first_data/data/training_data.csv", header = None)
        self.fv = pd.read_csv("./first_data/data/validation_data.csv", header = None)

    # -------------------------------------------------------------------
    def main(self, args):

        # df = pd.read_csv("first_data/data/training_data.csv", header = None)
        # print df
        # X = self.ft.ix[:,0:24]
        X = self.ft.ix[:,2:24]

        labels = self.ft.ix[:, 25]
        Y = []
        for i in range(len(labels)):
            if labels[i] == 'sitting':
                Y.append(0)
            elif labels[i] == 'standing':
                Y.append(1)
            elif labels[i] == 'walking':
                Y.append(2)
            else:
                Y.append(3)


        logreg = linear_model.LogisticRegression()
        logreg.fit(X, Y)

        test_X = self.ft.ix[1:200, 2:24]
        test_Y = self.ft.ix[1:200, 25]
        score = logreg.score(X, Y)
        print score


         
# =======================================================================
if __name__ == "__main__":
    lrmd = Logistic_Regression()
    lrmd.main(None)