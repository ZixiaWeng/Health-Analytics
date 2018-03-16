import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import csv
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class Logistic_Regression:
    def __init__(self):
        # global data
        # self.ft = pd.read_csv("first_data/data/training_data.csv", header = None)
        self.ft = pd.read_csv("./preprocessed_data.csv", header = None)
        # self.fv = pd.read_csv("first_data/data/validation_data.csv", header = None)

    # -------------------------------------------------------------------
    def main(self, args):

        X = self.ft.ix[:,2:48]
        labels = self.ft.ix[:, 49]
        
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

        heldout = [0.90, 0.8, 0.50, 0.2, 0.1, 0.01]
        rounds = 20

        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        #If you want to change the layer parameter, change in hidden_layer_sizes
        classifiers = [
            ("Neural Network 20", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 2), random_state=1)),
            ("Neural Network 25", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 2), random_state=1)),
            ("Neural Network 30", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 2), random_state=1)),
            ("Neural Network 35", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(35, 2), random_state=1)),
            ("Neural Network 40", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 2), random_state=1))
        ]



        xx = 1. - np.array(heldout)
        for name, clf in classifiers:
            print("training %s" % name)
            rng = np.random.RandomState(42)
            yy = []
            for i in heldout:
                yy_ = []
                for r in range(rounds):
                    X_train, X_test, y_train, y_test = \
                        train_test_split(X, Y, test_size=i, random_state=rng)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    yy_.append(1 - np.mean(y_pred == y_test))
                yy.append(np.mean(yy_))
            plt.plot(xx, yy, label=name)

        plt.legend(loc="upper right")
        plt.xlabel("Proportion train")
        plt.ylabel("Test Error Rate")
        plt.show()

         
# =======================================================================
if __name__ == "__main__":
    lrmd = Logistic_Regression()
    lrmd.main(None)