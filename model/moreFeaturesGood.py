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
from sklearn import neighbors, metrics


class Logistic_Regression:
    def __init__(self):
        # global data
        self.ft_acc = pd.read_csv("./find_sensor_data/accelerometer.csv", header = None)
        self.ft_mag = pd.read_csv("./find_sensor_data/magnetic.csv", header = None)
        self.ft_ori = pd.read_csv("./find_sensor_data/orientation.csv", header = None)
        self.ft_gro = pd.read_csv("./find_sensor_data/gyroscope.csv", header = None)
        self.ft_grav = pd.read_csv("./find_sensor_data/gravity.csv", header = None)
        self.ft_acceleration = pd.read_csv("./find_sensor_data/acceleration.csv", header = None)
        self.ft_rot = pd.read_csv("./find_sensor_data/rotation.csv", header = None)

    # -------------------------------------------------------------------
    def main(self, args):


        X_less = self.ft_ori.ix[:, 1:3]
        X_more = self.ft_ori.ix[:, 1:6]
        Xs = [X_less, X_more]

        labels = self.ft_ori.ix[:, 7]

        names = ['original features', 'advanced features']

        heldout = [0.95, 0.90, 0.75, 0.01]
        rounds = 20


        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        # If you want to change the basic classifier, you can change here!
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        # clf = linear_model.LogisticRegression()


        xx = 1. - np.array(heldout)
        for index in range(2):
            print("training "+ names[index])
            rng = np.random.RandomState(42)
            yy = []
            for ho in heldout:
                yy_ = []

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

                for r in range(rounds):
                    X_train, X_test, y_train, y_test = \
                        train_test_split(Xs[index], Y, test_size=ho, random_state=rng)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    # yy_.append(1 - np.mean(y_pred == y_test))
                    accuracy = metrics.accuracy_score(y_test, y_pred)
                    yy_.append(accuracy)
                yy.append(np.mean(yy_))
            plt.plot(xx, yy, label=names[index])

        plt.legend(loc="upper right")
        plt.xlabel("Proportion train")
        plt.ylabel("Accuracy Rate")
        plt.show()

         
# =======================================================================
if __name__ == "__main__":
    lrmd = Logistic_Regression()
    lrmd.main(None)