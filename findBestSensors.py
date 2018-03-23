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
        self.ft_acc = pd.read_csv("model/find_sensor_data/accelerometer.csv", header = None)
        self.ft_mag = pd.read_csv("model/find_sensor_data/magnetic.csv", header = None)
        self.ft_ori = pd.read_csv("model/find_sensor_data/orientation.csv", header = None)
        self.ft_gro = pd.read_csv("model/find_sensor_data/gyroscope.csv", header = None)
        self.ft_grav = pd.read_csv("model/find_sensor_data/gravity.csv", header = None)
        self.ft_acceleration = pd.read_csv("model/find_sensor_data/acceleration.csv", header = None)
        self.ft_rot = pd.read_csv("model/find_sensor_data/rotation.csv", header = None)

    # -------------------------------------------------------------------
    def main(self, args):

        X_acc = self.ft_acc.ix[:, 1:6]
        X_mag = self.ft_mag.ix[:, 1:6]
        X_ori = self.ft_ori.ix[:, 1:6]
        X_gro = self.ft_gro.ix[:, 1:6]
        X_grav = self.ft_grav.ix[:, 1:6]
        X_ft_acceleration = self.ft_acceleration.ix[:,1:6]
        X_rot = self.ft_rot.ix[:,1:8]
        Xs = [X_acc, X_mag, X_ori, X_gro, X_grav, X_ft_acceleration, X_rot]

        # X_less = self.ft

        Y_acc = self.ft_acc.ix[:, 7]
        Y_mag = self.ft_mag.ix[:, 7]
        Y_ori = self.ft_ori.ix[:, 7]
        Y_gro = self.ft_gro.ix[:, 7]
        Y_grav = self.ft_grav.ix[:, 7]
        Y_ft_acceleration = self.ft_acceleration.ix[:,7]
        Y_rot = self.ft_rot.ix[:,9]
        Ys = [Y_acc, Y_mag, Y_ori, Y_gro, Y_grav, Y_ft_acceleration, Y_rot]

        names = ['accelerometer', 'magnetic','orientation','gyroscope','gravity','acceleration','rotation']

        heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
        rounds = 20


        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
        # If you want to change the basic classifier, you can change here!
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        # clf = linear_model.LogisticRegression()


        xx = 1. - np.array(heldout)
        for index in range(7):
            print("training "+ names[index])
            rng = np.random.RandomState(42)
            yy = []
            for ho in heldout:
                yy_ = []

                labels = Ys[index]
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