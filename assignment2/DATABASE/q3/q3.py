from csv import reader
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics
import PIL.Image
import os
from cStringIO import StringIO
import pylab
import sys
import matplotlib
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import os
import sys

import argparse, os, sys

def lass_re():
    param_grid = {'C': [0.000001, 0.00001, 0.0001 ,0.001, 0.01, 0.02, 0.1, 1, 10, 100, 1000]}
    model = linear_model.LogisticRegression(penalty='l1')
    reg = GridSearchCV(estimator = model, param_grid = param_grid)
    reg.fit(predictors, labels)
    classifier = reg.best_estimator_
    images = reg.best_estimator_.coef_
    images = images.reshape((28, 28))
    imagesplot = plt.imshow(images, cmap='gray')
    pylab.show()
    y_pred = classifier.predict(test_variable)
    print metrics.accuracy_score(y_test, y_pred)

def regress_r():
    param_grid = {'C': [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.1, 1, 10, 100, 1000]}
    model = linear_model.LogisticRegression()
    reg = GridSearchCV(estimator = model, param_grid = param_grid)
    reg.fit(predictors, labels)
    images = reg.best_estimator_.coef_
    images = images.reshape((28, 28))
    imagesplot = plt.imshow(images, cmap='gray')
    pylab.show()
    y_pred = reg.predict(test_variable)
    print metrics.accuracy_score(y_test, y_pred)

def loading_csv_file(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    dataset = np.array(dataset).astype('float')
    return dataset

test_file = 'notMNIST_test_data.csv'
test_variable = loading_csv_file(test_file)
test_labelfile = 'notMNIST_test_labels.csv'
files_training = 'notMNIST_train_data.csv'
predictors = loading_csv_file(files_training)
train_labelfile = 'notMNIST_train_labels.csv'
labels = loading_csv_file(train_labelfile)
labels = labels[:,0]
y_test = loading_csv_file(test_labelfile)
y_test = y_test[:, 0]
regress_r()
lass_re()
