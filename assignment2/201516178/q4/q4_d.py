import sys
from csv import reader
import numpy
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    dataset = numpy.array(dataset).astype('float')
    predictors = dataset[:,0:11]
    labels = dataset[:,11]
    return predictors, labels

def load_test_csv(filename):
    file = open(filename, "rt")
    lines = reader(file)
    dataset = list(lines)
    dataset = numpy.array(dataset).astype('float')
    return dataset

filename = sys.argv[1]
testfile = sys.argv[2]
X_train, y_train = load_csv(filename)
test_predictors = load_test_csv(testfile)
reg = linear_model.LinearRegression(normalize = True, fit_intercept=True)
reg.fit(X_train, y_train)
y_pred = reg.predict(test_predictors)
total_true = 0
lamda = 0.554
for i in range(len(y_pred)):
    if(y_pred[i] >= lamda):
        y_pred[i] = 1
    else:
        y_pred[i] = 0
    print (int)(y_pred[i])
