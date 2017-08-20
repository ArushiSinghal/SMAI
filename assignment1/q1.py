#!/usr/bin/python
import numpy
import csv
from csv import reader
from matplotlib import pyplot
import sys

def load_csv(filename):
        csv_reader = reader(open(filename, 'rt'))
        x = list(csv_reader)
        result = numpy.array(x).astype("int")
        col = result[:,0]
        result = numpy.delete(result, 0, 1)
        num_rows, num_cols = result.shape
        b = numpy.ones((num_rows,1))
        result = numpy.hstack((result,b))
        result = numpy.array(result).astype("int")
        return result, col

def train_dataset(w, epoch, result, label, num_cols, num_rows, learning_rate):
    for j in range(epoch):
        for i in range(num_rows):
            ans = numpy.dot(w, result[i])
            if (ans <= 0) and (label[i] == 1):
                mm = learning_rate*result[i]
                w = numpy.add(w,result[i])
            elif (ans >= 0) and (label[i] == 0):
                mm = -1*learning_rate*result[i]
                w = numpy.add(w,mm)
    return w;

def test_dataset(w, result1, label1, num_cols, num_rows):
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(num_rows):
        ans = numpy.dot(w, result1[i])
        if (ans > 0) and (label1[i] == 1):
            count = count + 1
        if (ans < 0) and (label1[i] == 0):
            count1 = count1 + 1
        if (ans > 0) and (label1[i] == 0):
            count2 = count2 + 1
        if (ans < 0) and (label1[i] == 1):
            count3 = count3 + 1
    return count, count1, count2, count3;

dataset_test = sys.argv[1]
dataset_train = sys.argv[2]
#dataset_train = 'datasets/MNIST_data_updated/mnist_train.csv'
#dataset_train = 'datasets/MNIST_data_updated/a_train.csv'
result , label = load_csv(dataset_train)
result1, label1 = load_csv(dataset_test)
num_rows, num_cols = result.shape
w = numpy.zeros((1,num_cols))
w = numpy.array(w).astype("int")
epoch = 30
learning_rate = 0.4
w = train_dataset(w, epoch, result, label, num_cols, num_rows, learning_rate)
num_rows, num_cols = result1.shape
count, count1, count2, count3 = test_dataset(w, result1, label1, num_cols, num_rows)
total1 = count + count2
precision = (count * 1.0)/total1
total2 = count + count3
recall = (count * 1.0)/total2
print ("%f %f" %(precision, recall))
