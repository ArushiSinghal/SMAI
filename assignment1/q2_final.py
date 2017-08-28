#!/usr/bin/python
import numpy
import csv
from csv import reader
from matplotlib import pyplot
import sys
import pandas

def load_csv(filename):
        csv_reader = reader(open(filename, 'rt'))
        x = list(csv_reader)
        result = numpy.array(x).astype("float")
        col = result[:,10]
        result = numpy.delete(result, 10, 1)
        num_rows, num_cols = result.shape
        b = numpy.ones((num_rows,1)).astype("float")
        result = numpy.hstack((result,b))
        result = numpy.array(result).astype("float")
        return result, col

def margin_train_dataset(margin_w, epoch, result, label, num_cols, num_rows, learning_rate, b):
    for j in range(epoch):
        for i in range(num_rows):
            ans = numpy.dot(margin_w, result[i])
            #print ans
            if (ans <= b):
                c1 = b - numpy.dot(margin_w,result[i])
                c2 = learning_rate*c1*1.0
                c3 = numpy.dot(result[i], result[i])
                mm = ((c2)/(c3*1.0))*(result[i])
                margin_w = numpy.add(margin_w,mm)
    return margin_w

def margin_test_dataset(margin_w, result1, label1, num_cols, num_rows, b):
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        for i in range(num_rows):
            ans = numpy.dot(margin_w, result1[i])
            if (ans > 0) and (label1[i] == 2):
                count = count + 1
            if (ans < 0) and (label1[i] == 4):
                count1 = count1 + 1
            if (ans >= 0) and (label1[i] == 4):
                count2 = count2 + 1
            if (ans <= 0) and (label1[i] == 2):
                count3 = count3 + 1
        return count, count1, count2, count3

dataset_train = sys.argv[1]
dataset_test = sys.argv[2]
result , label = load_csv(dataset_train)
result1, label1 = load_csv(dataset_test)
num_rows, num_cols = result.shape
#w = numpy.zeros((1,num_cols))
margin_w = [1.0 for i in range(num_cols)]
margin_w = numpy.array(margin_w).astype("float")
min = []
max =[]
for j in range(num_cols):
    a = result[0][j]
    b = result[0][j]
    for k in range(num_rows):
        if result[k][j] < a:
            a = result[k][j]
        if result[k][j] > b:
            b = result[k][j]
    min.append(a)
    max.append(b)
for j in range(num_cols - 1):
    for i in range(num_rows):
        result[i][j] = ((result[i][j] - min[j])*1.0)/((max[j] - min[j]) * 1.0)
result1 = result
for i in range(num_rows):
        if label[i] == 4:
            result[i] = -1.0*result[i]
epoch = 100
learning_rate = 1.5
num_rows1, num_cols1 = result1.shape
b = 1
margin_w = margin_train_dataset(margin_w, epoch, result, label, num_cols, num_rows, learning_rate, b)
print result
for i in range(num_rows):
        if label[i] == 4:
            result1[i] = -1.0*result1[i]
count, count1, count2, count3 = margin_test_dataset(margin_w, result1, label1, num_cols1, num_rows1, b)
total1 = count + count2
precision = (count * 1.0)/total1
total2 = count + count3
recall = (count * 1.0)/total2
#print ("%f %f" %(precision, recall))
