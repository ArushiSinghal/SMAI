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

def load_csv1(filename):
        csv_reader = reader(open(filename, 'rt'))
        x = list(csv_reader)
        result = numpy.array(x).astype("int")
        num_rows, num_cols = result.shape
        b = numpy.ones((num_rows,1))
        result = numpy.hstack((result,b))
        result = numpy.array(result).astype("int")
        return result

def batch_margin_train_dataset(batch_margin_w, epoch, result, label, num_cols, num_rows, learning_rate,b):
    k = numpy.zeros((num_rows,num_cols))
    for j in range(epoch):
        l = 0
        k = numpy.zeros((num_rows,num_cols))
        for i in range(num_rows):
            ans = numpy.dot(batch_margin_w, result[i])
            if (ans <= b) and (label[i] == 1):
                k[l] = learning_rate*result[i]
                l +=1
            elif (ans >= -1*b) and (label[i] == 0):
                k[l] = -1*learning_rate*result[i]
                l +=1
        for m in range(l):
            batch_margin_w = numpy.add(batch_margin_w,k[m])
    return batch_margin_w;

def batch_margin_test_dataset(batch_margin_w, result1, num_cols, num_rows, b):
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(num_rows):
        ans = numpy.dot(batch_margin_w, result1[i])
        if (ans >= 0):
            count = count + 1
            print ("1")
        elif (ans < 0):
            count1 = count1 + 1
            print ("0")
    return count, count1;

def batch_train_dataset(batch_w, epoch, result, label, num_cols, num_rows, learning_rate):
    k = numpy.zeros((num_rows,num_cols))
    for j in range(epoch):
        l = 0
        k = numpy.zeros((num_rows,num_cols))
        for i in range(num_rows):
            ans = numpy.dot(batch_w, result[i])
            if (ans <= 0) and (label[i] == 1):
                k[l] = learning_rate*result[i]
                l +=1
            elif (ans >= 0) and (label[i] == 0):
                k[l] = -1*learning_rate*result[i]
                l +=1
        for m in range(l):
            batch_w = numpy.add(batch_w,k[m])
    return batch_w;

def batch_test_dataset(batch_w, result1, num_cols, num_rows):
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(num_rows):
        ans = numpy.dot(batch_w, result1[i])
        if (ans >= 0):
            count = count + 1
            print ("1")
        elif (ans < 0):
            count1 = count1 + 1
            print ("0")
    return count, count1;

def margin_train_dataset(margin_w, epoch, result, label, num_cols, num_rows, learning_rate, b):
    for j in range(epoch):
        for i in range(num_rows):
            ans = numpy.dot(margin_w, result[i])
            if (ans <= b) and (label[i] == 1):
                mm = learning_rate*result[i]
                margin_w = numpy.add(margin_w,mm)
            elif (ans >= -1*b) and (label[i] == 0):
                mm = -1*learning_rate*result[i]
                margin_w = numpy.add(margin_w,mm)
    return margin_w;

def margin_test_dataset(margin_w, result1, num_cols, num_rows, b):
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        for i in range(num_rows):
            ans = numpy.dot(margin_w, result1[i])
            if (ans >= 0):
                count = count + 1
                print ("1")
            elif (ans < 0):
                count1 = count1 + 1
                print ("0")
        return count, count1;

def train_dataset(w, epoch, result, label, num_cols, num_rows, learning_rate):
    for j in range(epoch):
        for i in range(num_rows):
            ans = numpy.dot(w, result[i])
            if (ans <= 0) and (label[i] == 1):
                mm = learning_rate*result[i]
                w = numpy.add(w,mm)
            elif (ans >= 0) and (label[i] == 0):
                mm = -1*learning_rate*result[i]
                w = numpy.add(w,mm)
    return w;

def test_dataset(w, result1, num_cols, num_rows):
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(num_rows):
        ans = numpy.dot(w, result1[i])
        if (ans >= 0):
            count = count + 1
            print ("1")
        elif (ans < 0):
            count1 = count1 + 1
            print ("0")
    return count, count1;

dataset_train = sys.argv[1]
dataset_test = sys.argv[2]

result , label = load_csv(dataset_train)
result1 = load_csv1(dataset_test)
num_rows, num_cols = result.shape
w = numpy.zeros((1,num_cols))
w = numpy.array(w).astype("float")
batch_w = w
margin_w = w
batch_margin_w = w
epoch = 100
learning_rate = 0.5
num_rows1, num_cols1 = result1.shape

w = train_dataset(w, epoch, result, label, num_cols, num_rows, learning_rate)
count, count1 = test_dataset(w, result1, num_cols1, num_rows1)

b = 1
margin_w = margin_train_dataset(margin_w, epoch, result, label, num_cols, num_rows, learning_rate, b)
count, count1 = margin_test_dataset(margin_w, result1,num_cols1, num_rows1, b)

batch_w = batch_train_dataset(batch_w, epoch, result, label, num_cols, num_rows, learning_rate)
count, count1 = batch_test_dataset(batch_w, result1, num_cols1, num_rows1)

b = 1
batch_margin_w = batch_margin_train_dataset(batch_margin_w, epoch, result, label, num_cols, num_rows, learning_rate, b)
count, count1 = batch_margin_test_dataset(batch_margin_w, result1,num_cols1, num_rows1, b)
