#!/usr/bin/python
import numpy
import csv
from csv import reader
from matplotlib import pyplot
import sys

def load_csv(filename):
        csv_reader = reader(open(filename, 'rt'))
        x = list(csv_reader)
        result = numpy.array(x)
	result = numpy.delete(result, 0, 0)
        col = result[:,6]
        result = numpy.delete(result, 6, 1)
        return result, col

dataset_train = 'datasets/q3/train.csv'
dataset_test = 'datasets/q3/test.csv'
result , label = load_csv(dataset_train)
result1, label1 = load_csv(dataset_test)
