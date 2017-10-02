#!/usr/bin/python
import sys
import os, cv2
import numpy
import csv
from csv import reader
from matplotlib import pyplot
import sys
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import keras
import keras.utils
import tensorflow as tf
from keras import backend
backend.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.preprocessing.image import img_to_array

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

if "__name__" != "__main__":

	'''
	Each of the batch files contains a dictionary with the following elements:

    1. data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    2. labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

	The  batches.meta file contains a Python dictionary object. It has the label names. 
	'''
	b = unpickle("batches.meta")
	label_names = b["label_names"]
	a = unpickle("data_batch_1")
	data = a["data"] 
	labels = a["labels"]
	a1 = unpickle("data_batch_2")
	data1 = a1["data"] 
	labels1 = a1["labels"]
	a2 = unpickle("data_batch_3")
	data2 = a2["data"] 
	labels2 = a2["labels"]
	a3 = unpickle("data_batch_4")
	data3 = a3["data"] 
	labels3 = a3["labels"]
	a4 = unpickle("data_batch_5")
	data4 = a4["data"] 
	labels4 = a4["labels"]
	data = numpy.concatenate((data,data1,data2,data3,data4),axis=0)
	labels = numpy.concatenate((labels,labels1,labels2,labels3,labels4),axis=0)
	labels = labels.tolist()
	num_rows, num_cols = data.shape
	temp = tf.truncated_normal([16,128,128,3])
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.shape(temp))
	#print labels
	#labels.type
	#print data.shape
	#print data.shape
	#print len(labels)
	#print label_names
	
	#Input Image Dimensions
	m =32
	n =32
