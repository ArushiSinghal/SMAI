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
from keras import backend
backend.set_image_dim_ordering('tf')
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.preprocessing.image import img_to_array
