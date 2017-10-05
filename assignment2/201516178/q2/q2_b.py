from keras.models import Sequential
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import sys


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

batch_size = 100
num_epochs = 20
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512
############# MODELLING STARTS ##################

'''
1) CONVOLUTION
2)BATCHNORMALIZATION
3) ACTIVATION
4)MAXPOOLING2D
5)FLATTEN
6)FULLY CONNECTED LAYER Dense layers are kerass alias for Fully connected layers.
These layers give the ability to classify the features learned by the CNN.
7)DROPOUT
8) SECOND LAST LAYER CONTAINS NEURONS EQUAL TO NUMBER OF CLASSES
9) LAST LAYER IS THE SOFTMAX LAYER
'''
#inp = Input(shape=(height, width, depth))

model = Sequential()

model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), input_shape=(32, 32, 3), padding='same'))
BatchNormalization(axis = -1)
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(64,(3, 3), padding='same'))
BatchNormalization(axis=-1)
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same'))
BatchNormalization(axis=-1)
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512))
BatchNormalization()
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))
####### MODELLING ENDS ###########################

startingDir = os.getcwd()
batch_folder = sys.argv[1]
testing_file = sys.argv[2]
num_classes = 10
f = open('q2_b_output.txt', 'w')
#f = open('q2_c_output.txt', 'w')
batch_data_list = ["data_batch_1" , "data_batch_2" , "data_batch_3" , "data_batch_4" , "data_batch_5"]
a  = unpickle(sys.argv[2])
testing_data = a["data"]
testing_data = testing_data.reshape(testing_data.shape[0], 32, 32, 3)
testing_data = testing_data.astype('float32')
testing_data = testing_data/255

os.chdir(batch_folder)
b = unpickle("batches.meta")
label_names = b["label_names"]
for k in range(len(batch_data_list)):
	a  = unpickle(batch_data_list[k])
	data = a["data"]
	labels = a["labels"]
	data = data.reshape(data.shape[0], 32, 32, 3)
	data = data.astype('float32')
	data = data/255
	Y_train = np_utils.to_categorical(labels, num_classes)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
	gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,height_shift_range=0.08, zoom_range=0.08)
	test_gen = ImageDataGenerator()
	train_generator = gen.flow(data, Y_train, batch_size =100)
	test_generator = gen.flow(data, Y_train, batch_size =100)
	model.fit(data, Y_train,batch_size=32, epochs=20, verbose=1, validation_split=0.1)
	score = model.evaluate(data, Y_train, verbose=0)
os.chdir(startingDir)
classes = model.predict(testing_data, batch_size=128)
classes1 = classes.max(axis=1)
classes_index = classes.argmax(axis=1)
for i in range(len(classes_index)):
	f.write(label_names[classes_index[i]] + "\n")
f.close()
