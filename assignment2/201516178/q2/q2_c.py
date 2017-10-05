from keras.models import Sequential
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
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

batch_size = 1000
num_epochs = 10
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

############
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
BatchNormalization(axis = -1)
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.1))

model.add(Conv2D(64,(3, 3)))
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
################################

startingDir = os.getcwd()
batch_folder = sys.argv[1]
testing_file = sys.argv[2]
num_classes = 10
#f = open('q2_b_output.txt', 'w')
f = open('q2_c_output.txt', 'w')
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
	model.fit(data, Y_train,batch_size=64, epochs=10, verbose=1, validation_split=0.1)



a  = unpickle(batch_data_list[k])
data = a["data"]
labels = a["labels"]

for k in range(len(batch_data_list)-1):
	a  = unpickle(batch_data_list[k+1])
	data1 = a["data"]
	labels1 = a["labels"]
	data = np.concatenate((data,data1),axis=0)
	labels = labels + labels1
data = data.reshape(data.shape[0], 32, 32, 3)
data = data.astype('float32')
print data.shape
data = data/255
print data.shape
os.chdir(startingDir)

print len(model.layers)
print model.summary()
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print layer_dict

layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)


intermediate_output = intermediate_layer_model.predict(data)
interm = intermediate_output
C_range = np.array([10])

C_range = C_range.flatten()
parameters = {'kernel':['rbf'], 'C':C_range}

grid_clsf = svm.SVC(C = 10.0, kernel = 'rbf', verbose = 1)
#grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2)

grid_clsf.fit(interm, labels)
#classifier = grid_clsf.best_estimator_

#print grid_clsf.best_estimator_

interm = intermediate_layer_model.predict(testing_data)
train_predictions = grid_clsf.predict(interm)
for i in range(len(train_predictions)):
	f.write(label_names[train_predictions[i]] + "\n")
f.close()
