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


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

a  = unpickle("data_batch_1")
data = a["data"]
labels = a["labels"]
b = unpickle("batches.meta")
label_names = b["label_names"]

batch_size = 1000
num_epochs = 10
kernel_size = 3
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

data = data.reshape(data.shape[0], 32, 32, 3)
num_train, height, width, depth = data.shape
num_classes = 10

data = data.astype('float32')
data = data/255
Y_train = np_utils.to_categorical(labels, num_classes)

print data.dtype
print data.shape
print len(labels)
print label_names

#MODELLING BEGINS
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
inp = Input(shape=(height, width, depth))

model = Sequential()

model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), input_shape=(32, 32, 3), padding='same'))
BatchNormalization(axis = -1)
model.add(Activation('relu'))

model.add(Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same'))
BatchNormalization(axis=-1)
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3, 3), padding='same'))
BatchNormalization(axis=-1)
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same'))
BatchNormalization(axis=-1)
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512))
BatchNormalization()
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

'''
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
test_gen = ImageDataGenerator()
'''

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(data, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1) # ...holding out 10% of the data for validation

'''
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

print functors
'''
print len(model.layers)
print model.summary()
layer_dict = dict([(layer.name, layer) for layer in model.layers])
print layer_dict

layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
print intermediate_output

interm = intermediate_output
C_range = np.array([1])
print C_range

C_range = C_range.flatten()
parameters = {'kernel':['linear'], 'C':C_range}

svm_clsf = svm.SVC()
grid_clsf = GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=1, verbose=2)

grid_clsf.fit(interm, labels)
classifier = grid_clsf.best_estimator_

print grid_clsf.best_estimator_
train_predictions = classifier.predict(interm)

labels = np.asarray(labels)

train_accuracy = accuracy_score(train_predictions, labels)
print "Training Accuracy: %.4f" % (train_accuracy)


