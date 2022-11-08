from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv1D
from tensorflow.keras.layers import BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import keras

initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)

pathToTrainFeature = './feature_extracted/BAD_wblr_feature.npy'
pathToTrainLabel = './feature_extracted/BAD_wblr_label.npy'

saveModelTo = './model/' # path where to save generated model and csv file with results
modelName = 'all_convnet_BAD.h5'
resultName = 'all_convnet_BAD.csv'

pathToTestMetadata = './tester/ff1010bird_metadata.txt'
pathToTestFeature = './tester/ff1010bird_feature.npy'


model = Sequential()

#conv1
model.add(ZeroPadding2D((2,2),input_shape=(40,500,1)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool1
model.add(Conv2D(16, (5, 1), strides=(5, 1), activation='relu'))
model.add(Dropout(0.50))

#conv2
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool2
model.add(Conv2D(16, (2, 1), strides=(2, 1), activation='relu'))
model.add(Dropout(0.50))

#conv3
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool3
model.add(Conv2D(16, (2, 1), strides=(2, 1), activation='relu'))
model.add(Dropout(0.50))

#conv4
model.add(ZeroPadding2D((2,2)))
model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', kernel_initializer=initializer))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

#learnedpool4
model.add(Conv2D(16, (2, 1), strides=(2, 1), activation='relu'))
model.add(Dropout(0.50))

#stacking(reshaping)
model.add(Reshape((500, 16)))

#learnedpool5
model.add(Conv1D(16, (500), strides=(1), activation='relu'))
model.add(Dropout(0.50))

#fully connected layers using conv
model.add(Reshape((1, 1,16)))
model.add(Conv2D(196,(1,1),activation = 'sigmoid'))
model.add(Dropout(0.50))
model.add(Conv2D(2,(1,1),activation = 'softmax'))
model.add(Reshape((2,)))
model.summary()

#train_data
classes = 2
feature = np.load(pathToTrainFeature)
label = np.load(pathToTrainLabel)
label = to_categorical(label, 2)
opt = Adam(decay = 1e-6)
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.50, shuffle=True) # assign validation if needed

# compile model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

#fit the model
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,verbose=2)

#save_model
if not os.path.exists(saveModelTo):
    os.mkdir(saveModelTo)
model.save(saveModelTo+modelName) 

#test_data
test_class_file = np.loadtxt(pathToTestMetadata, delimiter=',', dtype='str') # test_class_file[:,0] -> ID, test_class_file[:,1] -> real value of hasBird
test_data = np.load(pathToTestFeature)

#test_label_predict
predict_x = model.predict(test_data) 
classes_x = np.argmax(predict_x,axis=1)

#save_test_labels_&_probs
pred_probs=np.array(classes_x)
np.savetxt(saveModelTo+resultName,np.c_[test_class_file[:,0],pred_probs[:]],fmt='%s', delimiter=',')
