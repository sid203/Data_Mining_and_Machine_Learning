import tensorflow as tf
import numpy as np
from tensorflow.python.keras.applications import InceptionV3
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import os

# following are original dimensions for InceptionV3 
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
NUM_CLASSES = 2

# file path for 224x224x3 preprocessed images stored as .npy
x_train_file = "../persistent_storage/train_images_500.npy"
y_train_file = "../persistent_storage/train_labels_500.npy"
y_test_file = "../persistent_storage/test_images_500.npy"



def load_dataset(x_train_file,y_train_file,y_test_file):
    X_tr = np.load(x_train_file)
    y_tr = np.load(y_train_file)
    y_te = np.load(y_test_file)
    
    return X_tr,y_tr,y_te


def build_model(out_hidden_units = 1024,num_trainable = 249):

    input_tensor = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))  
    #let's use inceptionV3
    model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)

    #freeze first 249 layers and train the rest
    for layer in model.layers[:num_trainable]:
        layer.trainable = False
    for layer in model.layers[num_trainable:]:
        layer.trainable = True
    
    
    x = model.output
    x = Dense(out_hidden_units, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    inception_model = Model(inputs=model.input, outputs=predictions)
    return inception_model

def get_metric_plots(history):
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    



#load datasets
X_tr,y_tr,y_te = load_dataset(x_train_file,y_train_file,y_test_file)

#build and compile inception_model
inception_model = build_model()
inception_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
print(inception_model.summary())



#train on samples 
history = inception_model.fit(X_tr, y_tr, validation_split=0.5, epochs=10, batch_size=64)

#get plots
get_metric_plots(history)

inception_model.save('/inception_model.h5') 

