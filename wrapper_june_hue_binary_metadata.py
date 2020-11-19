from __future__ import print_function
import csv
import numpy as np
import datetime
import pandas as pd
import random
from numpy import argmax
np.random.seed(1337) 

import os 
import keras
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D, Input 
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import classification_report, confusion_matrix, f1_score, average_precision_score, precision_recall_curve

size = 224,224
batch_size = 20
epochs = 100
the_sizes = [100,280,460,640]
q=the_sizes[2]

target = 'hue'
num_classes = 2
path        = '/home/danh/Data/transfer_learning/Experiment_11-06-19/for_experiment'
path_label  = '/home/danh/Data/transfer_learning/Experiment_11-06-19/labels_classification_11062019_2_classes_equal_bin.csv'

now         = datetime.datetime.now()
path_save   = '/home/danh/Results/Experiment_11-06-19/classification/binary/' + '10252019' + '/' + target
path_load   = '/home/danh/Results/Experiment_11-06-19/classification/binary/' + '08242019' + '/' + target


# init metadata labels
meta_day_time  = [] # Morning=task_483=0, Noon=task_492=1, Afternoon=task_493=2
meta_camera    = [] # RGB=high=0, 3D=low=1
meta_is_sun    = [] # Shadow=0, Sun=1
annotations    = pd.read_csv(path_label)

# read images
data  = []
y     = []
ID    = []
tasks = os.listdir(path)
for task in tasks:
    cameras = os.listdir(path + '/' + task)
    for camera in cameras:
        conditions = os.listdir(path + '/' + task + '/' + camera)
        for condition in conditions:
            listing = os.listdir(path + '/' + task + '/' + camera + '/' + condition)
            curr_path = path + '/' + task + '/' + camera + '/' + condition + '/'
            for file in listing:
                im = Image.open(curr_path + file)
                im = im.resize(size)
                im = np.asarray(im, dtype="uint8")
                data.append(im)

                if 'task_483' in task:
                    meta_day_time.append(0)
                elif 'task_492' in task:
                    meta_day_time.append(1)
                else:
                    meta_day_time.append(2)

                if 'high' in camera:
                    meta_camera.append(0)
                else: # low=1
                    meta_camera.append(1)

                if 'shadow' in condition:
                    meta_is_sun.append(0)
                else: # Sun=1
                    meta_is_sun.append(1)

                file = file[:-4] #remove ".jpeg"/".bmp"
                temp = file.split("_")
                val = annotations[annotations['serial_id'] == int(temp[0])]
                y.append(float(val[target]))
                ID.append(temp[0]) #split train/test based on ID


for w in range(0, 10):
    # split to train-test 80-20 - Jun
    unique_ID = np.asarray(list(set(ID)))
    msk       = np.random.rand(unique_ID.shape[0]) < 0.8 # random split
    train_ID  = unique_ID[msk==True]
    test_ID   = unique_ID[msk==False]
    if not os.path.exists(path_save + '/regular/mask/'):
        os.makedirs(path_save + '/regular/mask/')
    pd.DataFrame(train_ID).to_csv(path_save + '/regular/mask/' + target + '_train_mask_' + str(w+1) + '.csv')
    pd.DataFrame(test_ID).to_csv(path_save + '/regular/mask/' + target + '_test_mask_' + str(w+1) + '.csv')
	
	
for w in range(0, 10):
    
    train_ID  = pd.read_csv(path_load + '/regular/mask/' + target + '_train_mask_' + str(w+1) + '.csv')
    test_ID   = pd.read_csv(path_load + '/regular/mask/' + target + '_test_mask_' + str(w+1) + '.csv')
    
    train_ID  = train_ID.iloc[:,1].values.tolist()
    test_ID   = test_ID.iloc[:,1].values.tolist()

    trainInd = []
    testInd  = []
    for i in range(0,len(ID)):
        if int(ID[i]) in train_ID:
            trainInd.append(i)
        if int(ID[i]) in test_ID:
            testInd.append(i)

    trainInd=np.asarray(trainInd)
    testInd=np.asarray(testInd)
    train_data=np.array(data)[trainInd.astype(int)]
    test_data=np.array(data)[testInd.astype(int)]
    train_y=np.array(y)[trainInd.astype(int)]
    test_y=np.array(y)[testInd.astype(int)]
    train_meta_day_time=np.array(meta_day_time)[trainInd.astype(int)]
    test_meta_day_time=np.array(meta_day_time)[testInd.astype(int)]
    train_meta_camera=np.array(meta_camera)[trainInd.astype(int)]
    test_meta_camera=np.array(meta_camera)[testInd.astype(int)]
    train_meta_is_sun=np.array(meta_is_sun)[trainInd.astype(int)]
    test_meta_is_sun=np.array(meta_is_sun)[testInd.astype(int)]

    train_data  = train_data.astype('float32')
    test_data   = test_data.astype('float32')
    train_data /= 255 #normalize 0-255 => 0-1
    test_data  /= 255 #normalize 0-255 => 0-1
    train_y     = np.asarray(train_y)
    test_y      = np.asarray(test_y)
    train_y     = train_y.astype('float32')
    test_y      = test_y.astype('float32')

    train_meta_day_time  = to_categorical(train_meta_day_time,3)
    train_meta_camera    = to_categorical(train_meta_camera,3)
    train_meta_is_sun    = to_categorical(train_meta_is_sun,2)
    test_meta_day_time   = to_categorical(test_meta_day_time,3)
    test_meta_camera     = to_categorical(test_meta_camera,3)
    test_meta_is_sun     = to_categorical(test_meta_is_sun,2)

    
    Ttrain_data = train_data[0:q,:,:,:]
    Ttrain_y    = train_y[0:q]
    Ttrain_meta_day_time  = train_meta_day_time[0:q,:]
    Ttrain_meta_camera    = train_meta_camera[0:q,:]
    Ttrain_meta_is_sun    = train_meta_is_sun[0:q,:]
                  
        
    # import network (from scratch)
    input = Input(shape=(224, 224, 3))
    old_model = InceptionV3(weights='imagenet', include_top=False)
    
    x = old_model.layers[-1].output
    x = GlobalAveragePooling2D()(x)
    img_out = Dense(200, activation='relu')(x)
    img_model = Model(inputs=old_model.input, outputs=img_out)

    meta_input1 = Input(shape=(3,), dtype='int32')
    meta1 = Embedding(3, 50, input_length=3)(meta_input1)
    meta1 = Flatten()(meta1)
    meta_out1 = Dense(50, activation='relu')(meta1)
    meta_model1 = Model(meta_input1, meta_out1)

    meta_input2 = Input(shape=(3,), dtype='int32')
    meta2 = Embedding(3, 50, input_length=3)(meta_input2)
    meta2 = Flatten()(meta2)
    meta_out2 = Dense(50, activation='relu')(meta2)
    meta_model2 = Model(meta_input2, meta_out2)

    meta_input3 = Input(shape=(2,), dtype='int32')
    meta3 = Embedding(2, 40, input_length=2)(meta_input3)
    meta3 = Flatten()(meta3)
    meta_out3 = Dense(40, activation='relu')(meta3)
    meta_model3 = Model(meta_input3, meta_out3)

    concatenated = concatenate([img_out, meta_out1, meta_out2, meta_out3], axis=-1)
    out = Dense(1, activation='sigmoid')(concatenated)

    merged_model = Model([old_model.input, meta_input1, meta_input2, meta_input3], out)

    for layer in old_model.layers:
        layer.trainable = True

    for layer in merged_model.layers:
        layer.trainable = True

    merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    merged_model.fit([Ttrain_data, Ttrain_meta_day_time, Ttrain_meta_camera, Ttrain_meta_is_sun],
                     y=Ttrain_y, batch_size=batch_size, epochs=epochs, verbose=1, 
                     validation_data=([test_data, test_meta_day_time, test_meta_camera, test_meta_is_sun], test_y))

    merged_model.save('binary_model_' + target + '_' + str(w+1) + '.h5')
    
    pred = merged_model.predict([test_data, test_meta_day_time, test_meta_camera, test_meta_is_sun])
    ap   = average_precision_score(test_y, pred)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    cm = confusion_matrix(test_y, pred)
    if not os.path.exists(path_save + '/metadata/cm/'):
        os.makedirs(path_save + '/metadata/cm/')
    if not os.path.exists(path_save + '/metadata/f1/'):
        os.makedirs(path_save + '/metadata/f1/')

    print(cm)
    print([ap,0])
    pd.DataFrame(cm).to_csv(path_save + '/metadata/cm/' + target + '_metadata_cm_' + str(q) + '_' + str(w+1) + '.csv')
    pd.DataFrame([ap,0]).to_csv(path_save + '/metadata/f1/' + target + '_metadata_F1_' + str(q) + '_' + str(w+1) + '.csv')
    
    
    # import network (from scratch)
    #input = Input(shape=(224, 224, 3))
    #old_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = old_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a layer for classification
    predictions = Dense(1, activation='sigmoid')(x)

    new_model = Model(inputs=old_model.input, outputs=predictions)

    for layer in old_model.layers:
        layer.trainable = False

    for layer in new_model.layers:
        layer.trainable = True
        
    new_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy']) # classification

    new_model.fit(Ttrain_data, Ttrain_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_data, test_y))

    pred = new_model.predict(test_data)
    ap   = average_precision_score(test_y, pred)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    cm = confusion_matrix(test_y, pred)
    if not os.path.exists(path_save + '/regular/cm/'):
        os.makedirs(path_save + '/regular/cm/')
    if not os.path.exists(path_save + '/regular/f1/'):
        os.makedirs(path_save + '/regular/f1/')

    print(cm)
    print([ap,0])
    pd.DataFrame(cm).to_csv(path_save + '/regular/cm/' + target + '_cm_' + str(q) + '_' + str(w+1) + '.csv')
    pd.DataFrame([ap,0]).to_csv(path_save + '/regular/f1/' + target + '_F1_' + str(q) + '_' + str(w+1) + '.csv')        