from __future__ import print_function
import csv
import numpy as np
import datetime
import pandas as pd 
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

size = 224,224
batch_size = 20
epochs = 200

target = 'chroma'
num_classes = 3
path       = '/home/danh/Data/transfer_learning/Experiment_11-06-19/for_experiment'
path_label = '/home/danh/Data/transfer_learning/Experiment_11-06-19/labels_classification_11062019_3_classes_equal_bin.csv'
now        = datetime.datetime.now()
path_save  = '/home/danh/Results/Experiment_11-06-19/classification/multiclass/' + '12242019' + '/' + target
path_save_mask  = '/home/danh/Results/Experiment_11-06-19/classification/multiclass/' + '12242019' + '/' + 'chroma'

# init metadata labels
meta_day_time  = [] # Morning=task_483=0, Noon=task_492=1, Afternoon=task_493=2
meta_camera    = [] # RGB=high=0, 3D=low=1
meta_is_sun    = [] # Shadow=0, Sun=1
meta_crop_type = [] # one of 4 values
meta_light     = [] # lightsensor values - vector of length 8
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
                meta_crop_type.append(int(temp[1])-1)
                meta_light.append([float(i) for i in temp[2:]])
				

for w in range(0, 10):
    # split to train-test 80-20
    unique_ID = np.asarray(list(set(ID)))
    msk       = np.random.rand(unique_ID.shape[0]) < 0.8 # random split
    train_ID  = unique_ID[msk==True]
    test_ID   = unique_ID[msk==False]
   
    if not os.path.exists(path_save_mask + '/regular/mask/'):
        os.makedirs(path_save_mask + '/regular/mask/')
    pd.DataFrame(train_ID).to_csv(path_save_mask + '/regular/mask/' + target + '_train_mask_' + str(w+1) + '.csv')
    pd.DataFrame(test_ID).to_csv(path_save_mask + '/regular/mask/' + target + '_test_mask_' + str(w+1) + '.csv')
	

for w in range(0, 10):
    
    train_ID  = pd.read_csv(path_save_mask + '/regular/mask/' + target + '_train_mask_' + str(w+1) + '.csv')
    test_ID   = pd.read_csv(path_save_mask + '/regular/mask/' + target + '_test_mask_' + str(w+1) + '.csv')
    
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
    train_meta_crop_type=np.array(meta_crop_type)[trainInd.astype(int)]
    test_meta_crop_type=np.array(meta_crop_type)[testInd.astype(int)]
    train_meta_light=np.array(meta_light)[trainInd.astype(int)]
    test_meta_light=np.array(meta_light)[testInd.astype(int)]

    train_data  = train_data.astype('float32')
    test_data   = test_data.astype('float32')
    train_data /= 255 #normalize 0-255 => 0-1
    test_data  /= 255 #normalize 0-255 => 0-1
    train_y     = train_y.astype('float32')
    test_y      = test_y.astype('float32')
    train_y     = keras.utils.to_categorical(train_y, num_classes)
    test_y      = keras.utils.to_categorical(test_y, num_classes)

    train_meta_day_time  = to_categorical(train_meta_day_time)
    train_meta_camera    = to_categorical(train_meta_camera)
    train_meta_is_sun    = to_categorical(train_meta_is_sun)
    train_meta_crop_type = to_categorical(train_meta_crop_type)

    test_meta_day_time   = to_categorical(test_meta_day_time)
    test_meta_camera     = to_categorical(test_meta_camera)
    test_meta_is_sun     = to_categorical(test_meta_is_sun)
    test_meta_crop_type  = to_categorical(test_meta_crop_type)

    
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

    meta_input2 = Input(shape=(2,), dtype='int32')
    meta2 = Embedding(2, 40, input_length=2)(meta_input2)
    meta2 = Flatten()(meta2)
    meta_out2 = Dense(40, activation='relu')(meta2)
    meta_model2 = Model(meta_input2, meta_out2)

    meta_input3 = Input(shape=(2,), dtype='int32')
    meta3 = Embedding(2, 40, input_length=2)(meta_input3)
    meta3 = Flatten()(meta3)
    meta_out3 = Dense(40, activation='relu')(meta3)
    meta_model3 = Model(meta_input3, meta_out3)

    meta_input4 = Input(shape=(4,), dtype='int32')
    meta4 = Embedding(3, 60, input_length=4)(meta_input4)
    meta4 = Flatten()(meta4)
    meta_out4 = Dense(60, activation='relu')(meta4)
    meta_model4 = Model(meta_input4, meta_out4)

    concatenated = concatenate([img_out, meta_out1, meta_out2, meta_out3, meta_out4], axis=-1)
    out = Dense(num_classes, activation='softmax')(concatenated)

    merged_model = Model([old_model.input, meta_input1, meta_input2, meta_input3, meta_input4], out)

    for layer in old_model.layers:
        layer.trainable = True
    
    for layer in merged_model.layers:
        layer.trainable = True

    merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    merged_model.fit([train_data, train_meta_day_time, train_meta_camera, train_meta_is_sun, train_meta_crop_type],
                     y=train_y, batch_size=batch_size, epochs=epochs, verbose=1, 
                     validation_data=([test_data, test_meta_day_time, test_meta_camera, test_meta_is_sun, test_meta_crop_type], test_y))

    merged_model.save('multiclass_model_' + target + '_' + str(w+1) + '.h5')
    
    pred = merged_model.predict([test_data, test_meta_day_time, test_meta_camera, test_meta_is_sun, test_meta_crop_type])
    pred_y = np.argmax(pred, axis=1)
    test_y2 = np.argmax(test_y, axis=1)
    cm = confusion_matrix(test_y2, pred_y)
    if not os.path.exists(path_save + '/metadata/cm/'):
        os.makedirs(path_save + '/metadata/cm/')
    if not os.path.exists(path_save + '/metadata/f1/'):
        os.makedirs(path_save + '/metadata/f1/')
    
    print(cm)
    print(f1_score(test_y2, pred_y, average=None))
    pd.DataFrame(cm).to_csv(path_save + '/metadata/cm/' + target + '_metadata_cm_' + str(w+1) + '.csv')
    pd.DataFrame(f1_score(test_y2, pred_y, average=None)).to_csv(path_save + '/metadata/f1/' + target + '_metadata_F1_' + str(w+1) + '.csv')
    
    
    # import network (from scratch)
    #input = Input(shape=(224, 224, 3))
    #old_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = old_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a layer for classification
    predictions = Dense(num_classes, activation='softmax')(x)

    new_model = Model(inputs=old_model.input, outputs=predictions)

    for layer in old_model.layers:
        layer.trainable = True

    for layer in new_model.layers:
        layer.trainable = True
        
    new_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) # classification

    new_model.fit(train_data, train_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_data, test_y))

    pred = new_model.predict(test_data)
    pred_y = np.argmax(pred, axis=1)
    test_y2 = np.argmax(test_y, axis=1)
    cm = confusion_matrix(test_y2, pred_y)
    if not os.path.exists(path_save + '/regular/cm/'):
        os.makedirs(path_save + '/regular/cm/')
    if not os.path.exists(path_save + '/regular/f1/'):
        os.makedirs(path_save + '/regular/f1/')
    
    print(cm)
    print(f1_score(test_y2, pred_y, average=None))
    pd.DataFrame(cm).to_csv(path_save + '/regular/cm/' + target + '_cm_' + str(w+1) + '.csv')
    pd.DataFrame(f1_score(test_y2, pred_y, average=None)).to_csv(path_save + '/regular/f1/' + target + '_F1_' + str(w+1) + '.csv')
