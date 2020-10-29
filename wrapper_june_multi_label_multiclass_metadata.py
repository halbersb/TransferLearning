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

num_classes = 3
path        = '/home/danh/Data/transfer_learning/Experiment_11-06-19/for_experiment'
path_label  = '/home/danh/Data/transfer_learning/Experiment_11-06-19/labels_classification_11062019_3_classes_equal_bin.csv'
now         = datetime.datetime.now()
path_save   = '/home/danh/Results/Experiment_11-06-19/classification/multilabel/' + '12252019'
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
y1    = []
y2    = []
y3    = []
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
                y1.append(float(val['chroma']))
                y2.append(float(val['hue']))
                y3.append(float(val['shine']))
                ID.append(temp[0]) #split train/test based on ID
                meta_crop_type.append(int(temp[1])-1)
                meta_light.append([float(i) for i in temp[2:]])

for w in range(0, 10):

    train_ID = pd.read_csv(path_save_mask + '/regular/mask/' + 'chroma' + '_train_mask_' + str(w + 1) + '.csv')
    test_ID = pd.read_csv(path_save_mask + '/regular/mask/' + 'chroma' + '_test_mask_' + str(w + 1) + '.csv')

    train_ID = train_ID.iloc[:, 1].values.tolist()
    test_ID = test_ID.iloc[:, 1].values.tolist()

    trainInd = []
    testInd = []
    for i in range(0, len(ID)):
        if int(ID[i]) in train_ID:
            trainInd.append(i)
        if int(ID[i]) in test_ID:
            testInd.append(i)

    trainInd = np.asarray(trainInd)
    testInd = np.asarray(testInd)
    train_data = np.array(data)[trainInd.astype(int)]
    test_data = np.array(data)[testInd.astype(int)]
    train_y1 = np.array(y1)[trainInd.astype(int)]
    test_y1 = np.array(y1)[testInd.astype(int)]
    train_y2 = np.array(y2)[trainInd.astype(int)]
    test_y2 = np.array(y2)[testInd.astype(int)]
    train_y3 = np.array(y3)[trainInd.astype(int)]
    test_y3 = np.array(y3)[testInd.astype(int)]
    train_meta_day_time = np.array(meta_day_time)[trainInd.astype(int)]
    test_meta_day_time = np.array(meta_day_time)[testInd.astype(int)]
    train_meta_camera = np.array(meta_camera)[trainInd.astype(int)]
    test_meta_camera = np.array(meta_camera)[testInd.astype(int)]
    train_meta_is_sun = np.array(meta_is_sun)[trainInd.astype(int)]
    test_meta_is_sun = np.array(meta_is_sun)[testInd.astype(int)]
    train_meta_crop_type = np.array(meta_crop_type)[trainInd.astype(int)]
    test_meta_crop_type = np.array(meta_crop_type)[testInd.astype(int)]
    train_meta_light = np.array(meta_light)[trainInd.astype(int)]
    test_meta_light = np.array(meta_light)[testInd.astype(int)]

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data /= 255  # normalize 0-255 => 0-1
    test_data /= 255  # normalize 0-255 => 0-1
    train_y1 = train_y1.astype('float32')
    test_y1 = test_y1.astype('float32')
    train_y1 = keras.utils.to_categorical(train_y1, num_classes)
    test_y1 = keras.utils.to_categorical(test_y1, num_classes)
    train_y2 = train_y2.astype('float32')
    test_y2 = test_y2.astype('float32')
    train_y2 = keras.utils.to_categorical(train_y2, num_classes)
    test_y2 = keras.utils.to_categorical(test_y2, num_classes)
    train_y3 = train_y3.astype('float32')
    test_y3 = test_y3.astype('float32')
    train_y3 = keras.utils.to_categorical(train_y3, num_classes)
    test_y3 = keras.utils.to_categorical(test_y3, num_classes)

    train_y = np.concatenate((train_y1, train_y2, train_y3), axis=1)
    test_y = np.concatenate((test_y1, test_y2, test_y3), axis=1)

    train_meta_day_time = to_categorical(train_meta_day_time)
    train_meta_camera = to_categorical(train_meta_camera)
    train_meta_is_sun = to_categorical(train_meta_is_sun)
    train_meta_crop_type = to_categorical(train_meta_crop_type)

    test_meta_day_time = to_categorical(test_meta_day_time)
    test_meta_camera = to_categorical(test_meta_camera)
    test_meta_is_sun = to_categorical(test_meta_is_sun)
    test_meta_crop_type = to_categorical(test_meta_crop_type)

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
    out = Dense(3 * num_classes, activation='sigmoid')(concatenated)

    merged_model = Model([old_model.input, meta_input1, meta_input2, meta_input3, meta_input4], out)

    for layer in old_model.layers:
        layer.trainable = True

    for layer in merged_model.layers:
        layer.trainable = True

    merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    merged_model.fit([train_data, train_meta_day_time, train_meta_camera, train_meta_is_sun, train_meta_crop_type],
                     y=train_y, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(
                     [test_data, test_meta_day_time, test_meta_camera, test_meta_is_sun, test_meta_crop_type], test_y))

    pred = merged_model.predict(
        [test_data, test_meta_day_time, test_meta_camera, test_meta_is_sun, test_meta_crop_type]).round()
    # y1 - chroma
    pred1 = np.argmax(pred[:, 0:3], axis=1)
    test_y1 = np.argmax(test_y1, axis=1)
    cm1 = confusion_matrix(test_y1, pred1)
    # y2 - hue
    pred2 = np.argmax(pred[:, 3:6], axis=1)
    test_y2 = np.argmax(test_y2, axis=1)
    cm2 = confusion_matrix(test_y2, pred2)
    # y3 - shine
    pred3 = np.argmax(pred[:, 6:9], axis=1)
    test_y3 = np.argmax(test_y3, axis=1)
    cm3 = confusion_matrix(test_y3, pred1)

    if not os.path.exists(path_save + '/chroma/metadata/cm/'):
        os.makedirs(path_save + '/chroma/metadata/cm/')
    if not os.path.exists(path_save + '/chroma/metadata/f1/'):
        os.makedirs(path_save + '/chroma/metadata/f1/')
    if not os.path.exists(path_save + '/hue/metadata/cm/'):
        os.makedirs(path_save + '/hue/metadata/cm/')
    if not os.path.exists(path_save + '/hue/metadata/f1/'):
        os.makedirs(path_save + '/hue/metadata/f1/')
    if not os.path.exists(path_save + '/shine/metadata/cm/'):
        os.makedirs(path_save + '/shine/metadata/cm/')
    if not os.path.exists(path_save + '/shine/metadata/f1/'):
        os.makedirs(path_save + '/shine/metadata/f1/')

    print(cm1)
    pd.DataFrame(cm1).to_csv(path_save + '/chroma/metadata/cm/chroma_metadata_cm_' + str(w + 1) + '.csv')
    pd.DataFrame(f1_score(test_y1, pred1, average=None)).to_csv(
        path_save + '/chroma/metadata/f1/chroma_metadata_F1_' + str(w + 1) + '.csv')
    print(cm2)
    pd.DataFrame(cm2).to_csv(path_save + '/hue/metadata/cm/hue_metadata_cm_' + str(w + 1) + '.csv')
    pd.DataFrame(f1_score(test_y2, pred2, average=None)).to_csv(
        path_save + '/hue/metadata/f1/hue_metadata_F1_' + str(w + 1) + '.csv')
    print(cm3)
    pd.DataFrame(cm3).to_csv(path_save + '/shine/metadata/cm/shine_metadata_cm_' + str(w + 1) + '.csv')
    pd.DataFrame(f1_score(test_y3, pred3, average=None)).to_csv(
        path_save + '/shine/metadata/f1/shine_metadata_F1_' + str(w + 1) + '.csv')

    # import network (from scratch)
    # input = Input(shape=(224, 224, 3))
    # old_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = old_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a layer for classification
    predictions = Dense(3 * num_classes, activation='sigmoid')(x)

    new_model = Model(inputs=old_model.input, outputs=predictions)

    for layer in old_model.layers:
        layer.trainable = True

    for layer in new_model.layers:
        layer.trainable = True

    new_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])  # classification

    new_model.fit(train_data, train_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(test_data, test_y))

    pred = new_model.predict(test_data).round()
    # y1 - chroma
    pred1 = np.argmax(pred[:, 0:3], axis=1)
    cm1 = confusion_matrix(test_y1, pred1)
    # y2 - hue
    pred2 = np.argmax(pred[:, 3:6], axis=1)
    cm2 = confusion_matrix(test_y2, pred2)
    # y3 - shine
    pred3 = np.argmax(pred[:, 6:9], axis=1)
    cm3 = confusion_matrix(test_y3, pred1)

    if not os.path.exists(path_save + '/chroma/regular/cm/'):
        os.makedirs(path_save + '/chroma/regular/cm/')
    if not os.path.exists(path_save + '/chroma/regular/f1/'):
        os.makedirs(path_save + '/chroma/regular/f1/')
    if not os.path.exists(path_save + '/hue/regular/cm/'):
        os.makedirs(path_save + '/hue/regular/cm/')
    if not os.path.exists(path_save + '/hue/regular/f1/'):
        os.makedirs(path_save + '/hue/regular/f1/')
    if not os.path.exists(path_save + '/shine/regular/cm/'):
        os.makedirs(path_save + '/shine/regular/cm/')
    if not os.path.exists(path_save + '/shine/regular/f1/'):
        os.makedirs(path_save + '/shine/regular/f1/')

    print(cm1)
    pd.DataFrame(cm1).to_csv(path_save + '/chroma/regular/cm/chroma_cm_' + str(w + 1) + '.csv')
    pd.DataFrame(f1_score(test_y1, pred1, average=None)).to_csv(
        path_save + '/chroma/regular/f1/chroma_F1_' + str(w + 1) + '.csv')
    print(cm2)
    pd.DataFrame(cm2).to_csv(path_save + '/hue/regular/cm/hue_cm_' + str(w + 1) + '.csv')
    pd.DataFrame(f1_score(test_y2, pred2, average=None)).to_csv(
        path_save + '/hue/regular/f1/hue_F1_' + str(w + 1) + '.csv')
    print(cm3)
    pd.DataFrame(cm3).to_csv(path_save + '/shine/regular/cm/shine_cm_' + str(w + 1) + '.csv')
    pd.DataFrame(f1_score(test_y3, pred3, average=None)).to_csv(
        path_save + '/shine/regular/f1/shine_F1_' + str(w + 1) + '.csv')