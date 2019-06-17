# -*- coding: utf-8 -*-
"""
@author: dinda
python 3.6.8
"""

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from os import listdir, path

from matplotlib import pyplot as plt

from sklearn import metrics

import tensorflow as tf

from keras.utils import np_utils
from keras.utils import to_categorical

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Masking, multiply
from keras.layers import Input, LSTM, concatenate, Activation
from keras.models import Model, load_model
from keras.optimizers import Adam

class Soteria:
    """
    Soteria: Safety Classification Engine
    Covers:
        1. Feature Engineering
        2. Building model
        3. Do classification
    """
    
    def __init__(self):
        self.epochs = 50
        self.batch_size = 128
        self.monitor="loss"
        self.optimization_mode='auto'
        self.max_timesteps  = 120
        
    def load_dataset(self, feature_dir, label_dir):        
        feature_filepaths = [path.join(feature_dir, f) for f in listdir(feature_dir) if f.endswith('.csv')]
        features = pd.concat(map(pd.read_csv, feature_filepaths)).sort_values(by=["bookingID","second"]).reset_index(drop=True)
        
        label_filepaths = [path.join(label_dir, f) for f in listdir("./{}".format(label_dir)) if f.endswith('.csv')]
        labels = pd.concat(map(pd.read_csv, label_filepaths))
        labels = labels.drop_duplicates(subset = 'bookingID', keep = False).sort_values(by="bookingID").reset_index(drop=True)
        self.num_classes = len(np.unique(labels["label"]))        
        return features, labels
    
    def add_features(self, X):
        X["distance"] = X["second"] * X["Speed"]
        return X

    def cutoff_dataset(self, df, label_list):
        segments = []
        labels = []
        # cut every booking_id data into 120 time steps
        for bid in label_list["bookingID"]:
            bid_data = df[df['bookingID']==bid].drop(['bookingID', 'second'], axis=1).reset_index(drop=True)
            cur_label = label_list["label"][label_list["bookingID"]==bid].values[0]
            data = []
            for key in bid_data.keys():
                data.append(bid_data[key].values[0:self.max_timesteps])
            segments.append(data)
            labels.append(cur_label)
        segments = np.asarray(segments, dtype= np.float32)
        labels = np.asarray(labels)
        self.max_num_variables = segments.shape[1]
        return segments, labels
    
    def generate_model(self):
        
        def squeeze_excite_block(input):
            ''' Create a squeeze-excite block
            Args:
                input: input tensor
                filters: number of output filters
                k: width factor
        
            Returns: a keras tensor
            '''
            filters = input._keras_shape[-1] # channel_axis = -1 for TF
        
            se = GlobalAveragePooling1D()(input)
            se = Reshape((1, filters))(se)
            se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
            se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
            se = multiply([input, se])
            return se
        
        ip = Input(shape=(self.max_num_variables, self.max_timesteps))
    
        x = Masking()(ip)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)
    
        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)
    
        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)
    
        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
    
        y = GlobalAveragePooling1D()(y)
    
        x = concatenate([x, y])
    
        out = Dense(self.num_classes, activation='softmax')(x)
    
        model = Model(ip, out)
        optm = Adam(lr=1e-3)
        model.compile(optimizer=optm, loss="categorical_crossentropy", metrics=[self.jacek_auc])
        model.summary()
        return model
    
    def jacek_auc(self, y_true, y_pred):
        # FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108
        # Use AUC instead of Accuracy since our dataset is imbalance
        score, up_opt = tf.metrics.auc(y_true, y_pred)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        return score    
    
    def train_test_split(self, X, y, ratio):
        bid_lim = y.loc[int(ratio*len(y)), "bookingID"]
        label_train = y[y["bookingID"]<bid_lim]
        label_test = y[y["bookingID"]>=bid_lim].reset_index(drop=True)
        data_train = X[X["bookingID"]<bid_lim]
        data_test = X[X["bookingID"]>=bid_lim].reset_index(drop=True)
        return data_train, data_test, label_train, label_test
    
    def train_model(self, model:Model, X_train,y_train, model_name):

        factor = 1. / np.cbrt(2)
        
        y_train_hot = np_utils.to_categorical(y_train, self.num_classes)
        
        le = LabelEncoder()
        y_ind = le.fit_transform(y_train.ravel())
        recip_freq = len(y_train_hot) / (len(le.classes_) *
                                   np.bincount(y_ind).astype(np.float64))
        class_weight = recip_freq[le.transform(np.unique(y_train_hot))]
    

        weight_fn = '{}.h5'.format(model_name)
        model_checkpoint = ModelCheckpoint(weight_fn, verbose=1, mode=self.optimization_mode,
                                           monitor=self.monitor, save_best_only=True, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=100, mode=self.optimization_mode,
                                      factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
        callback_list = [model_checkpoint, reduce_lr]
    
        
        model.fit(X_train, y_train_hot, batch_size=self.batch_size, epochs=self.epochs, callbacks=callback_list,
                  class_weight=class_weight, verbose=1, validation_split=0.2)
        
        return model
    
    def draw_roc_curve(self, true_label, pred_prob):
        fpr, tpr, _ = metrics.roc_curve(true_label, pred_prob, pos_label = 1)        
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='MLSTM-FCN')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()        
        
    def run_engine(self, feature_dir, label_dir): 
        #train_stat = json.load(open("train_stat.json","r"))
        num_classes = 2
        #load dataset
        X, y = self.load_dataset(feature_dir, label_dir)
        #add feature
        X = self.add_features(X)
        #cutoff dataset
        X, y = self.cutoff_dataset(X, y)
        #scale dataset
        #X = (X - np.float32(train_stat["X_train_mean"])) / (np.float32(train_stat["X_train_std"]) + 1e-8)
        #y = (y - y.min()) / (y.max() - y.min()) * (num_classes - 1)
        #load model
        model = self.generate_model()
        model.load_weights('model/model_tr_weight.h5')
        #do prediction
        _, auc_score = model.evaluate(X, to_categorical(y, num_classes), batch_size = self.batch_size)
        print("Jacek AUC Score: {}".format(auc_score))
        y_pred = model.predict(X)
        y_pred_prob = np.max(y_pred, axis=1)
        print("sklearn AUC score: {}".format(metrics.roc_auc_score(y, y_pred_prob)))            
    
 
        
