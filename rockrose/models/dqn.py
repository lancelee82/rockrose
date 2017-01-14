"""
"""

import json

from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

import base as rr_model_base


def rr_fn_conv_w_init(shape, name, scl=0.01, *args, **kwargs):
    return normal(shape, scale=scl, name=name)


class RRModelDQNConv(rr_model_base.RRModelBase):
    def __init__(self, cfg={}, *args, **kwargs):
        super(RRModelDQNConv, self).__init__(cfg, *args, **kwargs)

        self.input_shape = self.cfg.get('input_shape', [1])
        self.actn = self.cfg.get('actn', 1)
        self.lr = self.cfg.get('lr', 1e-6)
        self.loss_name = self.cfg.get('loss_name', 'mse')

        self.model_init()


    def model_init__0(self):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4),
                  init=rr_fn_conv_w_init, border_mode='same',
                  input_shape=(img_channels,img_rows,img_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2),
                  init=rr_fn_conv_w_init, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1),
                  init=rr_fn_conv_w_init, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init=rr_fn_conv_w_init))
        model.add(Activation('relu'))
        model.add(Dense(actn, init=rr_fn_conv_w_init))

        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)

        return model


    def model_init(self):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4),
                  init=rr_fn_conv_w_init, border_mode='same',
                  input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2),
                  init=rr_fn_conv_w_init, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1),
                  init=rr_fn_conv_w_init, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512, init=rr_fn_conv_w_init))
        model.add(Activation('relu'))
        model.add(Dense(self.actn, init=rr_fn_conv_w_init))
       
        adam = Adam(lr=self.lr)
        model.compile(loss=self.loss_name, optimizer=adam)

        self.model = model

        return model


    def model_init__2o(self):

        conv_model = Sequential()
        conv_model.add(Convolution2D(16, 8, 8, subsample=(4, 4),
            init=rr_fn_conv_w_init, border_mode='same',
            input_shape=self.input_shape,
            activation='relu', name='conv1'))
        conv_model.add(Convolution2D(32, 4, 4, subsample=(2, 2),
            init=rr_fn_conv_w_init, border_mode='same',
            activation='relu', name='conv2'))
        conv_model.add(Flatten())

        base_model = Sequential()
        base_model.add(conv_model)
        base_model.add(Dense(256, activation='relu', name='h1'))

        model = Sequential()
        model.add(base_model)
        model.add(Dense(self.actn, activation='linear', name='Q'))
       
        adam = Adam(lr=self.lr)
        model.compile(loss=self.loss_name, optimizer=adam)

        self.model = model

        return model


    def load(self, model_file):
        self.model_file = model_file

        try:
            self.model.load_weights(model_file)
            adam = Adam(lr=self.lr)
            self.model.compile(loss=self.loss_name, optimizer=adam)
        except Exception as e:
            raise e

    def save(self, model_file, model_json=None):
        self.model.save_weights(model_file, overwrite=True)

        if model_json:
            with open(model_json, "w") as outfile:
                json.dump(self.model.to_json(), outfile)


    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def train_on_batch(self, *args, **kwargs):
        return self.model.train_on_batch(*args, **kwargs)
