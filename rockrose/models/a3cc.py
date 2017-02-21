"""
"""

import json
import math
import random

import numpy as np

from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.layers.core import Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

from keras import backend as K

import base as rr_model_base


def rr_fn_conv_w_init(shape, name, scl=0.01, *args, **kwargs):
    return normal(shape, scale=scl, name=name)


class RRModelA3CConvPV(rr_model_base.RRModelBase):
    def __init__(self, cfg={}, *args, **kwargs):
        super(RRModelA3CConvPV, self).__init__(cfg, *args, **kwargs)

        self.input_shape = self.cfg.get('input_shape', [1])
        self.actn = self.cfg.get('actn', 1)
        self.lr = self.cfg.get('lr', 1e-6)
        self.loss_name = self.cfg.get('loss_name', 'mse')
        self.batch_n = self.cfg.get('batch_n', 32)
        self.p_entropy_beta = self.cfg.get('p_entropy_beta', 0.01)

        self.model_init()

        self.graph_init()


    def model_init__1(self):

        inputs = Input(shape=self.input_shape)
        shared = Convolution2D(name="conv1", nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
        shared = Convolution2D(name="conv2", nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(shared)
        shared = Flatten()(shared)
        shared = Dense(name="h1", output_dim=256, activation='relu')(shared)

        action_probs = Dense(name="p", output_dim=self.actn, activation='softmax')(shared)
        state_value = Dense(name="v", output_dim=1, activation='linear')(shared)

        self.policy_network = Model(input=inputs, output=action_probs)
        self.value_network = Model(input=inputs, output=state_value)
       

        #self.R_t = K.variable(0.0, name='R_t')
        #self.a_t = K.variable(K.zeros((self.actn,)), name='a_t')
        #self.v_t = K.variable(0.0, name='v_t')
        #x#self.R_t = K.variable(K.zeros((self.batch_n,)), name='R_t')
        #x#self.a_t = K.variable(K.zeros((self.batch_n, self.actn)), name='a_t')
        #x#self.v_t = K.variable(K.zeros((self.batch_n,)), name='v_t')
        self.R_t = K.variable(np.zeros((self.batch_n,)), name='R_t')
        self.a_t = K.variable(np.zeros((self.batch_n, self.actn)), name='a_t')
        self.v_t = K.variable(np.zeros((self.batch_n,)), name='v_t')

        adam = Adam(lr=self.lr)

        p_loss = self.a3c_p_loss(self.policy_network, self.value_network, self.R_t, self.a_t, self.v_t)
        v_loss = self.a3c_v_loss(self.policy_network, self.value_network, self.R_t)

        self.policy_network.compile(loss=p_loss, optimizer=adam)
        self.value_network.compile(loss=v_loss, optimizer=adam)

        return self.policy_network, self.value_network


    def model_init(self):

        #in_img = Input(shape=self.input_shape)

        conv_model = Sequential(name='conv_model')
        conv_model.add(Convolution2D(16, 8, 8, subsample=(4, 4),
            init=rr_fn_conv_w_init, border_mode='same',
            input_shape=self.input_shape,
            activation='relu', name='conv1'))
        conv_model.add(Convolution2D(32, 4, 4, subsample=(2, 2),
            init=rr_fn_conv_w_init, border_mode='same',
            activation='relu', name='conv2'))
        conv_model.add(Flatten())

        base_model = Sequential(name='base_model')
        base_model.add(conv_model)
        base_model.add(Dense(256, activation='relu', name='h1'))

        ##in_last_action_reward = Input(shape=(act_n + 1,))
        #last_a_r_model = Sequential(name='last_a_r_model')
        #last_a_r_model.add(Flatten(input_shape=(1, act_n + 1)))
        #merged_model = Sequential(name='merged_model')
        #merged_model.add(Merge([base_model, last_a_r_model], mode='concat', concat_axis=-1))
        #merged_model.add(Reshape((1, 256 + self.actn + 1)))
        #merged_model.add(LSTM(256, name='lstm1'))  # TODO:

        policy_network = Sequential(name='policy_network')
        #policy_network.add(merged_model)
        policy_network.add(base_model)
        policy_network.add(Dense(self.actn, activation='softmax', name='p'))

        value_network = Sequential(name='value_network')
        #value_network.add(merged_model)
        value_network.add(base_model)
        value_network.add(Dense(1, activation='linear', name='v'))

        self.policy_network = policy_network
        self.value_network = value_network

        self.R_t = K.variable(np.zeros((self.batch_n,)), name='R_t')
        self.a_t = K.variable(np.zeros((self.batch_n, self.actn)), name='a_t')
        self.v_t = K.variable(np.zeros((self.batch_n,)), name='v_t')

        adam = Adam(lr=self.lr)

        p_loss = self.a3c_p_loss(self.policy_network, self.value_network, self.R_t, self.a_t, self.v_t)
        #p_loss = self.a3c_p_loss_ent(self.policy_network, self.value_network, self.R_t, self.a_t, self.v_t)
        v_loss = self.a3c_v_loss(self.policy_network, self.value_network, self.R_t)

        self.policy_network.compile(loss=p_loss, optimizer=adam)
        self.value_network.compile(loss=v_loss, optimizer=adam)

        return self.policy_network, self.value_network


    def a3c_p_loss_ent(self, p_network, v_network, R_t, a_t, v_t):

        def _inner_loss(y_true, y_pred, *args, **kwargs):
            """
            """
            y_clip = K.clip(y_pred, 1e-20, 1.0)

            log_pi = K.log(y_clip)

            log_prob = K.sum(log_pi * a_t, axis=1)

            entropy = K.sum(y_clip * log_pi, axis=1)

            td = (R_t - v_t)

            p_loss = -log_prob * td

            p_loss = p_loss + entropy * self.p_entropy_beta

            return p_loss

        return _inner_loss


    def a3c_p_loss(self, p_network, v_network, R_t, a_t, v_t):

        def _inner_loss(y_true, y_pred, *args, **kwargs):
            """
            TODO: use like keras.objectives.binary_crossentropy()
            """

            y_clip = K.clip(y_pred, 1e-20, 1.0)
            log_prob = K.log(K.sum(y_clip * a_t, axis=1))
            p_loss = -log_prob * (R_t - v_t)

            v_loss = K.mean(K.square(R_t - v_t))

            return p_loss

        return _inner_loss


    def a3c_v_loss(self, p_network, v_network, R_t):

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            #o#v_loss = K.mean(K.square(R_t - v_t))
            v_loss = K.mean(K.square(y_true - y_pred))
            v_loss = 0.5 * v_loss

            return v_loss

        return _inner_loss


    def a3c_pv_loss__1(self, p_network, v_network, R_t, a_t):

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            #R_t = tf.placeholder("float", [None])
            #a_t = tf.placeholder("float", [None, ACTIONS])

            log_prob = K.log(K.sum(K.mul(p_network, a_t), reduction_indices=1))
            p_loss = -log_prob * (R_t - v_network)
            v_loss = K.mean(K.square(R_t - v_network))  # TODO: could we use v_network ???

            total_loss = p_loss + (0.5 * v_loss)
            return total_loss

        return _inner_loss


    def a3c_pv_loss(self, p_network, v_network, R_t, a_t, v_t):

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            y_clip = K.clip(y_pred, 1e-20, 1.0)
            log_prob = K.log(K.sum(y_clip * a_t, axis=1))
            p_loss = -log_prob * (R_t - v_t)

            v_loss = K.mean(K.square(R_t - v_t))

            total_loss = p_loss + (0.5 * v_loss)
            return total_loss

        return _inner_loss


    def graph_init(self):
        """Init the global graph if backend is tensorflow.

        File "rockrose/trainers/a3c.py", line 167, in train_a_thread
            probs = self.model.policy_network.predict(s_t)[0]
        ......
        ValueError: Tensor Tensor("Softmax:0", shape=(?, 5), dtype=float32) is not an element of this graph.

        >>> https://github.com/fchollet/keras/issues/2397
        """

        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            self.global_graph = tf.get_default_graph()
        else:
            self.global_graph = None


    def set_value(self, *args, **kwargs):
        #K.set_value(*args, **kwargs)

        if self.global_graph:
            with self.global_graph.as_default():
                K.set_value(*args, **kwargs)
        else:
            K.set_value(*args, **kwargs)


    def load(self, model_file_p, model_file_v):

        self.policy_network.load_weights(model_file_p)
        self.value_network.load_weights(model_file_v)

        adam = Adam(lr=self.lr)

        #p_loss = self.a3c_p_loss(self.policy_network, self.value_network, self.R_t, self.a_t, self.v_t)
        p_loss = self.a3c_p_loss_ent(self.policy_network, self.value_network, self.R_t, self.a_t, self.v_t)
        v_loss = self.a3c_v_loss(self.policy_network, self.value_network, self.R_t)

        self.policy_network.compile(loss=p_loss, optimizer=adam)
        self.value_network.compile(loss=v_loss, optimizer=adam)


    def save(self, model_file_p, model_file_v):
        self.policy_network.save(model_file_p, overwrite=True)

        self.value_network.save(model_file_v, overwrite=True)


    def save_weights(self, model_file_p, model_file_v, model_json_p=None, model_json_v=None):
        self.policy_network.save_weights(model_file_p, overwrite=True)
        if model_json_p:
            with open(model_json_p, "w") as outfile:
                json.dump(self.policy_network.to_json(), outfile)

        self.value_network.save_weights(model_file_v, overwrite=True)
        if model_json_v:
            with open(model_json_v, "w") as outfile:
                json.dump(self.value_network.to_json(), outfile)


    def predict_p(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.policy_network.predict(*args, **kwargs)
        else:
            r = self.policy_network.predict(*args, **kwargs)
        return r


    def predict_v(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.value_network.predict(*args, **kwargs)
        else:
            r = self.value_network.predict(*args, **kwargs)
        return r


    def train_on_batch_p(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.policy_network.train_on_batch(*args, **kwargs)
        else:
            r = self.policy_network.train_on_batch(*args, **kwargs)
        return r


    def train_on_batch_v(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.value_network.train_on_batch(*args, **kwargs)
        else:
            r = self.value_network.train_on_batch(*args, **kwargs)
        return r


    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


    def train_on_batch(self, *args, **kwargs):
        return self.model.train_on_batch(*args, **kwargs)
