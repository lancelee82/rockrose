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
from keras.engine.topology import Merge
from keras.layers import merge, Lambda
from keras.layers import Convolution2D, Deconvolution2D
from keras.layers import Flatten, Dense, Input
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD , Adam

from keras import backend as K

import base as rr_model_base


def rr_fn_conv_w_init(shape, name, scl=0.01, *args, **kwargs):
    return normal(shape, scale=scl, name=name)


def kr_merge_fn_mean(inputs, *args, **kwargs):
    # NOTE: we only use the first input !
    inpt = inputs[0]

    axis=kwargs.get('axis', 1)

    #outputs = K.mean(inpt, axis=3, keepdims=True)  # for K._BACKEND == 'th'
    #outputs = K.mean(inpt, axis=1, keepdims=True)  # for K._BACKEND == 'tf'
    outputs = K.mean(inpt, axis=axis, keepdims=True)
    return outputs


def kr_merge_fn_negative(inputs, *args, **kwargs):
    # NOTE: we only use the first input !
    inpt = inputs[0]

    outputs = inpt * (-1.0)
    return outputs


# TODO: use keras.layers.core.RepeatVector !?
def kr_merge_fn_repeat(inputs, *args, **kwargs):
    # NOTE: we only use the first input !
    inpt = inputs[0]

    rep = kwargs.get('rep', 2) # act_n
    #rep_axis = kwargs.get('axis', 3)
    rep_axis = kwargs.get('axis', 1)

    outputs = K.repeat_elements(inpt, rep, axis=rep_axis)
    return outputs


def kr_merge_fn_reduce_max(inputs, *args, **kwargs):
    # NOTE: we only use the first input !
    inpt = inputs[0]

    axis=kwargs.get('axis', 1)

    #outputs = K.max(inpt, axis=3, keepdims=False)
    #outputs = K.max(inpt, axis=1, keepdims=False)
    outputs = K.max(inpt, axis=axis, keepdims=False)
    return outputs


class RRModelUnreal(rr_model_base.RRModelBase):
    def __init__(self, cfg={}, *args, **kwargs):
        super(RRModelUnreal, self).__init__(cfg, *args, **kwargs)

        self.input_shape = self.cfg.get('input_shape', [1])  # (32, 4, 84, 84)
        self.pc_wh = self.cfg.get('pc_wh', self.input_shape[-1] / 4)  # 21 = 84 / 4
        self.pc_cw = self.cfg.get('pc_cw', self.pc_wh / 2 + (self.pc_wh % 2))  # 11
        self.actn = self.cfg.get('actn', 1)
        self.rp_n = self.cfg.get('rp_n', 3)  # 0 1 2
        self.batch_n = self.cfg.get('batch_n', 32)
        self.lr = self.cfg.get('lr', 1e-6)
        self.loss_name = self.cfg.get('loss_name', 'mse')
        self.pc_lambda = self.cfg.get('pc_lambda', 0.0001)

        self.model_init()

        self.graph_init()


    def model_init(self):

        act_n = self.actn
        rp_n = self.rp_n
        batch_n = self.batch_n
        pc_wh = self.pc_wh
        pc_cw = self.pc_cw


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


        #in_last_action_reward = Input(shape=(act_n + 1,))

        last_a_r_model = Sequential(name='last_a_r_model')
        last_a_r_model.add(Flatten(input_shape=(1, act_n + 1)))


        merged_model = Sequential(name='merged_model')
        merged_model.add(Merge([base_model, last_a_r_model],
                               mode='concat', concat_axis=-1))
        merged_model.add(Reshape((1, 256 + act_n + 1)))
        merged_model.add(LSTM(256, name='lstm1'))  # TODO:


        policy_network = Sequential(name='policy_network')
        policy_network.add(merged_model)
        policy_network.add(Dense(act_n, activation='softmax', name='p'))


        value_network = Sequential(name='value_network')
        value_network.add(merged_model)
        value_network.add(Dense(1, activation='linear', name='v'))


        rp_model = Sequential(name='rp_model')
        rp_model.add(conv_model)
        rp_model.add(Dense(128, activation='relu', name='fc_rp'))
        rp_model.add(Dense(rp_n, activation='softmax', name='rp'))


        #vr_model = Sequential(name='vr_model')
        #vr_model.add(merged_model)
        #vr_model.add(Dense(1, activation='linear', name='vr'))
        # NOTE: here we reuse value_network directly !!??
        vr_model = value_network


        pc_fc_model = Sequential(name='pc_fc_model')
        pc_fc_model.add(merged_model)
        pc_fc_model.add(Dense(pc_cw * pc_cw * 32, activation='relu', name='fc_pc'))
        pc_fc_model.add(Reshape((32, pc_cw, pc_cw)))  # (32, 11, 11)


        pc_deconv_v_model = Sequential(name='pc_deconv_v_model')
        pc_deconv_v_model.add(pc_fc_model)
        pc_deconv_v_model.add(Deconvolution2D(1, 4, 4,
            output_shape=(batch_n, 1, pc_wh, pc_wh),
            #input_shape=(32, pc_cw, pc_cw),
            subsample=(2, 2),
            init=rr_fn_conv_w_init,
            border_mode='same',
            activation='relu', name='deconv1'))


        pc_deconv_a_model = Sequential(name='pc_deconv_a_model')
        pc_deconv_a_model.add(pc_fc_model)
        pc_deconv_a_model.add(Deconvolution2D(act_n, 4, 4,
            output_shape=(batch_n, act_n, pc_wh, pc_wh),
            #input_shape=(32, pc_cw, pc_cw),
            subsample=(2, 2),
            init=rr_fn_conv_w_init,
            border_mode='same',
            activation='relu', name='deconv2'))


        pc_deconv_a_mean_model = Sequential(name='pc_deconv_a_mean_model')
        pc_deconv_a_mean_model.add(Merge([pc_deconv_a_model, pc_deconv_a_model],
                                         mode=kr_merge_fn_mean,
                                         arguments={'axis': 1},
                                         output_shape=(1, pc_wh, pc_wh),
                                         name='mg_pc_deconv_a_mean'))


        pc_deconv_a_mean_rep_model = Sequential(name='pc_deconv_a_mean_rep_model')
        pc_deconv_a_mean_rep_model.add(Merge([pc_deconv_a_mean_model,
                                              pc_deconv_a_mean_model],
                                             mode=kr_merge_fn_repeat,
                                             arguments={'axis': 1, 'rep': act_n},
                                             output_shape=(act_n, pc_wh, pc_wh),
                                             name='mg_pc_deconv_a_mean_rep'))


        pc_deconv_a_mean_neg_model = Sequential(name='pc_deconv_a_mean_neg_model')
        pc_deconv_a_mean_neg_model.add(Merge([pc_deconv_a_mean_rep_model,
                                              pc_deconv_a_mean_rep_model],
                                              mode=kr_merge_fn_negative,
                                              output_shape=(act_n, pc_wh, pc_wh),
                                              name='mg_pc_deconv_a_mean_neg'))


        pc_deconv_v_rep_model = Sequential(name='pc_deconv_v_rep_model')
        pc_deconv_v_rep_model.add(Merge([pc_deconv_v_model,
                                         pc_deconv_v_model],
                                        mode=kr_merge_fn_repeat,
                                        arguments={'axis': 1, 'rep': act_n},
                                        output_shape=(act_n, pc_wh, pc_wh),
                                        name='mg_pc_deconv_v_rep'))


        pc_q_model = Sequential(name='pc_q_model')
        pc_q_model.add(Merge([pc_deconv_v_rep_model,
                              #x#pc_deconv_v_model,
                              pc_deconv_a_model,
                              #x#pc_deconv_a_mean_neg_model],
                              pc_deconv_a_mean_rep_model],
                             mode='sum',
                             concat_axis=-1,
                             output_shape=(batch_n, act_n, pc_wh, pc_wh),
                             name='mg_pc_q'))


        pc_q_max_model = Sequential(name='pc_q_max_model')
        pc_q_max_model.add(Merge([pc_q_model, pc_q_model],
                                 mode=kr_merge_fn_reduce_max,
                                 arguments={'axis': 1},
                                 output_shape=(batch_n, pc_wh, pc_wh),
                                 name='mg_pc_q_max'))


        self.policy_network = policy_network
        self.value_network = value_network

        self.rp_model = rp_model
        self.vr_model = vr_model
        self.pc_q_model = pc_q_model
        self.pc_q_max_model = pc_q_max_model


        self.R_t = K.variable(np.zeros((self.batch_n,)), name='R_t')
        self.a_t = K.variable(np.zeros((self.batch_n, self.actn)), name='a_t')
        self.v_t = K.variable(np.zeros((self.batch_n,)), name='v_t')
        self.pc_a = K.variable(np.zeros((self.batch_n, self.actn)), name='pc_a')
        self.pc_r = K.variable(np.zeros((self.batch_n, pc_wh, pc_wh)), name='pc_r')


        pv_loss = self.a3c_pv_loss(self.policy_network, self.value_network, self.R_t, self.a_t, self.v_t)
        v_loss = self.a3c_v_loss(self.policy_network, self.value_network, self.R_t)
        rp_loss = self.unr_rp_loss(self.rp_model)
        #vr_loss = self.unr_vr_loss(self.vr_model)
        pc_q_loss = self.unr_pc_q_loss(self.pc_q_model, self.pc_a, self.pc_r)

        adam = Adam(lr=self.lr)

        self.policy_network.compile(loss=pv_loss, optimizer=adam)
        self.value_network.compile(loss=v_loss, optimizer=adam)
        self.rp_model.compile(loss=rp_loss, optimizer=adam)
        #self.vr_model.compile(loss=vr_loss, optimizer=adam)
        self.pc_q_model.compile(loss=pc_q_loss, optimizer=adam)
        #self.pc_q_max_model.compile(loss='mse', optimizer=adam)


        return self.policy_network, self.value_network, self.rp_model, \
            self.vr_model, self.pc_q_model, self.pc_q_max_model


    def a3c_pv_loss__1(self, p_network, v_network, R_t, a_t):  # not used !

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            log_prob = K.log(K.reduce_sum(K.mul(p_network, a_t), reduction_indices=1))
            p_loss = -log_prob * (R_t - v_network)
            v_loss = K.reduce_mean(K.square(R_t - v_network))  # TODO: could we use v_network ???

            total_loss = p_loss + (0.5 * v_loss)
            return total_loss

        return _inner_loss


    def a3c_pv_loss(self, p_network, v_network, R_t, a_t, v_t):

        def _inner_loss(y_true, y_pred, *args, **kwargs):
            """
            TODO: use like keras.objectives.binary_crossentropy()
            """

            #o#log_prob = K.log(K.sum(y_pred * a_t, axis=1))
            log_prob = K.log(K.sum(y_pred * a_t, axis=1) + K.epsilon())
            p_loss = -log_prob * (R_t - v_t)
            #x#p_loss = log_prob * (R_t - v_t)

            v_loss = K.mean(K.square(R_t - v_t))

            #total_loss = p_loss + (0.5 * v_loss)  # TODO: only use p_loss ???
            #return total_loss
            return p_loss

        return _inner_loss


    def a3c_v_loss(self, p_network, v_network, R_t):

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            #o#v_loss = K.mean(K.square(R_t - v_t))
            v_loss = K.mean(K.square(y_true - y_pred), axis=-1)  # mse
            v_loss = 0.5 * v_loss

            return v_loss

        return _inner_loss


    def unr_vr_loss(self, vr_model):

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            vr_loss = K.mean(K.square(y_true - y_pred), axis=-1)  # mse
            #vr_loss = 0.5 * vr_loss

            return vr_loss

        return _inner_loss


    def unr_rp_loss(self, rp_model):

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            #rp_loss = -tf.reduce_sum(self.rp_c_target * tf.log(self.rp_c))
            #rp_loss = K.mean(K.square(y_true - y_pred))
            rp_loss = -K.sum(y_true * K.log(y_pred + K.epsilon()))

            return rp_loss

        return _inner_loss


    def unr_pc_q_loss(self, pc_q_model, pc_a, pc_r):

        def _inner_loss(y_true, y_pred, *args, **kwargs):

            #pc_a_reshaped = K.reshape(pc_a, [self.batch_n, 1, 1, self.actn])
            pc_a_reshaped = K.reshape(pc_a, [self.batch_n, self.actn, 1, 1])

            pc_qa_ = y_pred * pc_a_reshaped
            #pc_qa = K.sum(pc_qa_, axis=3, keepdims=False)
            pc_qa = K.sum(pc_qa_, axis=1, keepdims=False)

            pc_q_loss = K.mean(K.square(pc_r - pc_qa))  # tf.nn.l2_loss

            pc_q_loss = self.pc_lambda * pc_q_loss

            return pc_q_loss

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


    def repeat_elements(self, *args, **kwargs):
        return K.repeat_elements(*args, **kwargs)


    def _get_model_saved_file_name(self, mfp, pt='x'):
        mf = mfp[:-4] + pt + mfp[-3:]
        return mf

    def load(self, model_file_p, model_file_v):

        self.policy_network.load_weights(model_file_p)
        self.value_network.load_weights(model_file_v)

        self.rp_model.load_weights(self._get_model_saved_file_name(model_file_p, pt='rp'))
        self.vr_model.load_weights(self._get_model_saved_file_name(model_file_p, pt='vr'))
        self.pc_q_model.load_weights(self._get_model_saved_file_name(model_file_p, pt='pc_q'))
        self.pc_q_max_model.load_weights(self._get_model_saved_file_name(model_file_p, pt='pc_q_max'))

        pv_loss = self.a3c_pv_loss(self.policy_network, self.value_network, self.R_t, self.a_t, self.v_t)
        v_loss = self.a3c_v_loss(self.policy_network, self.value_network, self.R_t)
        rp_loss = self.unr_rp_loss(self.rp_model)
        #vr_loss = self.unr_vr_loss(self.vr_model)
        pc_q_loss = self.unr_pc_q_loss(self.pc_q_model, self.pc_a, self.pc_r)

        adam = Adam(lr=self.lr)

        self.policy_network.compile(loss=pv_loss, optimizer=adam)
        self.value_network.compile(loss=v_loss, optimizer=adam)
        self.rp_model.compile(loss=rp_loss, optimizer=adam)
        #self.vr_model.compile(loss=vr_loss, optimizer=adam)
        self.pc_q_model.compile(loss=pc_q_loss, optimizer=adam)
        #self.pc_q_max_model.compile(loss='mse', optimizer=adam)


    def save(self, model_file_p, model_file_v):
        self.policy_network.save(model_file_p, overwrite=True)
        self.value_network.save(model_file_v, overwrite=True)

        self.rp_model.save(self._get_model_saved_file_name(model_file_p, pt='rp'), overwrite=True)
        self.vr_model.save(self._get_model_saved_file_name(model_file_p, pt='vr'), overwrite=True)
        self.pc_q_model.save(self._get_model_saved_file_name(model_file_p, pt='pc_q'), overwrite=True)
        self.pc_q_max_model.save(self._get_model_saved_file_name(model_file_p, pt='pc_q_max'), overwrite=True)


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

    def predict_pc_q_max(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.pc_q_max_model.predict(*args, **kwargs)
        else:
            r = self.pc_q_max_model.predict(*args, **kwargs)
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


    def train_on_batch_rp(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.rp_model.train_on_batch(*args, **kwargs)
        else:
            r = self.rp_model.train_on_batch(*args, **kwargs)
        return r

    def train_on_batch_vr(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.vr_model.train_on_batch(*args, **kwargs)
        else:
            r = self.vr_model.train_on_batch(*args, **kwargs)
        return r

    def train_on_batch_pc_q(self, *args, **kwargs):
        if self.global_graph:
            with self.global_graph.as_default():
                r = self.pc_q_model.train_on_batch(*args, **kwargs)
        else:
            r = self.pc_q_model.train_on_batch(*args, **kwargs)
        return r


    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


    def train_on_batch(self, *args, **kwargs):
        return self.model.train_on_batch(*args, **kwargs)
