import src.global_vars as g
import tensorflow as tf
# KGF: see below synchronization--- output is launched here
#
# KGF: (was used only in hyper_build_model())
from tensorflow.keras.layers import (
    Input,
    Dense, Activation, Dropout, Lambda,
    Reshape, Flatten, Permute,  # RepeatVector
    LSTM, SimpleRNN, BatchNormalization,
    Convolution1D, MaxPooling1D, TimeDistributed,
    Concatenate
    )
CuDNNLSTM = LSTM
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.regularizers import l2  # l1, l1_l2
import re
import os
import sys
import numpy as np
from copy import deepcopy
from src.utils.hashing import general_object_hash
from src.models.tensorflow.tcn import TCN
# TODO(KGF): consider using importlib.util.find_spec() instead (Py>3.4)
try:
    import keras2onnx
    import onnx
except ImportError:  # as e:
    _has_onnx = False
    # onnx = None
    # keras2onnx = None
else:
    _has_onnx = True

# Synchronize 2x stderr msg from TensorFlow initialization via Keras backend
# "Succesfully opened dynamic library... libcudart" "Using TensorFlow backend."
if g.comm is not None:
    g.flush_all_inorder()


def build_model(self, predict, custom_batch_size=None):
    conf = self.conf
    model_conf = conf['model']
    rnn_size = model_conf['rnn_size']
    rnn_type = model_conf['rnn_type']
    regularization = model_conf['regularization']
    dense_regularization = model_conf['dense_regularization']
    use_batch_norm = False
    if 'use_batch_norm' in model_conf:
        use_batch_norm = model_conf['use_batch_norm']

    dropout_prob = model_conf['dropout_prob']
    length = model_conf['length']
    pred_length = model_conf['pred_length']
    # skip = model_conf['skip']
    stateful = model_conf['stateful']
    return_sequences = model_conf['return_sequences']
    # model_conf['output_activation']
    output_activation = conf['data']['target'].activation
    use_signals = conf['paths']['use_signals']
    num_signals = sum([sig.num_channels for sig in use_signals])
    num_conv_filters = model_conf['num_conv_filters']
    # num_conv_layers = model_conf['num_conv_layers']
    size_conv_filters = model_conf['size_conv_filters']
    pool_size = model_conf['pool_size']
    dense_size = model_conf['dense_size']

    batch_size = self.conf['training']['batch_size']
    if predict:
        batch_size = self.conf['model']['pred_batch_size']
        # so we can predict with one time point at a time!
        if return_sequences:
            length = pred_length
        else:
            length = 1

    if custom_batch_size is not None:
        batch_size = custom_batch_size

    if rnn_type == 'LSTM':
        rnn_model = LSTM
    elif rnn_type == 'CuDNNLSTM':
        rnn_model = CuDNNLSTM
    elif rnn_type == 'SimpleRNN':
        rnn_model = SimpleRNN
    else:
        print('Unkown Model Type, exiting.')
        exit(1)

    batch_input_shape = (batch_size, length, num_signals)

    indices_0d, indices_1d, num_0D, num_1D = self.get_0D_1D_indices()

    # def slicer(x, indices):
    #     return x[:, indices]

    # def slicer_output_shape(input_shape, indices):
    #     shape_curr = list(input_shape)
    #     assert len(shape_curr) == 2  # only valid for 3D tensors
    #     shape_curr[-1] = len(indices)
    #     return tuple(shape_curr)

    pre_rnn_input = Input(shape=(num_signals,))

    if num_1D > 0:
        pre_rnn_1D = Lambda(lambda x: x[:, len(indices_0d):],
                            output_shape=(len(indices_1d),))(pre_rnn_input)
        pre_rnn_0D = Lambda(lambda x: x[:, :len(indices_0d)],
                            output_shape=(len(indices_0d),))(pre_rnn_input)
        # slicer(x,indices_0d),lambda s:
        # slicer_output_shape(s,indices_0d))(pre_rnn_input)
        pre_rnn_1D = Reshape((num_1D, len(indices_1d)//num_1D))(pre_rnn_1D)
        pre_rnn_1D = Permute((2, 1))(pre_rnn_1D)
        if ('simple_conv' in model_conf.keys()
                and model_conf['simple_conv'] is True):
            for i in range(model_conf['num_conv_layers']):
                pre_rnn_1D = Convolution1D(
                    num_conv_filters, size_conv_filters,
                    padding='valid', activation='relu')(pre_rnn_1D)
            pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
        else:
            for i in range(model_conf['num_conv_layers']):
                div_fac = 2**i
                '''The first conv layer learns `num_conv_filters//div_fac`
                filters (aka kernels), each of size
                `(size_conv_filters, num1D)`. Its output will have shape
                (None, len(indices_1d)//num_1D - size_conv_filters + 1,
                num_conv_filters//div_fac), i.e., for
                each position in the input spatial series (direction along
                radius), the activation of each filter at that position.

                '''

                '''For i=1 first conv layer would get:
                (None, (len(indices_1d)//num_1D - size_conv_filters
                + 1)/pool_size-size_conv_filters + 1,
                num_conv_filters//div_fac)

                '''
                pre_rnn_1D = Convolution1D(
                    num_conv_filters//div_fac, size_conv_filters,
                    padding='valid')(pre_rnn_1D)
                if use_batch_norm:
                    pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                    pre_rnn_1D = Activation('relu')(pre_rnn_1D)

                '''The output of the second conv layer will have shape
                (None, len(indices_1d)//num_1D - size_conv_filters + 1,
                num_conv_filters//div_fac),
                i.e., for each position in the input spatial series
                (direction along radius), the activation of each filter
                at that position.

                For i=1, the second layer would output
                (None, (len(indices_1d)//num_1D - size_conv_filters + 1)/
                pool_size-size_conv_filters + 1,num_conv_filters//div_fac)
                '''
                pre_rnn_1D = Convolution1D(
                    num_conv_filters//div_fac, 1, padding='valid')(
                        pre_rnn_1D)
                if use_batch_norm:
                    pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
                    pre_rnn_1D = Activation('relu')(pre_rnn_1D)
                '''Outputs (None, (len(indices_1d)//num_1D - size_conv_filters
                + 1)/pool_size, num_conv_filters//div_fac)

                For i=1, the pooling layer would output:
                (None,((len(indices_1d)//num_1D- size_conv_filters
                + 1)/pool_size-size_conv_filters+1)/pool_size,
                num_conv_filters//div_fac)

                '''
                pre_rnn_1D = MaxPooling1D(pool_size)(pre_rnn_1D)
        pre_rnn_1D = Flatten()(pre_rnn_1D)
        pre_rnn_1D = Dense(
            dense_size,
            kernel_regularizer=l2(dense_regularization),
            bias_regularizer=l2(dense_regularization),
            activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
        if use_batch_norm:
            pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
        pre_rnn_1D = Activation('relu')(pre_rnn_1D)
        pre_rnn_1D = Dense(
            dense_size//4,
            kernel_regularizer=l2(dense_regularization),
            bias_regularizer=l2(dense_regularization),
            activity_regularizer=l2(dense_regularization))(pre_rnn_1D)
        if use_batch_norm:
            pre_rnn_1D = BatchNormalization()(pre_rnn_1D)
        pre_rnn_1D = Activation('relu')(pre_rnn_1D)
        pre_rnn = Concatenate()([pre_rnn_0D, pre_rnn_1D])
    else:
        pre_rnn = pre_rnn_input

    if model_conf['rnn_layers'] == 0 or (
            'extra_dense_input' in model_conf.keys()
            and model_conf['extra_dense_input']):
        pre_rnn = Dense(
            dense_size,
            activation='relu',
            kernel_regularizer=l2(dense_regularization),
            bias_regularizer=l2(dense_regularization),
            activity_regularizer=l2(dense_regularization))(pre_rnn)
        pre_rnn = Dense(
            dense_size//2,
            activation='relu',
            kernel_regularizer=l2(dense_regularization),
            bias_regularizer=l2(dense_regularization),
            activity_regularizer=l2(dense_regularization))(pre_rnn)
        pre_rnn = Dense(
            dense_size//4,
            activation='relu',
            kernel_regularizer=l2(dense_regularization),
            bias_regularizer=l2(dense_regularization),
            activity_regularizer=l2(dense_regularization))(pre_rnn)

    pre_rnn_model = tf.keras.Model(inputs=pre_rnn_input, outputs=pre_rnn)
    x_input = Input(batch_shape=batch_input_shape)
    # TODO(KGF): Ge moved this inside a new conditional in Dec 2019. check
    # x_in = TimeDistributed(pre_rnn_model)(x_input)
    if (num_1D > 0 or (
            'extra_dense_input' in model_conf.keys()
            and model_conf['extra_dense_input'])):
        x_in = TimeDistributed(pre_rnn_model)(x_input)
    else:
        x_in = x_input

    # ==========
    # TCN MODEL
    # ==========
    if ('keras_tcn' in model_conf.keys()
            and model_conf['keras_tcn'] is True):
        print('Building TCN model....')
        tcn_layers = model_conf['tcn_layers']
        tcn_dropout = model_conf['tcn_dropout']
        nb_filters = model_conf['tcn_hidden']
        kernel_size = model_conf['kernel_size_temporal']
        nb_stacks = model_conf['tcn_nbstacks']
        use_skip_connections = model_conf['tcn_skip_connect']
        activation = model_conf['tcn_activation']
        use_batch_norm = model_conf['tcn_batch_norm']
        for _ in range(model_conf['tcn_pack_layers']):
            x_in = TCN(
                use_batch_norm=use_batch_norm, activation=activation,
                use_skip_connections=use_skip_connections,
                nb_stacks=nb_stacks, kernel_size=kernel_size,
                nb_filters=nb_filters, num_layers=tcn_layers,
                dropout_rate=tcn_dropout)(x_in)
            x_in = Dropout(dropout_prob)(x_in)
    else:  # end TCN model
        # ==========
        # RNN MODEL
        # ==========
        # LSTM in ONNX: "The maximum opset needed by this model is only 9."
        model_kwargs = dict(return_sequences=return_sequences,
                            # batch_input_shape=batch_input_shape,
                            stateful=stateful,
                            kernel_regularizer=l2(regularization),
                            recurrent_regularizer=l2(regularization),
                            bias_regularizer=l2(regularization),
                            )
        if rnn_type != 'CuDNNLSTM':
            # Dropout is unsupported in CuDNN library
            model_kwargs['dropout'] = dropout_prob
            model_kwargs['recurrent_dropout'] = dropout_prob
        for _ in range(model_conf['rnn_layers']):
            x_in = rnn_model(rnn_size, **model_kwargs)(x_in)
            x_in = Dropout(dropout_prob)(x_in)
        if return_sequences:
            # x_out = TimeDistributed(Dense(100,activation='tanh')) (x_in)
            x_out = TimeDistributed(
                Dense(1, activation=output_activation))(x_in)
    model = tf.keras.Model(inputs=x_input, outputs=x_out)
    model.reset_states()
    return model
