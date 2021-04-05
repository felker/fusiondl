import src.global_vars as g
import tensorflow as tf
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

    # if num_1D > 0:

    # else:
    #     pre_rnn = pre_rnn_input


    pre_rnn_model = tf.keras.Model(inputs=pre_rnn_input, outputs=pre_rnn)
    x_input = Input(batch_shape=batch_input_shape)
    # TODO(KGF): Ge moved this inside a new conditional in Dec 2019. check
    # x_in = TimeDistributed(pre_rnn_model)(x_input)
    if num_1D > 0:
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
        rnn_kwargs = dict(return_sequences=return_sequences,
                            # batch_input_shape=batch_input_shape,
                            stateful=stateful,
                            kernel_regularizer=l2(regularization),
                            recurrent_regularizer=l2(regularization),
                            bias_regularizer=l2(regularization),
                            )
        # https://stackoverflow.com/questions/60468385/is-there-cudnnlstm-or-cudnngru-alternative-in-tensorflow-2-0
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/keras/layers/recurrent_v2.py#L902
        if rnn_type != 'CuDNNLSTM':
            # Recurrent Dropout is unsupported in CuDNN library
            rnn_kwargs['dropout'] = dropout_prob
            rnn_kwargs['recurrent_dropout'] = dropout_prob
        for _ in range(model_conf['rnn_layers']):
            x_in = rnn_model(rnn_size, **rnn_kwargs)(x_in)
            x_in = Dropout(dropout_prob)(x_in)
        if return_sequences:
            x_out = TimeDistributed(
                Dense(1, activation=output_activation))(x_in)
    model = tf.keras.Model(inputs=x_input, outputs=x_out)
    model.reset_states()
    return model
