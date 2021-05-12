import tensorflow as tf
from tensorflow.keras.regularizers import l2

class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self,
                 num_filters,
                 kernel_size,
                 pool_size,
                 batch_norm=False,
                 activation='relu',
                 ):

        tf.keras.layers.Layer.__init__(self)

        self.use_batch_norm = batch_norm
        self.activation = activation
        '''The first conv layer learns `num_conv_filters//div_fac`
        filters (aka kernels), each of size
        `(size_conv_filters, num1D)`. Its output will have shape
        (None, len(indices_1d)//num_1D - size_conv_filters + 1,
        num_conv_filters//div_fac), i.e., for
        each position in the input spatial series (direction along

        '''

        '''For i=1 first conv layer would get:
        (None, (len(indices_1d)//num_1D - size_conv_filters
        + 1)/pool_size-size_conv_filters + 1,
        num_conv_filters//div_fac)

        '''
        self.conv1 = tf.keras.layers.Conv1D(num_filters, kernel_size, padding='valid')
        if self.use_batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization()
            pre_rnn_1D = tf.keras.layers.Activation(self.activation)

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
        self.conv2 = tf.keras.layers.Conv1D(num_filters, 1, padding='valid')
        if self.use_batch_norm:
            self.bn2 = tf.keras.layers.BatchNormalization()
            pre_rnn_1D = tf.keras.layers.Activation(self.activation)
        '''Outputs (None, (len(indices_1d)//num_1D - size_conv_filters
        + 1)/pool_size, num_conv_filters//div_fac)

        For i=1, the pooling layer would output:
        (None,((len(indices_1d)//num_1D- size_conv_filters
        + 1)/pool_size-size_conv_filters+1)/pool_size,
        num_conv_filters//div_fac)

        '''
        self.max_pool = tf.keras.layers.MaxPooling1D(pool_size)

    def call(self, x_in):
        x = self.conv1(x_in)
        if self.use_batch_norm:
            x = self.bn1(x)
            x = tf.keras.layers.Activation(self.activation)(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
            x = tf.keras.layers.Activation(self.activation)(x)

        return self.max_pool(x)


class InputBlock(tf.keras.layers.Layer):

    def __init__(self,
                 num_conv_filters,
                 size_conv_filters,
                 indices_1d,
                 indices_0d,
                 model_conf
                 ):

        tf.keras.layers.Layer.__init__(self)

        self.use_batch_norm = False
        if 'use_batch_norm' in model_conf:
            self.use_batch_norm = model_conf['use_batch_norm']

        self.num_conv_layers = model_conf['num_conv_layers']

        self.lambda_1D = tf.keras.layers.Lambda(lambda x: x[:, len(indices_0d):],
                                            output_shape=(len(indices_1d),))
        self.lambda_0D = tf.keras.layers.Lambda(lambda x: x[:, :len(indices_0d)],
                                            output_shape=(len(indices_0d),))
        self.conv_blocks = []

        for i in range(self.num_conv_layers):
            div_fac = 2**i
            self.conv_blocks.append(
                ConvolutionBlock(num_conv_filters//div_fac, size_conv_filters,
                                 pool_size,
                                 batch_norm=self.use_batch_norm,
                                 activation='relu',)
                                 )

        self.dense_1 = tf.keras.layers.Dense(
            dense_size,
            kernel_regularizer=l2(dense_regularization),
            bias_regularizer=l2(dense_regularization),
            activity_regularizer=l2(dense_regularization))
        if self.use_batch_norm:
            self.bn1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(
            dense_size//4,
            kernel_regularizer=l2(dense_regularization),
            bias_regularizer=l2(dense_regularization),
            activity_regularizer=l2(dense_regularization))
        if self.use_batch_norm:
            self.bn2 = tf.keras.layers.BatchNormalization()


    def call(self, x_in):
        pre_rnn_0D = self.lambda_0D(x_in)

        pre_rnn_1D = self.lambda_1D(x_in)
        pre_rnn_1D = Reshape((num_1D, len(indices_1d)//num_1D))(pre_rnn_1D)
        pre_rnn_1D = Permute((2, 1))(pre_rnn_1D)

        for i in range(self.num_conv_layers):
            pre_rnn_1D = self.conv_blocks(pre_rnn_1d)

        pre_rnn_1D = tf.keras.layers.Flatten()(pre_rnn_1D)
        pre_rnn_1D = self.dense_1(pre_rnn_1D)
        if self.use_batch_norm:
            pre_rnn_1D = self.bn1(pre_rnn_1D)
        pre_rnn_1D = tf.keras.layers.Activation('relu')(pre_rnn_1D)
        pre_rnn_1D = self.dense2(pre_rnn_1D)
        if self.use_batch_norm:
            pre_rnn_1D = self.bn2(pre_rnn_1D)
        pre_rnn_1D = tf.keras.layers.Activation('relu')(pre_rnn_1D)

        pre_rnn = Concatenate()([pre_rnn_0D, pre_rnn_1D])
        return pre_rnn
