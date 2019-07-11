# -*- coding: utf-8 -*-

import tensorflow as tf


class BatchNorm1D:
    """
    1D batch normalization layer.
    """

    def __init__(self, *args, **kwargs):
        super(BatchNorm1D, self).__init__()
        self.norm = tf.layers.BatchNormalization(*args, **kwargs)

    def __call__(self, x, is_training):
        with tf.variable_scope("batch_norm_1d"):
            y = tf.expand_dims(x, axis=1)
            y = self.norm(y, training=is_training)
            y = tf.squeeze(y, axis=1)
            return y


class ConvBlock:
    """
    Convolutional block for Centaur model.
    """

    def __init__(self,
                 name,
                 conv,
                 norm,
                 activation_fn,
                 dropout,
                 is_training,
                 is_residual,
                 is_causal):
        """
        Convolutional block constructor.

        Args:
          name: name of the block.
          conv: convolutional layer.
          norm: normalization layer to use after the convolutional layer.
          activation_fn: activation function to use after the normalization.
          dropout: dropout rate.
          is_training: whether it is training mode.
          is_residual: whether the block should contain a residual connection.
          is_causal: whether the convolutional layer should be causal.
        """

        self.name = name
        self.conv = conv
        self.norm = norm
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.is_training = is_training
        self.is_residual = is_residual
        self.is_casual = is_causal

    def __call__(self, x):
        with tf.variable_scope(self.name):
            if self.is_casual:
                # Add padding from the left side to avoid looking to the future
                pad_size = self.conv.kernel_size[0] - 1
                y = tf.pad(x, [[0, 0], [pad_size, 0], [0, 0]])
            else:
                y = x

            y = self.conv(y)

            if self.norm is not None:
                y = self.norm(y, is_training=self.is_training)

            if self.activation_fn is not None:
                y = self.activation_fn(y)

            if self.dropout is not None:
                y = self.dropout(y, training=self.is_training)

            return x + y if self.is_residual else y

    @staticmethod
    def create(index,
               conv_params,
               regularizer,
               bn_momentum,
               bn_epsilon,
               cnn_dropout_prob,
               is_training,
               is_residual=True,
               is_causal=False):
        activation_fn = conv_params.get("activation_fn", tf.nn.relu)

        conv = tf.layers.Conv1D(
            name="conv_%d" % index,
            filters=conv_params["num_channels"],
            kernel_size=conv_params["kernel_size"],
            strides=conv_params["stride"],
            padding=conv_params["padding"],
            kernel_regularizer=regularizer
        )

        norm = BatchNorm1D(
            name="bn_%d" % index,
            gamma_regularizer=regularizer,
            momentum=bn_momentum,
            epsilon=bn_epsilon
        )

        dropout = tf.layers.Dropout(
            name="dropout_%d" % index,
            rate=cnn_dropout_prob
        )

        if "is_causal" in conv_params:
            is_causal = conv_params["is_causal"]

        if "is_residual" in conv_params:
            is_residual = conv_params["is_residual"]

        return ConvBlock(
            name="layer_%d" % index,
            conv=conv,
            norm=norm,
            activation_fn=activation_fn,
            dropout=dropout,
            is_training=is_training,
            is_residual=is_residual,
            is_causal=is_causal
        )


class Prenet:
    """
    Centaur decoder pre-net.
    """

    def __init__(self,
                 n_layers,
                 hidden_size,
                 activation_fn,
                 dropout=0.5,
                 regularizer=None,
                 is_training=True,
                 dtype=None,
                 name="prenet"):
        """
        Pre-net constructor.

        Args:
          n_layers: number of fully-connected layers to use.
          hidden_size: number of units in each pre-net layer.
          activation_fn: activation function to use.
          dropout: dropout rate. Defaults to 0.5.
          regularizer: regularizer for the convolution kernel.
            Defaults to None.
          is_training: whether it is training mode. Defaults to None.
          dtype: dtype of the layer's weights. Defaults to None.
          name: name of the block.
        """

        self.name = name
        self.layers = []
        self.dropout = dropout
        self.is_training = is_training

        for i in range(n_layers):
            layer = tf.layers.Dense(
                name="layer_%d" % i,
                units=hidden_size,
                use_bias=True,
                activation=activation_fn,
                kernel_regularizer=regularizer,
                dtype=dtype
            )
            self.layers.append(layer)

    def __call__(self, x):
        with tf.variable_scope(self.name):
            for layer in self.layers:
                x = tf.layers.dropout(
                    layer(x),
                    rate=self.dropout,
                    training=self.is_training
                )

            return x


class Transformer_BatchNorm(tf.layers.Layer):
    """Transformer batch norn: supports [BTC](default) and [BCT] formats. """

    def __init__(self, is_training: bool = False, regularizer=None, data_format: str = 'channels_last',
                 momentum: float = 0.95
                 , epsilon: float = 0.0001, center_scale: bool = True, regularizer_params=None
                 ):
        super(Transformer_BatchNorm, self).__init__()
        if regularizer_params is None:
            regularizer_params = {'scale': 0.0}
        self.is_training = is_training
        self.data_format = data_format
        self.momentum = momentum
        self.epsilon = epsilon
        self.center_scale = center_scale
        self.regularizer = regularizer if self.center_scale else None
        if self.regularizer is not None:
            self.regularizer_params = regularizer_params
            self.regularizer = self.regularizer(self.regularizer_params['scale']) \
                if self.regularizer_params['scale'] > 0.0 else None
        # print("Batch norm, training=", training, params)

    def call(self, x, *args, **kwargs):
        x = tf.expand_dims(x, axis=2)
        axis = -1 if (self.data_format == 'channels_last') else 1
        y = tf.layers.batch_normalization(inputs=x, axis=axis,
                                          momentum=self.momentum, epsilon=self.epsilon,
                                          center=self.center_scale, scale=self.center_scale,
                                          beta_regularizer=self.regularizer, gamma_regularizer=self.regularizer,
                                          training=self.is_training)
        y = tf.squeeze(y, axis=[2])
        return y


class LayerNormalization(tf.layers.Layer):
    """Layer normalization for BTC format: supports L2(default) and L1 modes"""

    def __init__(self, hidden_size, norm_type: str = "layernorm_L2", epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.norm_type = norm_type
        self.epsilon = epsilon

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.keras.initializers.Ones(),
                                     dtype=tf.float32)
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.keras.initializers.Zeros(),
                                    dtype=tf.float32)
        self.built = True

    def call(self, x, *args, **kwargs):
        if self.norm_type == "layernorm_L2":
            epsilon = self.epsilon
            dtype = x.dtype
            x = tf.cast(x=x, dtype=tf.float32)
            mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
            norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
            result = norm_x * self.scale + self.bias
            return tf.cast(x=result, dtype=dtype)

        else:
            dtype = x.dtype
            if dtype == tf.float16:
                x = tf.cast(x, dtype=tf.float32)
            mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
            x = x - mean
            variance = tf.reduce_mean(tf.abs(x), axis=[-1], keepdims=True)
            norm_x = tf.div(x, variance + self.epsilon)
            y = norm_x * self.scale + self.bias
            if dtype == tf.float16:
                y = tf.saturate_cast(y, dtype)
            return y


class PrePostProcessingWrapper(object):
    """Wrapper around layer, that applies pre-processing and post-processing."""

    def __init__(self, layer, hidden_size, layer_postprocess_dropout,
                 is_training=False, norm_type="layernorm_L2"):
        self.layer = layer
        self.postprocess_dropout = layer_postprocess_dropout
        self.is_training = is_training
        # Create normalization layer
        if norm_type == "batch_norm":
            self.norm = Transformer_BatchNorm(is_training=is_training)
        else:
            self.norm = LayerNormalization(hidden_size=hidden_size, norm_type=norm_type)

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: normalization
        y = self.norm(x)
        y = self.layer(y, *args, **kwargs)
        # Postprocessing: dropout and residual connection
        if self.is_training:
            y = tf.nn.dropout(y, keep_prob=1 - self.postprocess_dropout)
        return x + y
