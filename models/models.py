# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import math_ops
from .modules import Prenet, LayerNormalization, ConvBlock
from .attention import AttentionBlock
import math

_NEG_INF = -1e9


def get_padding(x, padding_value=0, dtype=tf.float32):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that
      dtype: type of the output

    Returns:
      float tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    # print("get_padding", dtype)
    with tf.name_scope("padding"):
        return tf.cast(tf.equal(x, padding_value), dtype=dtype)


def get_padding_bias(x, res_rank=4, pad_sym=0):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]
      res_rank: int indicates the rank of attention_bias.
      dtype: type of the output attention_bias
      pad_sym: int the symbol used for padding

    Returns:
      Attention bias tensor of shape
      [batch_size, 1, 1, length] if  res_rank = 4 - for Transformer
      or [batch_size, 1, length] if res_rank = 3 - for ConvS2S
    """
    # print("get_padding_bias", dtype)
    with tf.name_scope("attention_bias"):
        padding = get_padding(x, padding_value=pad_sym, dtype=tf.float32)
        # padding = get_padding(x, padding_value=pad_sym, dtype=dtype)
        neg_inf = _NEG_INF  # if dtype==tf.float32 else _NEG_INF_FP16
        attention_bias = padding * neg_inf
        if res_rank == 4:
            attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, axis=1), axis=1)
        elif res_rank == 3:
            attention_bias = tf.expand_dims(attention_bias, axis=1)
        else:
            raise ValueError("res_rank should be 3 or 4 but got {}".format(res_rank))
    return attention_bias


def get_position_encoding(
        length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = tf.cast(tf.range(length), dtype=tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.cast((num_timescales) - 1, dtype=tf.float32)))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), dtype=tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size, init_var=None,
                 embed_scale=True, pad_sym=0, mask_paddings=True, regularizer=None):
        super(EmbeddingSharedWeights, self).__init__()
        self.hidden_size = hidden_size
        self.embed_scale = embed_scale
        self.pad_sym = pad_sym
        self.mask_paddings = mask_paddings
        self.regularizer = regularizer
        self.vocab_size = vocab_size

        if init_var is None:
            self.init_var = hidden_size ** -0.5
        else:
            self.init_var = init_var

    def build(self, _):
        with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
            # Create and initialize weights. The random normal initializer was chosen
            # randomly, and works well.
            self.shared_weights = tf.get_variable("weights", [self.vocab_size, self.hidden_size],
                                                  initializer=tf.random_normal_initializer(0., self.init_var),
                                                  regularizer=self.regularizer)

        self.built = True

    def call(self, x, *args, **kwargs):
        """Get token embeddings of x.

        Args:
          x: An int64 tensor with shape [batch_size, length]
        Returns:
          embeddings: float32 tensor with shape [batch_size, length, embedding_size]
          padding: float32 tensor with shape [batch_size, length] indicating the
            locations of the padding tokens in x.
        """
        with tf.name_scope("embedding"):
            # fills out of bound values with padding symbol
            out_bound_mask = tf.cast(x > (self.vocab_size - 1), dtype=tf.int32)
            x *= 1 - out_bound_mask
            x += out_bound_mask * tf.cast(self.pad_sym, dtype=tf.int32)

            embeddings = tf.gather(self.shared_weights, x)
            if self.embed_scale:
                # Scale embedding by the sqrt of the hidden size
                embeddings *= self.hidden_size ** 0.5

            if self.mask_paddings:
                # Create binary array of size [batch_size, length]
                # where 1 = padding, 0 = not padding
                padding = get_padding(x, padding_value=self.pad_sym)

                # Set all padding embedding values to 0
                # embeddings *= tf.expand_dims(1 - padding, -1)
                embeddings *= tf.cast(tf.expand_dims(1.0 - padding, -1), dtype=embeddings.dtype)
            return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.

        Args:
          x: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])


class CentaurEncoder:
    """
    Centaur encoder that consists of convolutional layers.
    """

    def __init__(self, src_vocab_size=256, embedding_size=256, output_size=256, conv_layers_num=4,
                 cnn_dropout_prob=0.1,
                 bn_momentum=0.95,
                 bn_epsilon=-1e8, kernel_size=3, regularizer=None, name="centaur_encoder", is_training=False):
        """
        Centaur encoder constructor.
        See parent class for arguments description.
        Config parameters:
        * **src_vocab_size** (int) --- number of symbols in alphabet.
        * **embedding_size** (int) --- dimensionality of character embedding.
        * **output_size** (int) --- dimensionality of output embedding.
        * **conv_layers_num** (list) --- number of convolutional
        * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.95.
        * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-8.
        * **cnn_dropout_prob** (float) --- dropout probabilty for cnn layers.
          Defaults to 0.5.
        """
        self.name = name
        self.is_training = is_training
        self.layers = []
        self.regularizer = regularizer
        self.embedding_size = embedding_size
        self.src_vocab_size = src_vocab_size
        self.cnn_dropout_prob = cnn_dropout_prob
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.output_size = output_size
        self.conv_layers_num = conv_layers_num
        self.kernel_size = kernel_size

    def __call__(self, inputs):
        if not self.layers:
            embedding = EmbeddingSharedWeights(
                vocab_size=self.src_vocab_size,
                hidden_size=self.embedding_size,
                regularizer=self.regularizer
            )
            self.layers.append(embedding)
            conv_params = {
                "kernel_size": [self.kernel_size],
                "stride": [1],
                "num_channels": self.output_size,
                "padding": "SAME",
                "activation_fn": tf.nn.relu
            }
            for index in range(self.conv_layers_num):
                layer = ConvBlock.create(
                    index=index,
                    conv_params=conv_params,
                    regularizer=self.regularizer,
                    bn_momentum=self.bn_momentum,
                    bn_epsilon=self.bn_epsilon,
                    cnn_dropout_prob=self.cnn_dropout_prob,
                    is_training=self.is_training
                )
                self.layers.append(layer)

            linear_projection = tf.layers.Dense(
                name="linear_projection",
                units=self.output_size,
                use_bias=False,
                kernel_regularizer=self.regularizer
            )
            self.layers.append(linear_projection)

        # Apply all layers
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        inputs_attention_bias = get_padding_bias(inputs)

        return outputs, inputs_attention_bias


class CentaurDecoder:
    """
    Centaur decoder that consists of attention blocks
    followed by convolutional layers.
    """

    def __init__(self, num_mels=80, num_freq=513, prenet_hidden_size=512, decoder_hidden_size=512,
                 attention_dropout=0.1,
                 layer_postprocess_dropout=0.1, prenet_activation_fn=None, conv_layers_num=4,
                 mag_conv_layers_num=4, prenet_layers=2,
                 prenet_dropout=0.5,
                 prenet_use_inference_dropout=False,
                 cnn_dropout_prob=0.1,
                 bn_momentum=0.95,
                 bn_epsilon=-1e8,
                 reduction_factor=2,
                 attention_layers=4,
                 self_attention_conv_params=None,
                 attention_heads=1,
                 attention_cnn_dropout_prob=0.5,
                 window_size=4,
                 back_step_size=0, kernel_size=5, regularizer=None,
                 force_layers=None, dtype=tf.float32, name="centaur_decoder", is_prediction=False, is_training=False,
                 is_validation=False):

        """
        Centaur decoder constructor.
        See parent class for arguments description.

        Config parameters:

        * **prenet_layers** (int) --- number of fully-connected layers to use.
        * **prenet_hidden_size** (int) --- number of units in each pre-net layer.
        * **hidden_size** (int) --- dimensionality of hidden embeddings.
        * **conv_layers_num** (int) --- number of convolutional
        * **mag_conv_layers_num** (int) --- number of convolutional
          layers to reconstruct magnitude.
        * **attention_dropout** (float) --- dropout rate for attention layers.
        * **layer_postprocess_dropout** (float) --- dropout rate for
          transformer block sublayers.
        * **prenet_activation_fn** (callable) --- activation function to use for the
          prenet lyaers. Defaults to relu.
        * **prenet_dropout** (float) --- dropout rate for the pre-net. Defaults to 0.5.
        * **prenet_use_inference_dropout** (bool) --- whether to use dropout during the inference.
          Defaults to False.
        * **cnn_dropout_prob** (float) --- dropout probabilty for cnn layers.
          Defaults to 0.5.
        * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.95.
        * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-8.
        * **reduction_factor** (int) --- number of frames to predict in a time.
          Defaults to 1.
        * **attention_layers** (int) --- number of attention blocks. Defaults to 4.
        * **self_attention_conv_params** (dict) --- description of convolutional
          layer inside attention blocks. Defaults to None.
        * **attention_heads** (int) --- number of attention heads. Defaults to 1.
        * **attention_cnn_dropout_prob** (float) --- dropout rate for convolutional
          layers inside attention blocks. Defaults to 0.5.
        * **window_size** (int) --- size of attention window for forcing
          monotonic attention during the inference. Defaults to None.
        * **back_step_size** (int) --- number of steps attention is allowed to
          go back during the inference. Defaults to 0.
        * **force_layers** (list) --- indices of layers where forcing of
          monotonic attention should be enabled. Defaults to all layers.
        """
        self.kernel_size = kernel_size

        if force_layers is None:
            force_layers = [1, 3]
        self.is_validation = is_validation
        self.is_prediction = is_prediction
        self.name = name
        self.is_training = is_training
        self.prenet = None
        self.linear_projection = None
        self.attentions = []
        self.output_normalization = None
        self.conv_layers = []
        self.mag_conv_layers = []
        self.conv_layers_num = conv_layers_num
        self.mag_conv_layers_num = mag_conv_layers_num
        self.stop_token_projection_layer = None
        self.mel_projection_layer = None
        self.mag_projection_layer = None
        self.regularizer = regularizer
        self.num_mels = num_mels
        self.num_freq = num_freq
        self.reduction_factor = reduction_factor
        self.prenet_layers = prenet_layers
        self.prenet_hidden_size = prenet_hidden_size
        self.prenet_activation_fn = prenet_activation_fn if prenet_activation_fn else tf.nn.relu
        self.prenet_use_inference_dropout = prenet_use_inference_dropout
        self.prenet_dropout = prenet_dropout
        self.cnn_dropout_prob = cnn_dropout_prob
        self.dtype = dtype
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_layers = attention_layers
        self.force_layers = force_layers

        self.window_size = window_size
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.layer_postprocess_dropout = layer_postprocess_dropout
        self.attention_cnn_dropout_prob = attention_cnn_dropout_prob
        self.back_step_size = back_step_size
        if self_attention_conv_params is None:
            self_attention_conv_params = {
                "kernel_size": [self.kernel_size],
                "stride": [1],
                "num_channels": self.decoder_hidden_size,
                "padding": "VALID",
                "is_causal": True,
                "activation_fn": tf.nn.relu
            }
        self.self_attention_conv_params = self_attention_conv_params

    def __call__(self, targets=None, targets_length=None, encoder_outputs=None, attention_bias=None,
                 batch_size_per_gpu=32,
                 duration_max=1000):

        #  build_layers
        self.prenet = Prenet(
            n_layers=self.prenet_layers,
            hidden_size=self.prenet_hidden_size,
            activation_fn=self.prenet_activation_fn,
            dropout=self.prenet_dropout,
            regularizer=self.regularizer,
            is_training=self.is_training or self.prenet_use_inference_dropout,
            dtype=self.dtype
        )

        self.linear_projection = tf.layers.Dense(
            name="linear_projection",
            units=self.decoder_hidden_size,
            use_bias=False,
            kernel_regularizer=self.regularizer,
            dtype=self.dtype
        )
        conv_params = self.self_attention_conv_params
        force_layers = self.force_layers if self.force_layers else range(self.attention_layers)

        for index in range(self.attention_layers):
            window_size = None

            if index in force_layers:
                window_size = self.window_size

            attention = AttentionBlock(
                name="attention_block_%d" % index,
                hidden_size=self.decoder_hidden_size,
                attention_dropout=self.attention_dropout,
                layer_postprocess_dropout=self.layer_postprocess_dropout,
                regularizer=self.regularizer,
                is_training=self.is_training,
                cnn_dropout_prob=self.attention_cnn_dropout_prob,
                conv_params=conv_params,
                n_heads=self.attention_heads,
                window_size=window_size,
                back_step_size=self.back_step_size
            )
            self.attentions.append(attention)

        self.output_normalization = LayerNormalization(self.decoder_hidden_size)
        conv_layers_params = {
            "kernel_size": [self.kernel_size],
            "stride": [1],
            "num_channels": self.decoder_hidden_size,
            "padding": "VALID",
            "is_causal": True,
            "activation_fn": tf.nn.relu
        }
        for index in range(self.conv_layers_num):
            if conv_layers_params["num_channels"] == -1:
                conv_layers_params["num_channels"] = self.num_mels * self.reduction_factor

            layer = ConvBlock.create(
                index=index,
                conv_params=conv_layers_params,
                regularizer=self.regularizer,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
                cnn_dropout_prob=self.cnn_dropout_prob,
                is_training=self.is_training
            )
            self.conv_layers.append(layer)
        mag_conv_layers_params = {
            "kernel_size": [self.kernel_size],
            "stride": [1],
            "num_channels": self.decoder_hidden_size,
            "padding": "VALID",
            "is_causal": True,
            "activation_fn": tf.nn.relu
        }
        for index in range(self.mag_conv_layers_num):
            if mag_conv_layers_params["num_channels"] == -1:
                mag_conv_layers_params["num_channels"] = self.num_freq * self.reduction_factor

            layer = ConvBlock.create(
                index=index,
                conv_params=mag_conv_layers_params,
                regularizer=self.regularizer,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
                cnn_dropout_prob=self.cnn_dropout_prob,
                is_training=self.is_training
            )
            self.mag_conv_layers.append(layer)

        self.stop_token_projection_layer = tf.layers.Dense(
            name="stop_token_projection",
            units=1 * self.reduction_factor,
            use_bias=True,
            kernel_regularizer=self.regularizer
        )

        self.mel_projection_layer = tf.layers.Dense(
            name="mel_projection",
            units=self.num_mels * self.reduction_factor,
            use_bias=True,
            kernel_regularizer=self.regularizer
        )

        self.mag_projection_layer = tf.layers.Dense(
            name="mag_projection",
            units=self.num_freq * self.reduction_factor,
            use_bias=True,
            kernel_regularizer=self.regularizer
        )
        if self.is_training or self.is_validation:
            # _train
            # Shift targets to the right, and remove the last element
            with tf.name_scope("shift_targets"):
                n_features = self.num_mels + self.num_freq
                targets = targets[:, :, :n_features]
                targets = self._shrink(targets, n_features, self.reduction_factor)
                decoder_inputs = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

            outputs = self._decode_pass(
                decoder_inputs=decoder_inputs,
                encoder_outputs=encoder_outputs,
                enc_dec_attention_bias=attention_bias,
                sequence_lengths=targets_length
            )

            with tf.variable_scope("alignments"):
                weights = []
                for index, attention in enumerate(self.attentions):
                    if isinstance(attention, AttentionBlock):
                        weights.append(attention.multiheaded_attention.attention_weights)
                outputs["alignments"] = [tf.stack(weights)]

            return self._convert_outputs(outputs, self.reduction_factor, batch_size_per_gpu)

        # _infer
        if targets_length is None:
            maximum_iterations = duration_max
        else:
            maximum_iterations = tf.reduce_max(targets_length)

        maximum_iterations //= self.reduction_factor

        state, state_shape_invariants = self._inference_initial_state(
            encoder_outputs,
            attention_bias
        )

        state = tf.while_loop(
            cond=self._inference_cond,
            body=self._inference_step,
            loop_vars=[state],
            shape_invariants=state_shape_invariants,
            back_prop=False,
            maximum_iterations=maximum_iterations,
            parallel_iterations=1
        )

        return self._convert_outputs(state["outputs"], self.reduction_factor, batch_size_per_gpu)

    def _decode_pass(self,
                     decoder_inputs,
                     encoder_outputs,
                     enc_dec_attention_bias,
                     sequence_lengths=None,
                     alignment_positions=None):
        y = self.prenet(decoder_inputs)
        y = self.linear_projection(y)

        with tf.variable_scope("decoder_pos_encoding"):
            pos_encoding = self._positional_encoding(y, self.dtype)
            y += pos_encoding

        with tf.variable_scope("encoder_pos_encoding"):
            pos_encoding = self._positional_encoding(encoder_outputs, self.dtype)
            encoder_outputs += pos_encoding

        for i, attention in enumerate(self.attentions):
            positions = None

            if alignment_positions is not None:
                positions = alignment_positions[i, :, :, :]

            y = attention(y, encoder_outputs, enc_dec_attention_bias, positions=positions)

        y = self.output_normalization(y)

        with tf.variable_scope("conv_layers"):
            for layer in self.conv_layers:
                y = layer(y)

        stop_token_logits = self.stop_token_projection_layer(y)
        mel_spec = self.mel_projection_layer(y)

        with tf.variable_scope("mag_conv_layers"):
            for layer in self.mag_conv_layers:
                y = layer(y)

        mag_spec = self.mag_projection_layer(y)

        if sequence_lengths is None:
            batch_size = tf.shape(y)[0]
            sequence_lengths = tf.zeros([batch_size])
        return {
            "spec": mel_spec,
            "post_net_spec": mel_spec,
            "alignments": None,
            "stop_token_logits": stop_token_logits,
            "lengths": sequence_lengths,
            "mag_spec": mag_spec
        }

    def _inference_initial_state(self, encoder_outputs, encoder_decoder_attention_bias):
        """Create initial state for inference."""

        with tf.variable_scope("inference_initial_state"):
            n_layers = self.attention_layers
            n_heads = self.attention_heads
            batch_size = tf.shape(encoder_outputs)[0]
            n_features = self.num_mels + self.num_freq

            state = {
                "iteration": tf.constant(0),
                "inputs": tf.zeros([batch_size, 1, n_features * self.reduction_factor]),
                "finished": tf.cast(tf.zeros([batch_size]), tf.bool),
                "alignment_positions": tf.zeros([n_layers, batch_size, n_heads, 1],
                                                dtype=tf.int32),
                "outputs": {
                    "spec": tf.zeros([batch_size, 0, self.num_mels * self.reduction_factor]),
                    "post_net_spec": tf.zeros([batch_size, 0, self.num_mels * self.reduction_factor]),
                    "alignments": [
                        tf.zeros([0, 0, 0, 0, 0])
                    ],
                    "stop_token_logits": tf.zeros([batch_size, 0, 1 * self.reduction_factor]),
                    "lengths": tf.zeros([batch_size], dtype=tf.int32),
                    "mag_spec": tf.zeros([batch_size, 0, self.num_freq * self.reduction_factor])
                },
                "encoder_outputs": encoder_outputs,
                "encoder_decoder_attention_bias": encoder_decoder_attention_bias
            }

            state_shape_invariants = {
                "iteration": tf.TensorShape([]),
                "inputs": tf.TensorShape([None, None, n_features * self.reduction_factor]),
                "finished": tf.TensorShape([None]),
                "alignment_positions": tf.TensorShape([n_layers, None, n_heads, None]),
                "outputs": {
                    "spec": tf.TensorShape([None, None, self.num_mels * self.reduction_factor]),
                    "post_net_spec": tf.TensorShape([None, None, self.num_mels * self.reduction_factor]),
                    "alignments": [
                        tf.TensorShape([None, None, None, None, None]),
                    ],
                    "stop_token_logits": tf.TensorShape([None, None, 1 * self.reduction_factor]),
                    "lengths": tf.TensorShape([None]),
                    "mag_spec": tf.TensorShape([None, None, None])
                },
                "encoder_outputs": encoder_outputs.shape,
                "encoder_decoder_attention_bias": encoder_decoder_attention_bias.shape
            }

            return state, state_shape_invariants

    @staticmethod
    def _inference_cond(state):
        """Check if it's time to stop inference."""

        with tf.variable_scope("inference_cond"):
            all_finished = math_ops.reduce_all(state["finished"])
            return tf.logical_not(all_finished)

    def _inference_step(self, state):
        """Make one inference step."""

        decoder_inputs = state["inputs"]
        encoder_outputs = state["encoder_outputs"]
        attention_bias = state["encoder_decoder_attention_bias"]
        alignment_positions = state["alignment_positions"]

        outputs = self._decode_pass(
            decoder_inputs=decoder_inputs,
            encoder_outputs=encoder_outputs,
            enc_dec_attention_bias=attention_bias,
            alignment_positions=alignment_positions
        )

        with tf.variable_scope("inference_step"):
            next_inputs_mel = outputs["post_net_spec"][:, -1:, :]
            next_inputs_mel = self._expand(next_inputs_mel, self.reduction_factor)
            next_inputs_mag = outputs["mag_spec"][:, -1:, :]
            next_inputs_mag = self._expand(next_inputs_mag, self.reduction_factor)
            next_inputs = tf.concat([next_inputs_mel, next_inputs_mag], axis=-1)

            n_features = self.num_mels + self.num_freq
            next_inputs = self._shrink(next_inputs, n_features, self.reduction_factor)

            # Set zero if sequence is finished
            next_inputs = tf.where(
                state["finished"],
                tf.zeros_like(next_inputs),
                next_inputs
            )
            next_inputs = tf.concat([decoder_inputs, next_inputs], 1)

            # Update lengths
            lengths = state["outputs"]["lengths"]
            lengths = tf.where(
                state["finished"],
                lengths,
                lengths + 1 * self.reduction_factor
            )
            outputs["lengths"] = lengths

            # Update spec, post_net_spec and mag_spec
            for key in ["spec", "post_net_spec", "mag_spec"]:
                output = outputs[key][:, -1:, :]
                output = tf.where(state["finished"], tf.zeros_like(output), output)
                outputs[key] = tf.concat([state["outputs"][key], output], 1)

            # Update stop token logits
            stop_token_logits = outputs["stop_token_logits"][:, -1:, :]
            stop_token_logits = tf.where(
                state["finished"],
                tf.zeros_like(stop_token_logits) + 1e9,
                stop_token_logits
            )
            stop_prediction = tf.sigmoid(stop_token_logits)
            stop_prediction = tf.reduce_max(stop_prediction, axis=-1)

            # Uncomment next line if you want to use stop token predictions
            finished = tf.reshape(tf.cast(tf.round(stop_prediction), tf.bool), [-1])
            finished = tf.reshape(finished, [-1])

            stop_token_logits = tf.concat(
                [state["outputs"]["stop_token_logits"], stop_token_logits],
                axis=1
            )
            outputs["stop_token_logits"] = stop_token_logits

            with tf.variable_scope("alignments"):
                weights = []
                for index, attention in enumerate(self.attentions):
                    if isinstance(attention, AttentionBlock):
                        weights.append(attention.multiheaded_attention.attention_weights)

                weights = tf.stack(weights)
                outputs["alignments"] = [weights]

            alignment_positions = tf.argmax(
                weights,
                axis=-1,
                output_type=tf.int32
            )[:, :, :, -1:]
            state["alignment_positions"] = tf.concat(
                [state["alignment_positions"], alignment_positions],
                axis=-1
            )

            state["iteration"] = state["iteration"] + 1
            state["inputs"] = next_inputs
            state["finished"] = finished
            state["outputs"] = outputs

        return state

    @staticmethod
    def _shrink(values, last_dim, reduction_factor):
        """Shrink the given input by reduction_factor."""

        shape = tf.shape(values)
        new_shape = [
            shape[0],
            shape[1] // reduction_factor,
            last_dim * reduction_factor
        ]
        values = tf.reshape(values, new_shape)
        return values

    @staticmethod
    def _expand(values, reduction_factor):
        """Expand the given input by reduction_factor."""

        shape = tf.shape(values)
        new_shape = [
            shape[0],
            shape[1] * reduction_factor,
            shape[2] // reduction_factor
        ]
        values = tf.reshape(values, new_shape)
        return values

    @staticmethod
    def _positional_encoding(x, dtype):
        """Add positional encoding to the given input."""

        length = tf.shape(x)[1]
        features_count = tf.shape(x)[2]
        features_count += features_count % 2
        pos_encoding = get_position_encoding(length, features_count)
        position_encoding = tf.cast(pos_encoding, dtype)
        position_encoding = position_encoding[:, :features_count]
        return position_encoding

    @staticmethod
    def _convert_outputs(outputs, reduction_factor, batch_size):
        """Convert output of the decoder to appropriate format."""

        with tf.variable_scope("output_converter"):
            for key in ["spec", "post_net_spec", "stop_token_logits", "mag_spec"]:
                outputs[key] = CentaurDecoder._expand(outputs[key], reduction_factor)

            alignments = []
            for sample in range(batch_size):
                alignments.append([outputs["alignments"][0][:, sample, :, :, :]])
            mel_spec = outputs["spec"]
            post_net_spec = outputs["post_net_spec"]
            alignments = alignments
            stop_token_logits = tf.sigmoid(outputs["stop_token_logits"])
            sequence_lengths = outputs["lengths"]
            mag_spec = outputs["mag_spec"]
            stop_token_prediction = outputs["stop_token_logits"]
            return mel_spec, post_net_spec, alignments, stop_token_logits, sequence_lengths, mag_spec, stop_token_prediction
