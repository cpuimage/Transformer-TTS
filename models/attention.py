import tensorflow as tf
from .modules import ConvBlock, PrePostProcessingWrapper


class MultiheadedAttention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(
            self,
            hidden_size,
            num_heads,
            attention_dropout,
            is_training,
            mode="loung",
            regularizer=None,
            window_size=None,
            back_step_size=None
    ):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(MultiheadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.is_training = is_training
        self.mode = mode
        self.attention_weights = None
        # Parameters for monotonic attention forcing during inference
        self.window_size = window_size
        self.back_step_size = back_step_size

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="q",
                                             kernel_regularizer=regularizer)
        self.k_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="k",
                                             kernel_regularizer=regularizer)
        self.v_dense_layer = tf.layers.Dense(hidden_size, use_bias=False, name="v",
                                             kernel_regularizer=regularizer)
        self.output_dense_layer = tf.layers.Dense(hidden_size, use_bias=False,
                                                  name="output_transform",
                                                  kernel_regularizer=regularizer)

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
          x: A tensor with shape [batch_size, length, hidden_size]

        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope("split_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with tf.name_scope("combine_heads"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None, positions=None):
        """Apply attention mechanism to x and y.

        Args:
          x: a tensor with shape [batch_size, length_x, hidden_size]
          y: a tensor with shape [batch_size, length_y, hidden_size]
          bias: attention bias that will be added to the result of the dot product.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {"k": tensor with shape [batch_size, i, key_channels],
                 "v": tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.
          positions: decoder-encoder alignment for previous steps [batch_size, n_heads, length_x]

        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if self.mode == "loung":
            # Scale q to prevent the dot product between q and k from growing too large.
            depth = (self.hidden_size // self.num_heads)
            q *= depth ** -0.5

            # Calculate dot product attention
            # logits = tf.matmul(q, k, transpose_b=True)
            # logits += bias
            # weights = tf.nn.softmax(logits, name="attention_weights")
            logits = tf.matmul(q, k, transpose_b=True)
            dtype = logits.dtype
            if dtype != tf.float32:
                # upcast softmax inputs
                logits = tf.cast(x=logits, dtype=tf.float32)
                logits += bias
                self.attention_weights = tf.nn.softmax(logits, name="attention_weights")
                # downcast softmax output
                self.attention_weights = tf.cast(self.attention_weights, dtype=dtype)
            else:
                # Logits shape: [batch, head, decoder, encoder]
                # Bias shape:   [batch, 1, 1, encoder]

                # Force monotonic attention during inference
                if positions is not None and self.window_size is not None:
                    assert self.back_step_size is not None

                    max_length = tf.shape(logits)[-1]

                    # Allow to make back_step_size steps back
                    window_pos = tf.maximum(positions - self.back_step_size, tf.zeros_like(positions))

                    # Create attention mask
                    mask_large = tf.sequence_mask(window_pos + self.window_size, maxlen=max_length)
                    mask_large = tf.cast(mask_large, tf.float32)
                    mask_small = tf.sequence_mask(window_pos, maxlen=max_length)
                    mask_small = tf.cast(mask_small, tf.float32)
                    mask = mask_large - mask_small
                    mask = -1e9 * (1 - mask)

                    bias = mask + bias

                    # Clipping
                    bias = tf.maximum(bias, -1e9)

                logits += bias
                self.attention_weights = tf.nn.softmax(logits, name="attention_weights")
        elif self.mode == "bahdanau":
            att_v = tf.get_variable(
                "attention_v", [self.hidden_size // self.num_heads], dtype=q.dtype
            )

            # Compute the attention score
            if bias is not None:
                self.attention_weights = tf.reduce_sum(
                    tf.nn.tanh(att_v * tf.nn.tanh(k + q + bias)), 3
                )
            else:
                self.attention_weights = tf.reduce_sum(
                    tf.nn.tanh(att_v * tf.nn.tanh(k + q)), 3
                )
            self.attention_weights = tf.nn.softmax(self.attention_weights)
            self.attention_weights = tf.expand_dims(self.attention_weights, 2)
        else:
            raise ValueError(
                "Mode for multi-head attention must be either loung for dot-product",
                "attention, or bahdanau for content-based/additive/mlp-base attention"
            )

        if self.is_training:
            self.attention_weights = tf.nn.dropout(self.attention_weights, keep_prob=1 - self.attention_dropout)
        attention_output = tf.matmul(self.attention_weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class AttentionBlock:
    """
    Attention block for Centaur model.
    """

    def __init__(self,
                 hidden_size,
                 attention_dropout,
                 layer_postprocess_dropout,
                 is_training,
                 cnn_dropout_prob,
                 regularizer=None,
                 conv_params=None,
                 n_heads=1,
                 window_size=None,
                 back_step_size=None,
                 name="attention_block"):
        """
        Attention block constructor.

        Args:
          hidden_size: dimensionality of hidden embeddings.
          attention_dropout: dropout rate for attention layer.
          layer_postprocess_dropout:  dropout rate for sublayer.
          is_training: whether it is training mode.
          cnn_dropout_prob: dropout probabilty for cnn layers.
          regularizer: regularizer for the convolution kernel.
          conv_params: description of convolutional layer.
          n_heads: number of attention heads. Defaults to 1.
          window_size: size of attention window for forcing
            monotonic attention during the inference. Defaults to None.
          back_step_size: number of steps attention is allowed to
            go back during the inference. Defaults to 0.
          name: name of the block.
        """

        self.name = name
        self.conv = None

        if conv_params:
            self.conv = ConvBlock.create(
                index=0,
                conv_params=conv_params,
                regularizer=regularizer,
                bn_momentum=0.95,
                bn_epsilon=1e-8,
                cnn_dropout_prob=cnn_dropout_prob,
                is_training=is_training
            )
            self.conv.name = "conv"

        self.multiheaded_attention = MultiheadedAttention(
            hidden_size=hidden_size,
            num_heads=n_heads,
            attention_dropout=attention_dropout,
            regularizer=regularizer,
            is_training=is_training,
            window_size=window_size,
            back_step_size=back_step_size,
        )

        feed_forward = tf.layers.Dense(
            units=hidden_size,
            use_bias=True,
            kernel_regularizer=regularizer
        )

        self.attention = PrePostProcessingWrapper(
            layer=self.multiheaded_attention,
            hidden_size=hidden_size,
            layer_postprocess_dropout=layer_postprocess_dropout,
            is_training=is_training
        )

        self.feed_forward = PrePostProcessingWrapper(
            layer=feed_forward,
            hidden_size=hidden_size,
            layer_postprocess_dropout=layer_postprocess_dropout,
            is_training=is_training
        )

    def __call__(self,
                 decoder_inputs,
                 encoder_outputs,
                 attention_bias,
                 positions=None):
        with tf.variable_scope(self.name):
            y = decoder_inputs
            if self.conv:
                y = self.conv(y)
            with tf.variable_scope("attention"):
                y = self.attention(
                    y,
                    encoder_outputs,
                    attention_bias,
                    positions=positions
                )
            with tf.variable_scope("feed_forward"):
                y = self.feed_forward(y)
            return y
