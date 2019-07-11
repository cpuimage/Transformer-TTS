# -*- coding: utf-8 -*-

from .models import *
from .attention import *
from utils.infolog import log
import numpy as np
from utils.audio import inv_linear_spectrogram_tensorflow


class Centaur:
    """Centaur Feature prediction Model.
    """

    def __init__(self, params):
        self.params = params
        self.is_training = False
        self.all_vars = dict()
        self.linear_loss = 0
        self.stop_token_loss = 0
        self.mel_loss = 0
        self.loss = 0
        self.learning_rate = 0
        self.encoder_outputs = None
        self.alignments = None
        self.decoder_predictions = None
        self.post_net_predictions = None
        self.stop_token_predictions = None
        self.mag_pred = None
        self.inputs = None
        self.input_lengths = None
        self.target_feats = None
        self.mag_target = None
        self.targets_stop_token = None
        self.targets_length = None

    def initialize(self, inputs, input_lengths, target_feats=None, targets_length=None,
                   targets_stop_token=None, is_training=False, is_validation=False,
                   is_prediction=False):
        self.is_validation = is_validation
        self.is_prediction = is_prediction
        self.is_training = is_training
        with tf.variable_scope("centaur_encoder"):
            encoder_outputs, attention_bias = CentaurEncoder(is_training=is_training,
                                                             src_vocab_size=self.params.src_vocab_size,
                                                             embedding_size=self.params.embedding_size,
                                                             output_size=self.params.output_size,
                                                             conv_layers_num=self.params.encoder_conv_layers_num,
                                                             cnn_dropout_prob=self.params.cnn_dropout_prob)(inputs)
        with tf.variable_scope("centaur_decoder"):
            decoder_predictions, post_net_predictions, alignments, stop_token_logits, sequence_lengths, mag_pred, stop_token_predictions = CentaurDecoder(
                num_mels=self.params.num_mels,
                num_freq=self.params.num_freq,
                conv_layers_num=self.params.decoder_conv_layers_num,
                reduction_factor=self.params.reduction_factor,
                decoder_hidden_size=self.params.decoder_hidden_size,
                prenet_hidden_size=self.params.prenet_hidden_size,
                attention_layers=self.params.attention_layers,
                attention_heads=self.params.attention_heads,
                window_size=self.params.window_size,
                attention_cnn_dropout_prob=self.params.attention_cnn_dropout_prob,
                kernel_size=self.params.kernel_size,
                is_training=is_training, is_prediction=is_prediction, is_validation=is_validation)(
                targets=target_feats, targets_length=targets_length, encoder_outputs=encoder_outputs,
                attention_bias=attention_bias, batch_size_per_gpu=self.params.batch_size,
                duration_max=self.params.max_iters)

        self.encoder_outputs = encoder_outputs
        self.alignments = alignments
        self.decoder_predictions = decoder_predictions
        self.post_net_predictions = post_net_predictions

        self.stop_token_predictions = stop_token_predictions
        self.mag_pred = mag_pred
        self.sequence_lengths = sequence_lengths
        self.inputs = inputs
        self.input_lengths = input_lengths
        self.target_feats = target_feats
        self.targets_stop_token = targets_stop_token
        self.targets_length = targets_length
        self.all_vars = tf.trainable_variables()

        log('Initialized Centaur model. Dimensions (? = dynamic shape): ')
        log('  Train mode:               {}'.format(is_training))
        log('  Input:                    {}'.format(inputs.shape))
        log('  encoder out:              {}'.format(encoder_outputs.shape))
        log('  mel out:                  {}'.format(decoder_predictions.shape))
        log('  linear out:               {}'.format(mag_pred.shape))
        log('  <stop_token> out:         {}'.format(stop_token_predictions.shape))

        # 1_000_000 is causing syntax problems for some people?! Python please :)
        log('  Centaur Parameters       {:.3f} Million.'.format(
            np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))

    def add_loss(self, scale=None,
                 l1_norm=True, use_mask=True,
                 mel_weight=1.0,
                 stop_token_weight=1.0, mag_weight=1.0):

        """
        Computes loss for text-to-speech model.

        Args:
            * "decoder_output": dictionary containing:
                "outputs": array containing:
                    * mel: mel-spectrogram predicted by the decoder [batch, time, n_mel]
                    * post_net_mel: spectrogram after adding the residual
                      corrections from the post net of shape [batch, time, feats]
                    * mag: mag-spectrogram predicted by the decoder [batch, time, n_mag]
                "stop_token_predictions": stop_token predictions of shape [batch, time, 1]

            * "target_tensors": array containing:
                * spec: the true spectrogram of shape [batch, time, feats]
                * stop_token: the stop_token of shape [batch, time]
                * spec_length: the length of specs [batch]

        Returns:
          Singleton loss tensor
        """

        mag_pred = tf.cast(self.mag_pred, dtype=tf.float32)

        targets_stop_token = tf.expand_dims(self.targets_stop_token, -1)
        batch_size = tf.shape(self.target_feats)[0]
        num_feats = tf.shape(self.target_feats)[2]

        decoder_predictions = tf.cast(self.decoder_predictions, dtype=tf.float32)
        post_net_predictions = tf.cast(self.post_net_predictions, dtype=tf.float32)
        stop_token_predictions = tf.cast(self.stop_token_predictions, dtype=tf.float32)
        targets_feat = tf.cast(self.target_feats, dtype=tf.float32)
        targets_stop_token = tf.cast(targets_stop_token, dtype=tf.float32)

        max_length = tf.cast(
            tf.maximum(
                tf.shape(targets_feat)[1],
                tf.shape(decoder_predictions)[1],
            ), tf.int32
        )

        decoder_pad = tf.zeros(
            [
                batch_size,
                max_length - tf.shape(decoder_predictions)[1],
                tf.shape(decoder_predictions)[2]
            ]
        )
        stop_token_pred_pad = tf.zeros(
            [batch_size, max_length - tf.shape(decoder_predictions)[1], 1]
        )
        spec_pad = tf.zeros([batch_size, max_length - tf.shape(targets_feat)[1], num_feats])
        stop_token_pad = tf.ones([batch_size, max_length - tf.shape(targets_feat)[1], 1])
        decoder_predictions = tf.concat(
            [decoder_predictions, decoder_pad],
            axis=1
        )
        post_net_predictions = tf.concat(
            [post_net_predictions, decoder_pad],
            axis=1
        )
        stop_token_predictions = tf.concat(
            [stop_token_predictions, stop_token_pred_pad],
            axis=1
        )
        targets_feat = tf.concat([targets_feat, spec_pad], axis=1)
        targets_stop_token = tf.concat([targets_stop_token, stop_token_pad], axis=1)

        if l1_norm:
            loss_func = tf.losses.absolute_difference
        else:
            loss_func = tf.losses.mean_squared_error

        mag_pad = tf.zeros(
            [
                batch_size,
                max_length - tf.shape(mag_pred)[1],
                tf.shape(mag_pred)[2]
            ]
        )
        mag_pred = tf.concat(
            [mag_pred, mag_pad],
            axis=1
        )
        targets_mel, targets_mag = tf.split(
            targets_feat,
            [self.params.num_mels, self.params.num_freq],
            axis=2
        )
        self.targets_mag = targets_mag
        self.targets_mel = targets_mel

        decoder_target = targets_mel
        post_net_target = targets_mel
        with tf.variable_scope('loss'):
            if use_mask:
                mask = tf.sequence_mask(
                    lengths=self.targets_length,
                    maxlen=max_length,
                    dtype=tf.float32
                )
                mask = tf.expand_dims(mask, axis=-1)

                decoder_loss = loss_func(
                    labels=decoder_target,
                    predictions=decoder_predictions,
                    weights=mask
                )
                post_net_loss = loss_func(
                    labels=post_net_target,
                    predictions=post_net_predictions,
                    weights=mask
                )

                mag_loss = loss_func(
                    labels=targets_mag,
                    predictions=mag_pred,
                    weights=mask
                )

                stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets_stop_token,
                    logits=stop_token_predictions
                )
                stop_token_loss = stop_token_loss * mask
                stop_token_loss = tf.reduce_sum(stop_token_loss) / tf.reduce_sum(mask)
            else:
                decoder_loss = loss_func(
                    labels=decoder_target,
                    predictions=decoder_predictions
                )
                post_net_loss = loss_func(
                    labels=post_net_target,
                    predictions=post_net_predictions
                )

                mag_loss = loss_func(
                    labels=targets_mag,
                    predictions=mag_pred
                )
                stop_token_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets_stop_token,
                    logits=stop_token_predictions
                )
                stop_token_loss = tf.reduce_mean(stop_token_loss)

            decoder_loss = mel_weight * decoder_loss
            post_net_loss = mel_weight * post_net_loss
            stop_token_loss = stop_token_weight * stop_token_loss
            if scale:
                loss = (decoder_loss + post_net_loss + stop_token_loss + mag_weight * mag_loss) * scale
            else:
                loss = decoder_loss + post_net_loss + stop_token_loss + mag_weight * mag_loss

            self.linear_loss = mag_loss
            self.stop_token_loss = stop_token_loss
            self.mel_loss = decoder_loss + post_net_loss
            self.loss = loss

    def add_optimizer(self, global_step, hvd=None, fp16=False, loss_scale=None):
        """Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.
        Args:
            global_step: int32 scalar Tensor representing current global step in training
            fp16:
            loss_scale:
        """
        with tf.variable_scope('optimizer'):
            hvd_size = 1.0
            if hvd:
                hvd_size = hvd.size()
            self.learning_rate = self._learning_rate_decay(self.params.initial_learning_rate,
                                                           global_step) * hvd_size
            optimizer = tf.train.AdamOptimizer(self.learning_rate, self.params.adam_beta1,
                                               self.params.adam_beta2, self.params.adam_epsilon)
            if hvd:
                if fp16:
                    # Choose a loss scale manager which decides how to pick the right loss scale
                    # throughout the training process.
                    if loss_scale is None:
                        loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(2 ** 32, 1000)
                    else:
                        loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(loss_scale)
                    # Wraps the original optimizer in a LossScaleOptimizer.
                    optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)
                compression = hvd.Compression.fp16 if fp16 else hvd.Compression.none
                optimizer = hvd.DistributedOptimizer(optimizer, compression=compression)
            #  Compute Gradient
            update_vars = [v for v in self.all_vars if not (
                    'embedding_and_softmax' in str(v.name).lower() or 'centaur_encoder' in str(
                v.name).lower())] if self.params.fine_tuning else None
            gradients, variables = zip(*optimizer.compute_gradients(self.loss, var_list=update_vars))
            self.gradients = gradients
            if self.params.clip_gradients:
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)  # __mark 0.5 refer
            else:
                clipped_gradients = gradients
            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step, name='train_op')

    def add_train_stats(self):
        with tf.variable_scope('stats'):
            tf.summary.histogram('mel_outputs ', self.decoder_predictions)
            tf.summary.histogram('target_mel ', self.targets_mel)
            tf.summary.scalar('mel_loss', self.mel_loss)
            tf.summary.scalar('linear_loss', self.linear_loss)
            tf.summary.histogram('linear_outputs ', self.mag_pred)
            tf.summary.scalar('stop_token_loss', self.stop_token_loss)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('learning_rate', self.learning_rate)  # Control learning rate decay speed
            gradient_norms = [tf.norm(grad) for grad in self.gradients]
            tf.summary.histogram('gradient_norm', gradient_norms)
            tf.summary.scalar('max_gradient_norm',
                              tf.reduce_max(gradient_norms))  # visualize gradients (in case of explosion)
            return tf.summary.merge_all()

    @staticmethod
    def _learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
        """Noam scheme from tensor2tensor"""
        step = tf.cast(global_step + 1, dtype=tf.float32)
        return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
