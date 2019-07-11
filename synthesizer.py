import os
import platform
import numpy as np
import tensorflow as tf
from utils.infolog import log
from models import create_model
from utils import plot, audio
from utils.text_sequence import text_to_sequence


class Synthesizer:
    def load(self, checkpoint_path, hparams, freezer=False):
        log('Constructing model: Centaur')
        if freezer:
            try:
                checkpoint_path = tf.train.get_checkpoint_state(checkpoint_path).model_checkpoint_path
            except:
                raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint_path))
        # Force the batch size to be known in order to use attention masking in batch synthesis
        self.inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
        self.input_lengths = tf.placeholder(tf.int32, (None,), name='input_lengths')

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.model = create_model(hparams)
            self.model.initialize(self.inputs, self.input_lengths, is_training=False,
                                  is_validation=False, is_prediction=True)
            self.mel_outputs = self.model.decoder_predictions
            self.linear_outputs = self.model.mag_pred
            self.alignments = self.model.alignments
            self.wav_output = self.model.audio
            self.stop_token_prediction = self.model.stop_token_predictions
            self.audio_length = self.model.sequence_lengths

        self._hparams = hparams
        # pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0

        log('Loading checkpoint: %s' % checkpoint_path)
        # Memory allocation on the GPUs as needed
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def calc_num_pad(self, r, target_length):
        padded_target_length = (target_length // r + 1) * r
        num_pad = padded_target_length - target_length
        return num_pad

    def get_alignments(self, attention_mask):
        alignments_name = ["align"]

        specs = []
        titles = []

        for name, alignment in zip(alignments_name, attention_mask):
            for layer in range(len(alignment)):
                for head in range(alignment.shape[1]):
                    specs.append(alignment[layer][head])
                    titles.append("{}_layer_{}_head_{}".format(name, layer, head))

        return specs, titles

    def synthesize(self, texts, basenames, log_dir, mel_filenames):
        hparams = self._hparams

        # Repeat last sample until number of samples is dividable by the number of GPUs (last run scenario)
        while len(texts) % hparams.synthesis_batch_size != 0:
            texts.append(texts[-1])
            basenames.append(basenames[-1])
            if mel_filenames is not None:
                mel_filenames.append(mel_filenames[-1])
        sequences = [np.asarray(text_to_sequence(text)) for text in texts]
        input_lengths = [len(seq) for seq in sequences]
        seqs, max_seq_len = self._prepare_inputs(sequences)

        feed_dict = {
            self.inputs: seqs,
            self.input_lengths: np.asarray(input_lengths, dtype=np.int32)
        }

        linears, mels, alignments, audio_length = self.session.run(
            [self.linear_outputs, self.mel_outputs, self.alignments[0], self.audio_length],
            feed_dict=feed_dict)
        # Natural batch synthesis
        # Get Mel/Linear lengths for the entire batch from stop_tokens predictions
        target_lengths = audio_length

        if basenames is None:
            # Generate wav and read it
            wav = audio.inv_mel_spectrogram(mels[0].T, hparams)
            audio.save_wav(wav, 'temp.wav', sr=hparams.sample_rate)  # Find a better way

            if platform.system() == 'Linux':
                # Linux wav reader
                os.system('aplay temp.wav')

            elif platform.system() == 'Windows':
                # windows wav reader
                os.system('start /min mplay32 /play /close temp.wav')

            else:
                raise RuntimeError(
                    'Your OS type is not supported yet, please add it to "centaur/synthesizer.py, line-165" and feel free to make a Pull Request ;) Thanks!')

            return

        for i, mel in enumerate(mels):

            if log_dir is not None:
                # save wav (mel -> wav)
                wav = audio.inv_mel_spectrogram(mel.T, hparams)
                audio.save_wav(wav, os.path.join(log_dir, 'wavs/wav-{}-mel.wav'.format(basenames[i])),
                               sr=hparams.sample_rate)
                alignments_samples, alignment_titles = self.get_alignments(alignments)
                for idx in range(len(alignments_samples)):
                    # save alignments
                    plot.plot_alignment(alignments_samples[idx],
                                        os.path.join(log_dir, 'plots/{}.png'.format(
                                            alignment_titles[
                                                idx])),
                                        title='{}'.format(texts[i]), split_title=True, max_len=target_lengths[i])

                # save mel spectrogram plot
                plot.plot_spectrogram(mel,
                                      os.path.join(log_dir, 'plots/mel-{}.png'.format(basenames[i])),
                                      title='{}'.format(texts[i]), split_title=True)

                # save wav (linear -> wav)

                wav = audio.inv_linear_spectrogram(linears[i].T, hparams)
                audio.save_wav(wav,
                               os.path.join(log_dir, 'wavs/wav-{}-linear.wav'.format(basenames[i])),
                               sr=hparams.sample_rate)

                # save linear spectrogram plot
                plot.plot_spectrogram(linears[i],
                                      os.path.join(log_dir, 'plots/linear-{}.png'.format(basenames[i])),
                                      title='{}'.format(texts[i]), split_title=True, auto_aspect=True)

    @staticmethod
    def _round_up(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    def _prepare_inputs(self, inputs):
        max_len = max([len(x) for x in inputs])
        return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)
