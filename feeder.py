import os
import threading
import time

import numpy as np
import tensorflow as tf
from utils.infolog import log
from sklearn.model_selection import train_test_split


class Feeder:
    """
        Feeds batches of data into queue on a background thread.
    """

    def __init__(self, coordinator, input_path, hparams):
        super(Feeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._start_step = 0
        self._batches_per_group = 32
        self._train_offset_list = []
        self._test_offset_list = []
        self._pad = 0
        # Load metadata
        input_paths = [input_path]
        if not os.path.exists(os.path.join(input_path, 'train.txt')):
            path_list = os.listdir(input_path)
            input_paths = []
            for name in path_list:
                input_paths.append(os.path.abspath(os.path.join(input_path, name)))
        self.data_dirs = input_paths
        all_hours = 0.0
        metadata_size = 0
        self._metadata_list = []
        for input_path in input_paths:
            with open(os.path.join(input_path, 'train.txt'), encoding='utf-8') as f:
                metadata_vec = []
                for line in f:
                    npz_filename, time_steps, mel_frames, text = line.strip().split('|')
                    metadata_vec.append(
                        [os.path.join(input_path, os.path.basename(npz_filename)), time_steps, mel_frames, text])
                self._metadata_list.append(metadata_vec)
                frame_shift_ms = hparams.hop_size / hparams.sample_rate
                hours = sum([int(x[2]) for x in metadata_vec]) * frame_shift_ms / 3600
                all_hours += hours
                log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata_vec), hours))
                metadata_size += len(metadata_vec)
                self._train_offset_list.append(0)
                self._test_offset_list.append(0)
        log('Loaded ({:.2f} hours)'.format(all_hours))
        # Train test split
        if hparams.test_size is None:
            assert hparams.test_batches is not None
        test_size = (hparams.test_size if hparams.test_size is not None
                     else hparams.test_batches * hparams.batch_size)
        self._train_meta_list = []
        self._test_meta_list = []
        if self._hparams.symmetric_mels:
            self._pad_value = -self._hparams.max_abs_value
        else:
            self._pad_value = 0.
        data_ratio = 1.0 // len(self._metadata_list)
        test_size = test_size * data_ratio
        sum_test_meta = 0
        for metadata in self._metadata_list:
            indices = np.arange(len(metadata))
            train_indices, test_indices = train_test_split(indices,
                                                           test_size=test_size,
                                                           random_state=hparams.data_random_state)
            # Make sure test_indices is a multiple of batch_size else round down
            len_test_indices = self._round_down(len(test_indices), hparams.batch_size)
            extra_test = test_indices[len_test_indices:]
            test_indices = test_indices[:len_test_indices]
            train_indices = np.concatenate([train_indices, extra_test])
            _train_meta = list(np.array(metadata)[train_indices])
            _test_meta = list(np.array(metadata)[test_indices])
            sum_test_meta += len(_test_meta)
            self._train_meta_list.append(_train_meta)
            self._test_meta_list.append(_test_meta)
        self.test_steps = sum_test_meta // hparams.batch_size
        if hparams.test_size is None:
            assert hparams.test_batches == self.test_steps

        with tf.device('/cpu:0'):
            # Create placeholders for inputs and targets. Don't specify batch size because we want
            # to be able to feed different batch sizes at eval time.

            self._placeholders = [
                tf.placeholder(tf.int32, [None, None], 'inputs'),
                tf.placeholder(tf.int32, [None], 'input_lengths'),
                tf.placeholder(tf.float32, [None, None, hparams.num_mels + hparams.num_freq], 'target_mels'),
                tf.placeholder(tf.int32, (None,), 'target_lengths'),
                tf.placeholder(tf.float32, (None, None), 'stop_tokens'),
            ]
            dtypes = [tf.int32, tf.int32, tf.float32, tf.int32, tf.float32]
            # Create queue for buffering data
            queue = tf.FIFOQueue(8, dtypes, name='input_queue')
            self._enqueue_op = queue.enqueue(self._placeholders)

            self.inputs, self.input_lengths, self.target_feats, self.target_lengths, self.stop_tokens = queue.dequeue()
            self.inputs.set_shape(self._placeholders[0].shape)
            self.input_lengths.set_shape(self._placeholders[1].shape)
            self.target_feats.set_shape(self._placeholders[2].shape)
            self.target_lengths.set_shape(self._placeholders[3].shape)
            self.stop_tokens.set_shape(self._placeholders[4].shape)

            # Create eval queue for buffering eval data
            eval_queue = tf.FIFOQueue(1, dtypes, name='eval_queue')
            self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)

            self.eval_inputs, self.eval_input_lengths, self.eval_target_feats, self.eval_target_lengths, self.eval_stop_tokens, = eval_queue.dequeue()

            self.eval_inputs.set_shape(self._placeholders[0].shape)
            self.eval_input_lengths.set_shape(self._placeholders[1].shape)
            self.eval_target_feats.set_shape(self._placeholders[2].shape)
            self.eval_target_lengths.set_shape(self._placeholders[3].shape)
            self.eval_stop_tokens.set_shape(self._placeholders[4].shape)

    def start_threads(self, session, start_step):
        self._start_step = start_step
        self._session = session
        thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

        thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
        thread.daemon = True  # Thread will close when parent quits
        thread.start()

    def _get_test_groups(self, _test_meta, idx):
        if self._test_offset_list[idx] >= len(_test_meta):
            self._test_offset_list[idx] = 0
        npz_filename, time_steps, mel_frames, text = _test_meta[self._test_offset_list[idx]]
        self._test_offset_list[idx] += 1
        npz_data = np.load(npz_filename)

        input_data = npz_data['input_data']
        target_mel = npz_data['mel']
        target_spec = npz_data['linear']
        stop_token_target = npz_data['stop_token_target']
        return input_data, target_mel, target_spec, stop_token_target, len(target_mel)

    def make_test_batches(self):
        start = time.time()
        # Read a group of examples
        n = self._hparams.batch_size
        r = self._hparams.reduction_factor
        # Test on entire test set
        examples = []
        examples_list = []
        examples_size = []
        data_ratio = 1.0 / len(self._test_meta_list)
        for idx, _test_meta in enumerate(self._test_meta_list):
            example = [self._get_next_example(_test_meta, idx) for _ in
                       range(int(n * self._batches_per_group * data_ratio))]
            example.sort(key=lambda x: x[-1])
            examples_size.append(len(example))
            examples_list.append(example)
        examples_size.sort(reverse=True)
        max_step = examples_size[0] if len(examples_size) > 0 else 0
        num_vec = len(examples_size)
        for index in range(max_step):
            for num in range(num_vec):
                if examples_size[num] > index:
                    example = examples_list[num][index]
                    examples.append(example)
        batches = [examples[i: i + n] for i in range(0, len(examples), n)]
        np.random.shuffle(batches)
        log('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
        return batches, r

    def _enqueue_next_train_group(self):
        while not self._coord.should_stop():
            start = time.time()

            # Read a group of examples
            n = self._hparams.batch_size
            # Bucket examples based on similar output sequence length for efficiency
            examples = []  # 存放预训练样本
            examples_list = []  # 总共有多少个人
            examples_size = []  # 每个样本量多大
            data_ratio = 1.0 / len(self._train_meta_list)
            for idx, _train_meta in enumerate(self._train_meta_list):
                if self._start_step < self._hparams.initial_phase_step:  # 'initial_phase_step': 8000
                    example = [self._get_next_example(_train_meta, idx) for _ in
                               range(int(n * self._batches_per_group // len(
                                   self.data_dirs)))]
                else:
                    example = [self._get_next_example(_train_meta, idx) for _ in
                               range(int(n * self._batches_per_group * data_ratio))]
                example.sort(key=lambda x: x[-1])
                examples_size.append(len(example))
                examples_list.append(example)
            examples_size.sort(reverse=True)
            max_step = examples_size[0] if len(examples_size) > 0 else 0
            num_vec = len(examples_size)
            for index in range(max_step):
                for num in range(num_vec):
                    if examples_size[num] > index:
                        example = examples_list[num][index]
                        examples.append(example)
            batches = [examples[i: i + n] for i in range(0, len(examples), n)]
            np.random.shuffle(batches)

            log('\nGenerated {} train batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
            for batch in batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
                self._session.run(self._enqueue_op, feed_dict=feed_dict)
                self._start_step += 1

    def _enqueue_next_test_group(self):
        # Create test batches once and evaluate on them for all test steps
        test_batches, r = self.make_test_batches()
        while not self._coord.should_stop():
            for batch in test_batches:
                feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch)))
                self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self, _train_meta, idx):
        """Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
        """
        if self._train_offset_list[idx] >= len(_train_meta):
            self._train_offset_list[idx] = 0
            np.random.shuffle(_train_meta)
        npz_filename, time_steps, mel_frames, text = _train_meta[self._train_offset_list[idx]]
        self._train_offset_list[idx] += 1
        npz_data = np.load(npz_filename)
        input_data = npz_data['input_data']
        target_mel = npz_data['mel']
        target_spec = npz_data['linear']
        stop_token_target = npz_data['stop_token_target']
        return input_data, target_mel, target_spec, stop_token_target, len(target_mel)

    def _prepare_batch(self, batches):
        np.random.shuffle(batches)
        inputs = []
        input_lengths = []
        target_feats = []
        target_lengths = []
        stop_tokens = []
        for item in batches:
            text_input = item[0]
            mel_spectrogram = item[1]
            spectrogram = item[2]
            stop_token_target = item[3]
            target_feat = np.concatenate((mel_spectrogram, spectrogram), axis=1)
            target_feats.append(target_feat)
            stop_tokens.append(stop_token_target)
            target_lengths.append(len(target_feat))
            inputs.append(text_input)
            input_lengths.append(len(text_input))
        max_frames = max((len(t) for t in target_feats))
        max_inputs = max((len(t) for t in inputs))
        max_stop_tokens = max((len(t) for t in stop_tokens))
        inputs = np.stack([self._pad_1d(item, max_inputs, self._pad) for item in inputs])
        input_lengths = np.asarray(input_lengths, dtype=np.int32)
        target_feats = np.stack([self._pad_2d(item, max_frames) for item in target_feats])
        target_lengths = np.stack(target_lengths)
        stop_tokens = np.stack([self._pad_1d(item, max_stop_tokens, 1) for item in stop_tokens])
        return inputs, input_lengths, target_feats, target_lengths, stop_tokens

    @staticmethod
    def _pad_1d(t, length, pad):
        return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=pad)

    @staticmethod
    def _pad_2d(t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=0)

    def _prepare_inputs(self, inputs):
        max_len = max((len(x) for x in inputs))
        return np.stack([self._pad_1d(x, max_len, 0) for x in inputs])

    @staticmethod
    def _round_up(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder

    @staticmethod
    def _round_down(x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder
