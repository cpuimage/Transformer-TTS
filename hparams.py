import tensorflow as tf
from utils.text_sequence import symbols

# Default hyperparameters
hparams = tf.contrib.training.HParams(
    # encoder
    src_vocab_size=256,
    embedding_size=256,
    output_size=256,
    encoder_conv_layers_num=4,
    cnn_dropout_prob=0.1,
    # decoder
    decoder_conv_layers_num=4,
    window_size=4,
    attention_layers=4,
    attention_heads=1,
    attention_cnn_dropout_prob=0.5,
    prenet_hidden_size=256,
    decoder_hidden_size=256,
    kernel_size=5,
    # Audio
    n_fft=1024,
    num_mels=80,

    reduction_factor=4,
    # Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
    preemphasize=True,  # whether to apply filter
    preemphasis=0.97,
    ref_level_db=20,
    min_level_db=-100,
    clip_mels_length=True,
    # For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
    max_mel_frames=1000,
    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
    rescale=True,  # Whether to rescale audio prior to preprocessing
    rescaling_max=0.999,
    # M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
    trim_silence=True,  # Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
    trim_fft_size=512,  # Trimming window size
    trim_hop_size=128,  # Trimmin hop length
    trim_top_db=23,  # Trimming db difference from reference db (smaller==harder trim.)
    num_silent_frames=4,

    # Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.
    # Mel spectrogram
    magnitude_power=2.,  # The power of the spectrogram magnitude (1. for energy, 2. for power)
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    # Whether to normalize mel spectrograms to some predefined range (following below parameters)
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,
    # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
    max_abs_value=4.,
    # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion,
    # not too small for fast convergence)
    pad_sides=1,  # Can be 1 or 2. 1 for pad right only, 2 for both sides padding.
    # Centaur Training
    batch_size=64,
    learning_rate_step_factor=1,
    initial_phase_step=8000,
    # Reproduction seeds
    random_seed=5339,
    # Determines initial graph and operations (i.e: model) random state for reproducibility
    data_random_state=1234,  # random state for train test split repeatability
    # Centaur Batch synthesis supports ~16x the training batch size (no gradients during testing).
    # Training Centaur with unmasked paddings makes it aware of them, which makes synthesis times different from training. We thus recommend masking the encoder.
    synthesis_batch_size=1,
    # DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
    test_size=0.05,
    # % of data to keep as test data, if None, test_batches must be not None. (5% is enough to have a good idea about overfit)
    test_batches=None,  # number of test batches.
    # Learning rate schedule
    initial_learning_rate=1e-3,  # starting learning rate
    # Optimization parameters
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    clip_gradients=True,  # whether to clip gradients
    # Speaker adaptation parameters
    fine_tuning=False,
    # Set to True to freeze encoder and only keep training pretrained decoder. Used for speaker adaptation with small data.
    min_text_tokens=20,  # originally 50, 30 is good,filter npz
    max_iters=1000,
    # Griffin Lim
    power=1.5,  # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
    griffin_lim_iters=60,  # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
)
# 22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
hparams.sample_rate = 16000
# Can replace hop_size parameter. (Recommended: 12.5)
hparams.frame_shift_ms = 12.5
hparams.frame_length_ms = 50.0
# For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
hparams.hop_size = int(hparams.frame_shift_ms / 1000.0 * hparams.sample_rate)
# For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
hparams.win_size = int(hparams.frame_length_ms / 1000.0 * hparams.sample_rate)
# (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
hparams.num_freq = int(hparams.n_fft / 2 + 1)
hparams.min_mel_frames = int(hparams.min_text_tokens * 5)
hparams.num_symbols = len(symbols)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
    return 'Hyperparameters:\n' + '\n'.join(hp)
