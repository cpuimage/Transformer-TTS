import argparse
from multiprocessing import cpu_count

from hparams import hparams
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils.text_sequence import text_to_sequence
import numpy as np
from utils import audio


def build_from_path(hparams, input_dir, out_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - input_dir: input directory that contains the files to prerocess
        - out_dir: output directory of npz files
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    if isinstance(input_dir, str):
        sub_dirs = []
        if not os.path.exists(os.path.join(input_dir, 'transcript.txt')):
            sub_names = os.listdir(input_dir)
            for name in sub_names:
                sub_dirs.append(os.path.join(input_dir, name))
        else:
            sub_dirs = [input_dir]
    else:
        sub_dirs = input_dir
    for sub_dir in sub_dirs:
        with open(os.path.join(sub_dir, 'transcript.txt'), encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) < hparams.batch_size:
                continue
            for line in lines:
                wav_path, _, pinyin_text = line.strip().split('|')
                wav_path = os.path.join(sub_dir, wav_path)
                if not os.path.exists(wav_path):
                    continue
                futures.append(executor.submit(
                    partial(_process_utterance, out_dir, index, wav_path, pinyin_text, hparams)))
                index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(out_dir, index, wav_path, text, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    # Trim lead/trail silences
    if hparams.trim_silence:
        wav = audio.trim_silence(wav, hparams)

    # Pre-emphasize
    preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

    # rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

        # Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))
        if (preem_wav > 1.).any() or (preem_wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))

    # [-1, 1]
    out = wav
    constant_values = 0.

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if (mel_frames > hparams.max_mel_frames and hparams.clip_mels_length) or (
            hparams.min_text_tokens > len(text) or hparams.min_mel_frames > mel_frames):
        return None

    # Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio.linearspectrogram(preem_wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    # sanity check
    assert linear_frames == mel_frames

    # Ensure time resolution adjustement between audio and mel-spectrogram
    l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.hop_size, hparams.pad_sides)

    # Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

    assert len(out) >= mel_frames * hparams.hop_size

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * hparams.hop_size]
    assert len(out) % hparams.hop_size == 0
    time_steps = len(out)
    npz_filename = '{}.npz'.format(index)
    mel_spectrogram = mel_spectrogram.T
    linear_spectrogram = linear_spectrogram.T

    r = hparams.reduction_factor
    if hparams.symmetric_mels:
        _pad_value = -hparams.max_abs_value
    else:
        _pad_value = 0.
    target_length = len(linear_spectrogram)
    mel_spectrogram = np.pad(mel_spectrogram, [[r, r], [0, 0]], "constant", constant_values=_pad_value)
    linear_spectrogram = np.pad(linear_spectrogram, [[r, r], [0, 0]], "constant", constant_values=_pad_value)
    target_length = target_length + 2 * r
    padded_target_length = (target_length // r + 1) * r
    num_pad = padded_target_length - target_length
    stop_token_target = np.pad(np.zeros(padded_target_length - 1, dtype=np.float32), (0, 1), "constant",
                               constant_values=1)
    mel_spectrogram = np.pad(mel_spectrogram, ((0, num_pad), (0, 0)), "constant",
                             constant_values=_pad_value)
    linear_spectrogram = np.pad(linear_spectrogram, ((0, num_pad), (0, 0)), "constant", constant_values=_pad_value)

    data = {
        'mel': mel_spectrogram,
        'linear': linear_spectrogram,
        'input_data': text_to_sequence(text),  # eos(~)
        'time_steps': time_steps,
        'stop_token_target': stop_token_target,
        'mel_frames': padded_target_length,
        'text': text,
    }
    np.savez(os.path.join(out_dir, npz_filename), **data, allow_pickle=False)
    # Return a tuple describing this training example
    return npz_filename, time_steps, padded_target_length, text


def preprocess(args, input_folder, out_dir, hparams):
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(hparams, input_folder, out_dir, args.n_jobs,
                               tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        sum_mel_frames = 0.0
        sum_time_steps = 0.0
        max_text = 0.0
        max_mel_frames = 0.0
        max_time_steps = 0.0
        for npz_filename, time_steps, mel_frames, text in metadata:
            len_text = len(text)
            lines = [npz_filename, time_steps, mel_frames, text]
            f.write('|'.join([str(x) for x in lines]) + '\n')
            sum_mel_frames += mel_frames
            sum_time_steps += time_steps
            max_text = max(max_text, len_text)
            max_time_steps = max(max_time_steps, time_steps)
            max_mel_frames = max(max_mel_frames, mel_frames)
    sr = hparams.sample_rate
    hours = sum_time_steps / sr / 3600
    print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
        len(metadata), sum_mel_frames, sum_time_steps, hours))
    print('Max input length (text chars): {}'.format(int(max_text)))
    print('Max mel frames length: {}'.format(int(max_mel_frames)))
    print('Max audio timesteps length: {}'.format(max_time_steps))


def run_preprocess(args, hparams):
    name = args.name
    if str(name).strip() == "":
        name_list = os.listdir(args.dataset)
        print("datasets : {}".format(name_list))
        for name in name_list:
            in_dir = args.dataset + str(name)
            out_dir = args.output + str(name)
            print("name : {}".format(name))
            preprocess(args, in_dir, out_dir, hparams)
    else:
        in_dir = args.dataset + str(name)
        out_dir = args.output + str(name)
        print("name : {}".format(name))
        preprocess(args, in_dir, out_dir, hparams)
    print("Sampling frequency: {}".format(hparams.sample_rate))


def main():
    print('initializing preprocessing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--dataset', type=str, default='../train_datasets/')
    parser.add_argument('--output', type=str, default='../train_data/')
    parser.add_argument('--n_jobs', type=int, default=cpu_count())
    args = parser.parse_args()

    modified_hp = hparams.parse(args.hparams)

    run_preprocess(args, modified_hp)


if __name__ == '__main__':
    main()
