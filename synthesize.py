import tensorflow as tf
from hparams import hparams, hparams_debug_string
from utils.infolog import log
from synthesizer import Synthesizer
from tqdm import tqdm
import argparse
import os


def run_eval(checkpoint_path, output_dir, hparams, sentences):
    # Create output path if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

    log(hparams_debug_string())
    synth = Synthesizer()
    synth.load(checkpoint_path, hparams)

    # Set inputs batch wise
    sentences = [sentences[i: i + hparams.synthesis_batch_size] for i in
                 range(0, len(sentences), hparams.synthesis_batch_size)]

    log('Starting Synthesis')
    for i, texts in enumerate(tqdm(sentences)):
        basenames = ['{}_sentence_{}'.format(i, j) for j in range(len(texts))]
        synth.synthesize(texts, basenames, output_dir, None)


def synthesize(args, hparams, checkpoint, sentences=None):
    output_dir = 'centaur_' + args.output_dir

    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

    run_eval(checkpoint_path, output_dir, hparams, sentences)


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run_name = args.name or args.name or args.model
    cent_checkpoint = os.path.join('logs-' + run_name, 'cent_' + args.checkpoint)
    return cent_checkpoint, modified_hp


def get_sentences(args):
    if args.text_list != '':
        with open(args.text_list, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = ["hello world."]
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='pretrained/', help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--name', help='Name of logging directory if the two models were trained together.')
    parser.add_argument('--centaur_name', help='Name of logging directory of Centaur. If trained separately')
    parser.add_argument('--model', default='Centaur')
    parser.add_argument('--output_dir', default='output/', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--text_list', default='',
                        help='Text file contains list of texts to be synthesized. Valid if mode=eval')
    args = parser.parse_args()

    cent_checkpoint, hparams = prepare_run(args)
    sentences = get_sentences(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    synthesize(args, hparams, cent_checkpoint, sentences)


if __name__ == '__main__':
    main()
