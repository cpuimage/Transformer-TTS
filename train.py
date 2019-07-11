import argparse
import os
import time
import traceback
from datetime import datetime

import numpy as np
import tensorflow as tf
from hparams import hparams_debug_string
from feeder import Feeder
from models import create_model
from utils import ValueWindow, audio, infolog
from utils import plot
from utils.text_sequence import sequence_to_text
from utils.text_sequence import symbols
from tqdm import tqdm

from hparams import hparams

log = infolog.log


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def add_embedding_stats(summary_writer, embedding_names, paths_to_meta, checkpoint_path):
    # Create tensorboard projector
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    config.model_checkpoint_path = checkpoint_path

    for embedding_name, path_to_meta in zip(embedding_names, paths_to_meta):
        # Initialize config
        embedding = config.embeddings.add()
        # Specifiy the embedding variable and the metadata
        embedding.tensor_name = embedding_name
        embedding.metadata_path = path_to_meta

    # Project the embeddings to space dimensions for visualization
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)


def add_eval_stats(summary_writer, step, linear_loss, mel_loss, stop_token_loss, loss):
    values = [
        tf.Summary.Value(tag='eval_model/eval_stats/eval_mel_loss', simple_value=mel_loss),
        tf.Summary.Value(tag='eval_model/eval_stats/stop_token_loss', simple_value=stop_token_loss),
        tf.Summary.Value(tag='eval_model/eval_stats/eval_loss', simple_value=loss),
    ]
    if linear_loss is not None:
        values.append(tf.Summary.Value(tag='eval_model/eval_stats/eval_linear_loss', simple_value=linear_loss))
    test_summary = tf.Summary(value=values)
    summary_writer.add_summary(test_summary, step)


def model_train_mode(feeder, hparams, global_step, hvd=None):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        model = create_model(hparams)
        model.initialize(inputs=feeder.inputs, input_lengths=feeder.input_lengths,
                         target_feats=feeder.target_feats,
                         targets_length=feeder.target_lengths,
                         targets_stop_token=feeder.stop_tokens,
                         is_training=True, is_validation=False)
        model.add_loss()
        model.add_optimizer(global_step, hvd=hvd)
        stats = model.add_train_stats()
        return model, stats


def model_test_mode(feeder, hparams):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        model = create_model(hparams)
        model.initialize(inputs=feeder.eval_inputs, input_lengths=feeder.eval_input_lengths,
                         target_feats=feeder.eval_target_feats,
                         targets_length=feeder.eval_target_lengths,
                         targets_stop_token=feeder.eval_stop_tokens,
                         is_training=False, is_validation=True)

        model.add_loss()
        return model


def get_alignments(attention_mask):
    alignments_name = ["align"]

    specs = []
    titles = []

    for name, alignment in zip(alignments_name, attention_mask):
        for layer in range(len(alignment)):
            for head in range(alignment.shape[1]):
                specs.append(alignment[layer][head])
                titles.append("{}_layer_{}_head_{}".format(name, layer, head))

    return specs, titles


def init_dir(log_dir):
    save_dir = os.path.join(log_dir, 'cent_pretrained')
    plot_dir = os.path.join(log_dir, 'plots')
    wav_dir = os.path.join(log_dir, 'wavs')
    eval_dir = os.path.join(log_dir, 'eval-dir')
    eval_plot_dir = os.path.join(eval_dir, 'plots')
    eval_wav_dir = os.path.join(eval_dir, 'wavs')
    tensorboard_dir = os.path.join(log_dir, 'centaur_events')
    meta_folder = os.path.join(log_dir, 'metas')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)
    return eval_dir, eval_plot_dir, eval_wav_dir, meta_folder, plot_dir, save_dir, tensorboard_dir, wav_dir


def run_eval(args, eval_dir, eval_model, eval_plot_dir, eval_wav_dir, feeder, hparams, sess,
             step, summary_writer):
    # Run eval and save eval stats
    log('\nRunning evaluation at step {}'.format(step))
    sum_eval_loss = 0.0
    sum_mel_loss = 0.0
    sum_stop_token_loss = 0.0
    sum_linear_loss = 0.0
    count = 0.0
    mel_p = None
    mel_t = None
    t_len = None
    attention_mask_sample = None
    lin_p = None
    lin_t = None
    for _ in tqdm(range(feeder.test_steps)):
        test_eloss, test_mel_loss, test_stop_token_loss, test_linear_loss, mel_p, mel_t, t_len, attention_mask_sample, lin_p, lin_t = sess.run(
            [
                eval_model.loss,
                eval_model.mel_loss,
                eval_model.stop_token_loss,
                eval_model.linear_loss,
                eval_model.post_net_predictions[0],
                eval_model.targets_mel[0],
                eval_model.targets_length[0],
                eval_model.alignments[0],
                eval_model.mag_pred[0],
                eval_model.targets_mag[0],
            ])
        sum_eval_loss += test_eloss
        sum_mel_loss += test_mel_loss
        sum_stop_token_loss += test_stop_token_loss
        sum_linear_loss += test_linear_loss
        count += 1.0
    wav = audio.inv_linear_spectrogram(lin_p.T, hparams)
    audio.save_wav(wav,
                   os.path.join(eval_wav_dir, '{}-eval-linear.wav'.format(step)),
                   sr=hparams.sample_rate)
    if count > 0.0:
        eval_loss = sum_eval_loss / count
        mel_loss = sum_mel_loss / count
        stop_token_loss = sum_stop_token_loss / count
        linear_loss = sum_linear_loss / count
    else:
        eval_loss = sum_eval_loss
        mel_loss = sum_mel_loss
        stop_token_loss = sum_stop_token_loss
        linear_loss = sum_linear_loss
    log('Saving eval log to {}..'.format(eval_dir))
    # Save some log to monitor model improvement on same unseen sequence
    wav = audio.inv_mel_spectrogram(mel_p.T, hparams)
    audio.save_wav(wav, os.path.join(eval_wav_dir, '{}-eval-mel.wav'.format(step)),
                   sr=hparams.sample_rate)
    alignments, alignment_titles = get_alignments(attention_mask_sample)
    for i in range(len(alignments)):
        plot.plot_alignment(alignments[i], os.path.join(eval_plot_dir,
                                                        '{}_{}-eval-align.png'.format(step,
                                                                                      alignment_titles[
                                                                                          i])),
                            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step,
                                                                        eval_loss),
                            max_len=t_len // hparams.reduction_factor)
    plot.plot_spectrogram(mel_p,
                          os.path.join(eval_plot_dir, '{}-eval-mel-spectrogram.png'.format(step)),
                          title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step,
                                                                      eval_loss),
                          target_spectrogram=mel_t,
                          max_len=t_len)
    plot.plot_spectrogram(lin_p, os.path.join(eval_plot_dir,
                                              '{}-eval-linear-spectrogram.png'.format(step)),
                          title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(),
                                                                      step, eval_loss),
                          target_spectrogram=lin_t,
                          max_len=t_len, auto_aspect=True)
    log('Eval loss for global step {}: {:.3f}'.format(step, eval_loss))
    log('Writing eval summary!')
    add_eval_stats(summary_writer, step, linear_loss, mel_loss, stop_token_loss,
                   eval_loss)


def save_current_model(args, checkpoint_path, global_step, hparams, loss, model, plot_dir, saver, sess, step, wav_dir):
    # Save model and current global step
    saver.save(sess, checkpoint_path, global_step=global_step)
    log('\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')
    input_seq, mel_prediction, linear_prediction, attention_mask_sample, targets_mel, target_length, linear_target = sess.run(
        [
            model.inputs[0],
            model.post_net_predictions[0],
            model.mag_pred[0],
            model.alignments[0],
            model.targets_mel[0],
            model.targets_length[0],
            model.targets_mag[0],
        ])
    alignments, alignment_titles = get_alignments(attention_mask_sample)
    # save griffin lim inverted wav for debug (linear -> wav)
    wav = audio.inv_linear_spectrogram(linear_prediction.T, hparams)
    audio.save_wav(wav, os.path.join(wav_dir, '{}-linear.wav'.format(step)),
                   sr=hparams.sample_rate)
    # Save real and predicted linear-spectrogram plot to disk (control purposes)
    plot.plot_spectrogram(linear_prediction,
                          os.path.join(plot_dir, '{}-linear-spectrogram.png'.format(step)),
                          title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(),
                                                                      step, loss),
                          target_spectrogram=linear_target,
                          max_len=target_length, auto_aspect=True)
    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    audio.save_wav(wav, os.path.join(wav_dir, '{}-mel.wav'.format(step)),
                   sr=hparams.sample_rate)
    # save alignment plot to disk (control purposes)
    for i in range(len(alignments)):
        plot.plot_alignment(alignments[i], os.path.join(plot_dir, '{}_{}-align.png'.format(step,
                                                                                           alignment_titles[
                                                                                               i])),
                            title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step,
                                                                        loss),
                            max_len=target_length // hparams.reduction_factor)
    # save real and predicted mel-spectrogram plot to disk (control purposes)
    plot.plot_spectrogram(mel_prediction,
                          os.path.join(plot_dir, '{}-mel-spectrogram.png'.format(step)),
                          title='{}, {}, step={}, loss={:.5f}'.format(args.model, time_string(), step,
                                                                      loss),
                          target_spectrogram=targets_mel,
                          max_len=target_length)
    log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))


def update_character_embedding(char_embedding_meta, save_dir, summary_writer):
    # Get current checkpoint state
    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
    # Update Projector
    log('\nSaving Model Character Embeddings visualization..')
    add_embedding_stats(summary_writer, ["embedding_and_softmax"], [char_embedding_meta],
                        checkpoint_state.model_checkpoint_path)
    log('Centaur Character embeddings have been updated on tensorboard!')


def restore_model(saver, sess, global_step, save_dir, checkpoint_path, reset_global_step=False):
    try:
        checkpoint_state = tf.train.get_checkpoint_state(save_dir)

        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
            saver.restore(sess, checkpoint_state.model_checkpoint_path)

        else:
            log('No model to load at {}'.format(save_dir), slack=True)
            saver.save(sess, checkpoint_path, global_step=global_step)
        if reset_global_step:
            zero_step_assign = tf.assign(global_step, 0)
            sess.run(zero_step_assign)
            log('=' * 50)
            log(' [*] Global step is reset to {}'.format(0))
            log('=' * 50)
    except tf.errors.OutOfRangeError as e:
        log('Cannot restore checkpoint: {}'.format(e), slack=True)


def train(log_dir, args, hparams, use_hvd=False):
    if use_hvd:
        import horovod.tensorflow as hvd
        # Initialize Horovod.
        hvd.init()
    else:
        hvd = None
    eval_dir, eval_plot_dir, eval_wav_dir, meta_folder, plot_dir, save_dir, tensorboard_dir, wav_dir = init_dir(log_dir)

    checkpoint_path = os.path.join(save_dir, 'centaur_model.ckpt')
    input_path = os.path.join(args.base_dir, args.input_dir)

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_path))
    log('Using model: {}'.format(args.model))
    log(hparams_debug_string())

    # Start by setting a seed for repeatability
    tf.set_random_seed(hparams.random_seed)

    # Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder'):
        feeder = Feeder(coord, input_path, hparams)

    # Set up model:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    model, stats = model_train_mode(feeder, hparams, global_step, hvd=hvd)
    eval_model = model_test_mode(feeder, hparams)

    # Embeddings metadata
    char_embedding_meta = os.path.join(meta_folder, 'CharacterEmbeddings.tsv')
    if not os.path.isfile(char_embedding_meta):
        with open(char_embedding_meta, 'w', encoding='utf-8') as f:
            for symbol in symbols:
                if symbol == ' ':
                    symbol = '\\s'  # For visual purposes, swap space with \s

                f.write('{}\n'.format(symbol))

    char_embedding_meta = char_embedding_meta.replace(log_dir, '..')
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=2)

    log('Centaur training set to a maximum of {} steps'.format(args.train_steps))

    # Memory allocation on the GPU as needed
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    if use_hvd:
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Train
    with tf.Session(config=config) as sess:
        try:

            # Init model and load weights
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                restore_model(saver, sess, global_step, save_dir, checkpoint_path, args.reset_global_step)
            else:
                log('Starting new training!', slack=True)
                saver.save(sess, checkpoint_path, global_step=global_step)

            # initializing feeder
            start_step = sess.run(global_step)
            feeder.start_threads(sess, start_step=start_step)
            # Horovod bcast vars across workers
            if use_hvd:
                # Horovod: broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                bcast = hvd.broadcast_global_variables(0)
                bcast.run()
                log('Worker{}: Initialized'.format(hvd.rank()))
            # Training loop
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            while not coord.should_stop() and step < args.train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.train_op])
                if use_hvd:
                    main_process = hvd.rank() == 0
                if main_process:
                    time_window.append(time.time() - start_time)
                    loss_window.append(loss)
                    message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
                        step, time_window.average, loss, loss_window.average)
                    log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

                    if np.isnan(loss) or loss > 100.:
                        log('Loss exploded to {:.5f} at step {}'.format(loss, step))
                        raise Exception('Loss exploded')

                    if step % args.summary_interval == 0:
                        log('\nWriting summary at step {}'.format(step))
                        summary_writer.add_summary(sess.run(stats), step)

                    if step % args.eval_interval == 0:
                        run_eval(args, eval_dir, eval_model, eval_plot_dir, eval_wav_dir, feeder,
                                 hparams, sess, step, summary_writer)

                    if step % args.checkpoint_interval == 0 or step == args.train_steps or step == 300:
                        save_current_model(args, checkpoint_path, global_step, hparams, loss, model,
                                           plot_dir, saver, sess, step, wav_dir)

                    if step % args.embedding_interval == 0 or step == args.train_steps or step == 1:
                        update_character_embedding(char_embedding_meta, save_dir, summary_writer)

            log('Centaur training complete after {} global steps!'.format(args.train_steps), slack=True)
            return save_dir

        except Exception as e:
            log('Exiting due to exception: {}'.format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
    return log_dir, modified_hp


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--name', help='Name of logging directory.')
    parser.add_argument('--model', default='Centaur')
    parser.add_argument('--input_dir', default='../train_data/', help='folder to contain inputs sentences/targets')
    parser.add_argument('--output_dir', default='output', help='folder to contain synthesized mel spectrograms')
    parser.add_argument('--mode', default='synthesis', help='mode for synthesis of centaur after training')
    parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=250,
                        help='Steps between running summary ops')
    parser.add_argument('--embedding_interval', type=int, default=5000,
                        help='Steps between updating embeddings projection visualization')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1500,
                        help='Steps between eval on test data')
    parser.add_argument('--train_steps', type=int, default=1000000,
                        help='total number of centaur training steps')
    parser.add_argument('--reset_global_step', type=bool, default=False,
                        help='Set this to True to do reset global_step')
    parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')
    parser.add_argument('--gpu_assignment', default='hvd', help='Set the gpu the model should run on.(eg:hvd,0,1,2...)')
    args = parser.parse_args(argv[1:])
    use_hvd = args.gpu_assignment == 'hvd'
    if not use_hvd:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_assignment
    log_dir, hparams = prepare_run(args)
    train(log_dir, args, hparams, use_hvd=use_hvd)


if __name__ == '__main__':
    tf.app.run(main=main)
