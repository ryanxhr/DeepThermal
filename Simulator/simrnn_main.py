import os
import numpy as np
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
sys.path.append('../')

from Simulator.simrnn_model import RNNSimulatorModel
from Simulator.data_model import BoilerDataSet

flags = tf.app.flags
# Data and model checkpcheckpointsoints directories
flags.DEFINE_integer("display_iter", 200, "display_iter")
flags.DEFINE_integer("save_log_iter", 100, "save_log_iter")
# Model params
flags.DEFINE_integer("input_size", 109, "Input size")  # external_input + state + action
flags.DEFINE_integer("output_size", 47, "Output size")  # state size
# Optimization
flags.DEFINE_integer("num_steps", 10, "Num of steps")
flags.DEFINE_integer("batch_size", 1, "The size of batch")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches")
flags.DEFINE_float("grad_clip", 5., "Clip gradients at this value")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.95, "Decay rate of learning rate. [0.99]")
flags.DEFINE_float("keep_prob", 1, "Keep probability of input data and dropout layer. [0.8]")
flags.DEFINE_float("l2_weight", 0.0, "weight of l2 loss")

FLAGS = flags.FLAGS


class cell_config(object):
    """ Simulator Cell config """
    # list, [coaler_num_units, burner_num_units, steamer_num_units]
    num_units = [128, 64, 64]

    # data is [external_input, state(coaler, burner, steamer), action(coaler, burner, steamer)]
    external_state_pos = 0
    external_state_size = 11
    coaler_state_pos = external_state_pos + external_state_size
    coaler_state_size = 25
    burner_state_pos = coaler_state_pos + coaler_state_size
    burner_state_size = 7
    steamer_state_pos = burner_state_pos + burner_state_size
    steamer_state_size = 15
    coaler_action_pos = steamer_state_pos + steamer_state_size
    coaler_action_size = 31
    burner_action_pos = coaler_action_pos + coaler_action_size
    burner_action_size = 15
    steamer_action_pos = burner_action_pos + burner_action_size
    steamer_action_size = 5


pp = pprint.PrettyPrinter()

if not os.path.exists("logs"):
    os.mkdir("logs")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def main(_):
    np.random.seed(2019)

    pp.pprint(flags.FLAGS.__flags)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    # read data
    boiler_dataset = BoilerDataSet(num_steps=FLAGS.num_steps)
    train_X = boiler_dataset.train_X
    train_y = boiler_dataset.train_y
    val_X = boiler_dataset.val_X
    val_y = boiler_dataset.val_y
    # print dataset info
    num_train = len(train_X)
    num_valid = len(val_X)
    print('train samples: {0}'.format(num_train))
    print('eval samples: {0}'.format(num_valid))

    # model construction
    tf.reset_default_graph()
    rnn_model = RNNSimulatorModel(cell_config(), FLAGS)

    # print trainable params
    for i in tf.trainable_variables():
        print(i)
    # count the parameters in our model
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print('total parameters: {}'.format(total_parameters))

    # path for log saving
    model_name = "sim_rnn"
    logdir = './logs/{}-{}-{}-{}-{}-{:.2f}-{:.4f}-{:.2f}-{:.5f}/'.format(
        model_name, cell_config.num_units[0], cell_config.num_units[1], cell_config.num_units[2],
        FLAGS.num_steps, FLAGS.keep_prob, FLAGS.learning_rate, FLAGS.learning_rate_decay, FLAGS.l2_weight)
    model_dir = logdir + 'saved_models/'

    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    results_dir = logdir + 'results/'

    with tf.Session(config=run_config) as sess:
        summary_writer = tf.summary.FileWriter(logdir)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        iter = 0
        valid_losses = [np.inf]

        for i in range(FLAGS.max_epoch):
            print('----------epoch {}-----------'.format(i))
            # learning_rate = FLAGS.learning_rate
            learning_rate = FLAGS.learning_rate * (
                FLAGS.learning_rate_decay ** i
            )

            for batch_X, batch_y in boiler_dataset.generate_one_epoch(train_X, train_y, FLAGS.batch_size):
                iter += 1
                train_data_feed = {
                    rnn_model.learning_rate: learning_rate,
                    rnn_model.keep_prob: FLAGS.keep_prob,
                    rnn_model.inputs: batch_X,
                    rnn_model.targets: batch_y,
                }
                train_loss, _, merged_summ = sess.run(
                    [rnn_model.loss, rnn_model.train_opt, rnn_model.merged_summ], train_data_feed)
                if iter % FLAGS.save_log_iter == 0:
                    summary_writer.add_summary(merged_summ, iter)
                if iter % FLAGS.display_iter == 0:
                    valid_loss = 0
                    for val_batch_X, val_batch_y in boiler_dataset.generate_one_epoch(val_X, val_y, FLAGS.batch_size):
                        val_data_feed = {
                            rnn_model.keep_prob: 1.0,
                            rnn_model.inputs: val_batch_X,
                            rnn_model.targets: val_batch_y,
                        }
                        batch_loss = sess.run(rnn_model.loss, val_data_feed)
                        valid_loss += batch_loss
                    num_batches = int(len(val_X)) // FLAGS.batch_size
                    valid_loss /= num_batches
                    valid_losses.append(valid_loss)
                    valid_loss_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_loss)])
                    summary_writer.add_summary(valid_loss_sum, iter)

                    if valid_loss < min(valid_losses[:-1]):
                        print('iter {}\tvalid_loss = {:.6f}\tmodel saved!!'.format(
                            iter, valid_loss))
                        saver.save(sess, model_dir +
                                   'model_{}.ckpt'.format(iter))
                        saver.save(sess, model_dir + 'final_model.ckpt')
                    else:
                        print('iter {}\tvalid_loss = {:.6f}\t'.format(
                            iter, valid_loss))

    print('stop training !!!')


if __name__ == '__main__':
    tf.app.run()