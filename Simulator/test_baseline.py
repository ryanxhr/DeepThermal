import tensorflow as tf
import os
import numpy as np
import pandas as pd
from Simulator.baseline_model import BaseLineModel
from Simulator.baseline_main import FLAGS
from Simulator.data_model import BoilerDataSet


def root_mean_squared_error(labels, preds):
    total_size = np.size(labels)
    return np.sqrt(np.sum(np.square(labels - preds)) / total_size)


def mean_absolute_error(labels, preds):
    total_size = np.size(labels)
    return np.sum(np.abs(labels - preds)) / total_size


if __name__ == '__main__':
    # use specific gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # load hyperparameters
    session = tf.Session(config=tf_config)
    print(FLAGS)

    # model construction
    tf.reset_default_graph()
    boiler_dataset = BoilerDataSet(num_steps=FLAGS.num_steps)
    baseline_model = BaseLineModel(FLAGS)

    # read data from test set
    test_data = pd.read_csv(os.path.join("data", "sim_test.csv"), index_col='date')
    if FLAGS.model == 'dnn':
        test_X = np.array([test_data.iloc[i: i + 1].values
                           for i in range(len(test_data) - 1)])
        test_y = np.array([test_data.ix[i + 1, 'A磨煤机料位':'1号机组下部水冷壁出口平均壁温'].values
                           for i in range(len(test_data) - 1)])
        test_X = test_X.reshape([-1, FLAGS.input_size])
        test_y = test_y.reshape([-1, FLAGS.output_size])
    else:
        test_X = np.array([test_data.iloc[i: i + FLAGS.num_steps].values
                           for i in range(len(test_data) - FLAGS.num_steps)])
        test_y = np.array([test_data.ix[i + FLAGS.num_steps, 'A磨煤机料位':'1号机组下部水冷壁出口平均壁温'].values
                           for i in range(len(test_data) - FLAGS.num_steps)])

    # Read inv-normlization file
    inv_norm = pd.read_csv(os.path.join("data", "反归一化_new.csv"), index_col='name')['A磨煤机料位':'1号机组下部水冷壁出口平均壁温']
    inv_norm_min = inv_norm['min'].values  # convert to ndarray
    inv_norm_max = inv_norm['max'].values  # convert to ndarray
    num_test = len(test_data)
    print('test samples: {0}'.format(num_test))

    # path
    model_name = "baseline_" + FLAGS.model
    model_path = './logs/{}-{}-{}-{}-{:.2f}-{:.4f}-{:.2f}-{:.5f}/'.format(
        model_name, FLAGS.num_layers, FLAGS.num_units, FLAGS.num_steps,
        FLAGS.keep_prob, FLAGS.learning_rate, FLAGS.learning_rate_decay, FLAGS.l2_weight)
    model_path += 'saved_models/final_model.ckpt'


    # test params
    test_rmses = []
    test_maes = []

    # restore model
    print("Starting loading model...")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model successfully restored from file: %s" % model_path)

        # test
        test_loss = 0

        for test_batch_X, test_batch_y in boiler_dataset.generate_one_epoch(test_X, test_y, FLAGS.batch_size):
            test_data_feed = {
                baseline_model.keep_prob: 1.0,
                baseline_model.inputs: test_batch_X,
                baseline_model.targets: test_batch_y,
            }
            # re-scale predicted labels
            batch_preds = sess.run(baseline_model.pred, test_data_feed)
            batch_preds = inv_norm_min + batch_preds * (inv_norm_max - inv_norm_min)  # broadcast by axis 0

            # re-scale real labels
            batch_labels = inv_norm_min + test_batch_y * (inv_norm_max - inv_norm_min)  # broadcast by axis 0
            test_rmses.append(root_mean_squared_error(
                batch_labels, batch_preds))
            test_maes.append(mean_absolute_error(batch_labels, batch_preds))

    test_rmses = np.asarray(test_rmses)
    test_maes = np.asarray(test_maes)

    print('===============METRIC===============')
    print('rmse = {:.6f}'.format(test_rmses.mean()))
    print('mae = {:.6f}'.format(test_maes.mean()))