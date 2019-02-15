import tensorflow as tf
from RL.env import SimulatorEnvironment
from RL.actor_network import ActorNetwork
from RL.primal_dual_ddpg import PrimalDualDDPG

from RL.train_primal_dual import input_config
from Simulator.simrnn_model import RNNSimulatorModel
from Simulator.simrnn_main import FLAGS, cell_config
import pandas as pd
import numpy as np


config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

# build agent graph
tf.reset_default_graph()

actor_graph = tf.Graph()
actor_sess = tf.Session(config=config, graph=actor_graph)

with actor_graph.as_default():
    agent = PrimalDualDDPG(sess=actor_sess, input_config=input_config, is_batch_norm=False, load_model=False)
    actor_saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("./models")
    if checkpoint and checkpoint.model_checkpoint_path:
        actor_saver.restore(actor_sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")


# build environment graph
env_graph = tf.Graph()
env_sess = tf.Session(config=config, graph=env_graph)
with env_graph.as_default():
    # model construction
    rnn_model = RNNSimulatorModel(cell_config(), FLAGS)
    # path
    model_name = "sim_rnn"
    model_path = '../Simulator/logs/{}-{}-{}-{}-{}-{:.2f}-{:.4f}-{:.2f}-{:.5f}/'.format(
        model_name, cell_config.num_units[0], cell_config.num_units[1], cell_config.num_units[2],
        FLAGS.num_steps, FLAGS.keep_prob, FLAGS.learning_rate, FLAGS.learning_rate_decay, FLAGS.l2_weight)
    model_path += 'saved_models/final_model.ckpt'

    saver = tf.train.Saver()
    saver.restore(env_sess, model_path)
    print("Model successfully restored from file: %s" % model_path)

    env_sess.run(tf.global_variables_initializer())


file_test = pd.read_csv('../Simulator/data/sim_train.csv', index_col='date')
df_state_data = pd.DataFrame(columns=file_test.columns[:58])  # 生成空的pandas表
df_action_data = pd.DataFrame(columns=file_test.columns[58:109])
df_next_state_opt = pd.DataFrame(columns=file_test.columns[11:58])
df_action_opt = pd.DataFrame(columns=file_test.columns[58:109])

for i in range(10, (len(file_test)-1)//10):
    previous_states = file_test.ix[i-9:i, :]
    state = file_test.ix[i, '分析基水份%':'1号机组下部水冷壁出口平均壁温']
    limit_load = file_test.ix[i, 'lim_load']
    coal_effiencicy = file_test.ix[i, '1号机组锅炉效率']
    action = file_test.ix[i, 'A给煤机给煤量反馈':'给水流量']
    next_state = file_test.ix[i+1, 'A磨煤机料位':'1号机组下部水冷壁出口平均壁温']

    action_opt = agent.actor_network.action(state.values)
    input = np.concatenate([state.values, action_opt])
    input = input.reshape(1, len(input))
    all_input = np.concatenate([previous_states.values, input], axis=0)
    all_input = all_input.reshape(1, 10, 109)

    test_data_feed = {
        rnn_model.keep_prob: 1.0,
        rnn_model.inputs: all_input,
    }
    next_state_opt = env_sess.run(rnn_model.pred, test_data_feed)

    df_state_data.loc[i-10] = state
    df_action_data.loc[i-10] = action
    df_next_state_opt.loc[i-10] = next_state_opt[0]
    df_action_opt.loc[i-10] = action_opt
    print(i)

df_state_data.to_csv('results/test_actor/test_result_state_data.csv')
df_action_data.to_csv('results/test_actor/test_result_action_data.csv')
df_next_state_opt.to_csv('results/test_actor/test_result_next_state_opt.csv')
df_action_opt.to_csv('results/test_actor/test_result_action_opt.csv')
