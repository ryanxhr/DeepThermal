import os
import sys
sys.path.append('../')
from RL.primal_dual_ddpg import *
from RL.env import *



MAX_EPISODES = 30000
MAX_EP_STEPS = 10
# TEST = 10
SIM_REAL_RATIO = 1



class input_config():
    batch_size = 32
    init_dual_lambda = 1
    state_dim = 58
    action_dim = 51
    clip_norm = 5.
    train_display_iter = 200
    model_save_path = './models/'
    # model_name = "sim_ddpg"
    # logdir = './logs/{}-{}-{}-{:.2f}/'.format(
    #     model_name, MAX_EP_STEPS, SIM_REAL_RATIO, init_dual_lambda)
    # log_path = logdir + 'saved_models/'
    log_path = "logs/nonpre_nonexp_" + str(SIM_REAL_RATIO) + "_pdddpg_summary"
    save_iter = 500
    log_iter = 100


def pre_train_actor_network(agent, epochs=3):
    replay_buffer = agent.replay_buffer

    for epoch in range(epochs):
        step = 0
        while step < 1000:
            minibatch = replay_buffer.get_real_batch(batch_size=input_config.batch_size)
            step += 1
            state_batch, action_batch, _, _ = convert_to_tuple(minibatch)

            _, mse = agent.actor_network.pretrain(state=state_batch, label=action_batch)

        # display
        if epoch % 1 == 0:
            print('-----------------pre-train actor network-----------------')
            print('epoch = {} mse = {:.4f}'.format(epoch, mse))


def pre_train_reward_critic_network(agent, epochs=3):
    replay_buffer = agent.replay_buffer
    for train_times in range(epochs):
        step = 0
        while step < 1000:
            minibatch = replay_buffer.get_real_batch(batch_size=input_config.batch_size)
            step += 1
            state_batch, action_batch, next_state_batch, _ = convert_to_tuple(minibatch)
            reward_batch = compute_reward(state_batch)

            y_batch = []
            target_action = agent.actor_network.target_actions(next_state_batch)
            target_value = agent.reward_critic_network.target_reward(next_state_batch, target_action)

            for i in range(len(minibatch)):
                y_batch.append(reward_batch[i] + agent.gamma * target_value[i])

            # update critic network
            reward_critic_loss = agent.reward_critic_network.pretrain(y_batch, state_batch, action_batch)

        # display
        if train_times % 1 == 0:
            print('-----------------pre-train reward critic network-----------------')
            print("reward_critic: loss:{:.3f}".format(reward_critic_loss))


def pre_train_cost_critic_network(agent, epochs=3):
    replay_buffer = agent.replay_buffer
    step = 0
    for train_times in range(epochs):
        step = 0
        while step < 1000:
            minibatch = replay_buffer.get_real_batch(batch_size=input_config.batch_size)
            step += 1
            state_batch, action_batch, next_state_batch, _ = convert_to_tuple(minibatch)
            cost_batch = compute_cost(state_batch)

            z_batch = []
            target_action = agent.actor_network.target_actions(next_state_batch)
            target_value = agent.cost_critic_network.target_cost(next_state_batch, target_action)

            for i in range(len(minibatch)):
                z_batch.append(cost_batch[i] + agent.gamma * target_value[i])

            # update critic network
            cost_critic_loss = agent.cost_critic_network.pretrain(z_batch, state_batch, action_batch)

        # display
        if train_times % 1 == 0:
            print('-----------------pre-train cost critic network-----------------')
            print("reward_critic: loss:{:.3f}".format(cost_critic_loss))


def main():
    # Set up summary writer
    summary_writer = tf.summary.FileWriter(input_config.log_path)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    # build agent graph
    tf.reset_default_graph()
    agent_graph = tf.Graph()
    agent_sess = tf.Session(config=config, graph=agent_graph)
    with agent_graph.as_default():
        agent = PrimalDualDDPG(sess=agent_sess, input_config=input_config, is_batch_norm=False, summ_writer=summary_writer)
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

    # build environment graph
    env_graph = tf.Graph()
    env_sess = tf.Session(config=config, graph=env_graph)
    with env_graph.as_default():
        env = SimulatorEnvironment(sess=env_sess)

    # pre_train
    # pre_train_actor_network(agent=agent, epochs=1)
    # pre_train_reward_critic_network(agent=agent, epochs=1)
    # pre_train_cost_critic_network(agent=agent, epochs=1)
    # agent.actor_network.update_target()
    # agent.reward_critic_network.update_target()
    # agent.cost_critic_network.update_target()

    for episode in range(MAX_EPISODES):
        dual_variable = input_config.init_dual_lambda
        ep_reward = 0
        ep_cost = 0
        state = env.reset()

        for step in range(MAX_EP_STEPS):
            # action = restrictive_action(agent.action(state), episode)
            action = agent.noise_action(state, episode)
            next_state, reward, cost, done = env.step(action)
            ep_reward += reward
            ep_cost += cost
            agent.perceive(state, action, reward, cost, next_state, done, mix_ratio=SIM_REAL_RATIO)
            dual_variable = agent.get_dual_lambda()
            state = next_state
        summary = tf.Summary()
        summary.value.add(tag='Steps_sum_Reward', simple_value=float(ep_reward/MAX_EP_STEPS))
        summary.value.add(tag='Steps_sum_Cost', simple_value=float(ep_cost/MAX_EP_STEPS))
        summary.value.add(tag='Dual_variable', simple_value=float(dual_variable))
        summary_writer.add_summary(summary, episode)

        summary_writer.flush()

        print('Episode:{} | Reward: {:.2f} | Cost: {:.2f}'.format(episode, ep_reward/MAX_EP_STEPS, ep_cost/MAX_EP_STEPS))

        if episode % 100 == 0 and episode >= 100:
            agent.save_model()

    print("-------------save model--------------------")
    agent.save_model()


if __name__ == '__main__':
    main()