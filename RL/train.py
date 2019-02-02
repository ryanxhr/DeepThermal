from RL.primal_dual_ddpg import *
from RL.env import *

EPISODES = 100000
TIME_LIMIT = 30
TEST = 10

flags = tf.app.flags
# Data and model checkpcheckpointsoints directories
flags.DEFINE_integer("display_iter", 200, "display_iter")
flags.DEFINE_integer("save_log_iter", 100, "save_log_iter")
# Model params
flags.DEFINE_integer("input_size", 109, "Input size")  # external_input + state + action
flags.DEFINE_integer("output_size", 47, "Output size")  # state size
# Optimization
flags.DEFINE_integer("num_steps", 1, "Num of steps")
flags.DEFINE_integer("batch_size", 256, "The size of batch")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches")
flags.DEFINE_float("grad_clip", 5., "Clip gradients at this value")
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.95, "Decay rate of learning rate. [0.99]")
flags.DEFINE_float("keep_prob", 1, "Keep probability of input data and dropout layer. [0.8]")
flags.DEFINE_float("l2_weight", 0.0, "weight of l2 loss")

FLAGS = flags.FLAGS


def main():
    env = SimulatorEnvironment(sess=env_sess, model_path=env_path, FLAGS=FLAGS)
    agent = PrimalDualDDPG()

    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in range(TIME_LIMIT):
            action = agent.noise_action(state)
            next_state, reward, cost, done = env.step(action)
            agent.perceive(state, action, reward, cost, next_state, done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    #env.render()
					action = agent.action(state) # direct action for test
                    # state, reward, done, _ = env.step(action)
					total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
        print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
    env.monitor.close()

if __name__ == '__main__':
    main()