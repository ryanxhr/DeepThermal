

def pre_train_actor_network(agent, train_data, epochs=50, load_model=False):
    """
            train critic network of agent
            data : from train_data_path (eg. origin data)
    """
    input_config = InputConfig_RL()
    replay_buffer = agent.replay_buffer
    replay_buffer.read_from_csv(train_data)

    step = 0
    del_list = list(range(12, 24)) + list(range(36, 48))
    for epoch in range(epochs):
        while 1:
            if replay_buffer.use_nums > replay_buffer.count():
                replay_buffer.read_from_csv(train_data)
                step = 0
                break

            mini_batch = replay_buffer.get_batch(batch_size=input_config.batch_size)
            step += 1
            mini_batch = np.bmat(list(map(list, mini_batch))).A.flatten().reshape(-1, DONE_END)
            state_batch = mini_batch[:, :OUTER_END]
            action_batch = mini_batch[:, OUTER_END + 32:ACTION_END]
            for i in range(12):
                action_batch[:, i] = (action_batch[:, i] + action_batch[:, 23 - i]) / 2
                action_batch[:, 24 + i] = (action_batch[:, 24 + i] + action_batch[:, 47 - i]) / 2
            action_batch = np.delete(action_batch, del_list, axis=1)
            limit_batch = mini_batch[:, ACTION_END:LIMIT_LOAD_END]
            state_limit_batch = np.concatenate((state_batch, limit_batch), axis=1)

            mse, _ = agent.train_actor(state=state_limit_batch, action=action_batch)

            # display
            if step % 100 == 0:
                print(replay_buffer.use_nums)
                print('-----------------pretrain actor network-----------------')
                print('epoch = {} step = {} mse = {:.6f}'.format(epoch, step, mse))