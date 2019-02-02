import pandas as pd
import numpy as np

OUTER_START_POS = 0
OUTER_SIZE = 11
STATE_SIZE = 47
ACTION_SIZE = 51
STATE_START_POS = OUTER_START_POS + OUTER_SIZE
ACTION_START_POS = STATE_START_POS + STATE_SIZE
NEW_STATE_START_POS = ACTION_START_POS + ACTION_SIZE

NOX_POS = 40
STEAM_TEMP_POS = 48
STEAM_PRES_POS = 49
NEG_PRES_POS = 44
LIM_LOAD_POS = 11
LOAD_POS = 46
EFFI_WEIGHT = 0.8

# Read inv-normlization file
inv_norm = pd.read_csv('/Users/xhr/PycharmProjects/Boiler/Simulator/data/反归一化_new.csv', index_col='name')
inv_norm_min = inv_norm['min'].values  # convert to ndarray
inv_norm_max = inv_norm['max'].values  # convert to ndarray


def get_efficiency(state):
    if len(state.shape) == 1:
        # 主蒸汽流量
        h = state[47] * (inv_norm.loc['主蒸汽流量']['max'] - inv_norm.loc['主蒸汽流量']['min']) + inv_norm.loc['主蒸汽流量']['min']
        # 排烟含氧量
        i = state[39] * (inv_norm.loc['排烟含氧量']['max'] - inv_norm.loc['排烟含氧量']['min']) + inv_norm.loc['排烟含氧量']['min']
        # 引风机入口温度
        j = state[38] * (inv_norm.loc['引风机入口温度']['max'] - inv_norm.loc['引风机入口温度']['min']) + inv_norm.loc['引风机入口温度']['min']
        # 送风机入口温度
        k = state[10] * (inv_norm.loc['送风机入口温度']['max'] - inv_norm.loc['送风机入口温度']['min']) + inv_norm.loc['送风机入口温度']['min']
        # 低位发热量MJ/kg
        l = state[7] * (inv_norm.loc['低位发热量MJ/kg']['max'] - inv_norm.loc['低位发热量MJ/kg']['min']) + inv_norm.loc['低位发热量MJ/kg']['min']
        # 收到基水份%
        m = state[4] * (inv_norm.loc['收到基水份%']['max'] - inv_norm.loc['收到基水份%']['min']) + inv_norm.loc['收到基水份%']['min']
        # 收到基灰份%
        n = state[5] * (inv_norm.loc['收到基灰份%']['max'] - inv_norm.loc['收到基灰份%']['min']) + inv_norm.loc['收到基灰份%']['min']
        # 飞灰%
        p = state[9] * (inv_norm.loc['飞灰%']['max'] - inv_norm.loc['飞灰%']['min']) + inv_norm.loc['飞灰%']['min']
        # 渣%
        q = state[8] * (inv_norm.loc['渣%']['max'] - inv_norm.loc['渣%']['min']) + inv_norm.loc['渣%']['min']
        # 分析基水份%
        analytical_base_moisture = state[1] * (inv_norm.loc['分析基水份%']['max'] - inv_norm.loc['分析基水份%']['min']) + inv_norm.loc['分析基水份%']['min']
        # 分析基挥发分%
        analytical_base_volatile = state[3] * (inv_norm.loc['分析基挥发分%']['max'] - inv_norm.loc['分析基挥发分%']['min']) + inv_norm.loc['分析基挥发分%']['min']
    else:
        # 主蒸汽流量
        h = state[:, 47] * (inv_norm.loc['主蒸汽流量']['max'] - inv_norm.loc['主蒸汽流量']['min']) + inv_norm.loc['主蒸汽流量']['min']
        # 排烟含氧量
        i = state[:, 39] * (inv_norm.loc['排烟含氧量']['max'] - inv_norm.loc['排烟含氧量']['min']) + inv_norm.loc['排烟含氧量']['min']
        # 引风机入口温度
        j = state[:, 38] * (inv_norm.loc['引风机入口温度']['max'] - inv_norm.loc['引风机入口温度']['min']) + inv_norm.loc['引风机入口温度']['min']
        # 送风机入口温度
        k = state[:, 10] * (inv_norm.loc['送风机入口温度']['max'] - inv_norm.loc['送风机入口温度']['min']) + inv_norm.loc['送风机入口温度']['min']
        # 低位发热量MJ/kg
        l = state[:, 7] * (inv_norm.loc['低位发热量MJ/kg']['max'] - inv_norm.loc['低位发热量MJ/kg']['min']) + inv_norm.loc['低位发热量MJ/kg']['min']
        # 收到基水份%
        m = state[:, 4] * (inv_norm.loc['收到基水份%']['max'] - inv_norm.loc['收到基水份%']['min']) + inv_norm.loc['收到基水份%']['min']
        # 收到基灰份%
        n = state[:, 5] * (inv_norm.loc['收到基灰份%']['max'] - inv_norm.loc['收到基灰份%']['min']) + inv_norm.loc['收到基灰份%']['min']
        # 飞灰%
        p = state[:, 9] * (inv_norm.loc['飞灰%']['max'] - inv_norm.loc['飞灰%']['min']) + inv_norm.loc['飞灰%']['min']
        # 渣%
        q = state[:, 8] * (inv_norm.loc['渣%']['max'] - inv_norm.loc['渣%']['min']) + inv_norm.loc['渣%']['min']
        # 分析基水份%
        analytical_base_moisture = state[:, 1] * (inv_norm.loc['分析基水份%']['max'] - inv_norm.loc['分析基水份%']['min']) + inv_norm.loc['分析基水份%']['min']
        # 分析基挥发分%
        analytical_base_volatile = state[:, 3] * (inv_norm.loc['分析基挥发分%']['max'] - inv_norm.loc['分析基挥发分%']['min']) + inv_norm.loc['分析基挥发分%']['min']

    o = (100 - m) / (100 - analytical_base_moisture) * analytical_base_volatile
    l = l * 1000
    u = 10 * q / (100 - q) + 90 * p / (100 - p)
    v = 0.257 * (l - 3.3727 * n * u) / 1000
    w = 0.98 * v
    x = o * 100 / (100 - m - n)
    y = 2.1236 * x ** 0.2319
    z = y * (100 - m - n) / 100
    aa = 21 / (21 - i)
    ab = w + (aa - 1) * v
    ac = 1.24 * ((9 * z + m) / 100 + 1.293 * aa * v * 0.01)
    ad = 5.82 * 2141 ** (-0.38)
    s = ab * 1.38 * (j - k)
    t = ac * 1.51 * (j - k)

    c = (s + t) / l * 100
    # d = 126.36 * r * ab / l * 100
    e = 337.27 * n * u / l
    f = ad * (1095.4 / h)
    g = n * (10 * (800 - k) * 0.96 / (100 - q) + 90 * (j - k) * 0.82 / (100 - p)) / l

    effi = 100 - c - e - f - g
    norm_effi = (effi - inv_norm.loc['1号机组锅炉效率']['min']) / (inv_norm.loc['1号机组锅炉效率']['max'] - inv_norm.loc['1号机组锅炉效率']['min'])

    return norm_effi


def get_emission(state):
    if len(state.shape) == 1:
        return state[NOX_POS]
    else:
        return state[:, NOX_POS]


def get_steam_temp(state):
    if len(state.shape) == 1:
        return state[STEAM_TEMP_POS]
    else:
        return state[:, STEAM_TEMP_POS]


def get_given_steam_pres(state, load):
    if len(state.shape) == 1:
        if load >= 560:
            given_steam_pres = 24.0
        else:
            given_steam_pres =  0.036072 * load + 3.89199
    else:
        given_steam_pres = np.ones([load.shape[0]]) * 24
        given_steam_pres[load < 560] = 0.036072 * load[load < 560] + 3.89199
    return given_steam_pres



def get_steam_pres(state):
    if len(state.shape) == 1:
        return state[STEAM_PRES_POS]
    else:
        return state[:, STEAM_PRES_POS]


def get_neg_pres(state):
    if len(state.shape) == 1:
        return state[NEG_PRES_POS]
    else:
        return state[:, NEG_PRES_POS]


def get_lim_load(state):
    if len(state.shape) == 1:
        return state[LIM_LOAD_POS]
    else:
        return state[:, LIM_LOAD_POS]


def get_load(state):
    if len(state.shape) == 1:
        return state[LOAD_POS]
    else:
        return state[:, LOAD_POS]


def compute_reward(state):
    # coals = get_coals(action)
    efficiency = get_efficiency(state)
    emission = get_emission(state)
    # print('effi', EFFI_WEIGHT * efficiency - (1-EFFI_WEIGHT) * emission)
    reward = EFFI_WEIGHT * efficiency - (1-EFFI_WEIGHT) * emission
    if np.mean(reward) > 1:
        print(reward, efficiency, emission)
    return 10*(EFFI_WEIGHT * efficiency - (1-EFFI_WEIGHT) * emission)


# def compute_cost(state):
#     lim_load = get_lim_load(state) * (inv_norm.loc['lim_load']['max'] - inv_norm.loc['lim_load']['min']) + inv_norm.loc['lim_load']['min']
#     load = get_load(state) * (inv_norm.loc['#1机组锅炉负荷']['max'] - inv_norm.loc['#1机组锅炉负荷']['min']) + inv_norm.loc['#1机组锅炉负荷']['min']
#     steam_temp = get_steam_temp(state) * (inv_norm.loc['锅炉主蒸汽温度']['max'] - inv_norm.loc['锅炉主蒸汽温度']['min']) + inv_norm.loc['锅炉主蒸汽温度']['min']
#     given_steam_pres = get_given_steam_pres(load)
#     steam_pres = get_steam_pres(state) * (inv_norm.loc['主蒸汽压力']['max'] - inv_norm.loc['主蒸汽压力']['min']) + inv_norm.loc['主蒸汽压力']['min']
#     # neg_pressure = get_neg_pres(state) * (inv_norm.loc['炉膛负压']['max'] - inv_norm.loc['炉膛负压']['min']) + inv_norm.loc['炉膛负压']['min']
#
#     # cost 1, 负荷:lim_load ~ limload+25
#     if len(state.shape) == 1:
#         if load - lim_load > 25 or load < lim_load:
#             cost_load = 1
#         else:
#             cost_load = 0
#     else:
#         cost_load = np.zeros([len(state), 1])
#         cost_load[(load-lim_load > 25) | (load-lim_load < 0)] = 1
#
#     # else:
#     #     if diff < 0.01:
#     #         return 0
#     #     elif diff < 0.1:
#     #         return 0.2
#     #     elif diff < 0.5:
#     #         return 0.5
#     #     else:
#     #         return 1
#
#     # cost 2, 主蒸汽温度:569-10 ~ 569+5
#     if len(state.shape) == 1:
#         if steam_temp > 569+5 or steam_temp < 569-10:
#             cost_steam_temp = 1
#         else:
#             cost_steam_temp = 0
#     else:
#         cost_steam_temp = np.zeros([len(state), 1])
#         cost_steam_temp[(steam_temp > 569+5) | (steam_temp < 569-10)] = 1
#
#     # cost 3, 主蒸汽压力:given_pres-0.5 ~ given_pres+0.5
#     if len(state.shape) == 1:
#         if steam_pres > given_steam_pres+0.5 or steam_pres < given_steam_pres-0.5:
#             cost_steam_pres = 1
#         else:
#             cost_steam_pres = 0
#     else:
#         cost_steam_pres = np.zeros([len(state), 1])
#         cost_steam_pres[(steam_pres > given_steam_pres+0.5) & (steam_pres < given_steam_pres-0.5)] = 1
#
#     return 1/3*cost_load + 1/3*cost_steam_temp + 1/3*cost_steam_pres

def compute_cost(state):
    lim_load = get_lim_load(state) * (inv_norm.loc['lim_load']['max'] - inv_norm.loc['lim_load']['min']) + inv_norm.loc['lim_load']['min']
    load = get_load(state) * (inv_norm.loc['#1机组锅炉负荷']['max'] - inv_norm.loc['#1机组锅炉负荷']['min']) + inv_norm.loc['#1机组锅炉负荷']['min']
    steam_temp = get_steam_temp(state) * (inv_norm.loc['锅炉主蒸汽温度']['max'] - inv_norm.loc['锅炉主蒸汽温度']['min']) + inv_norm.loc['锅炉主蒸汽温度']['min']
    given_steam_pres = get_given_steam_pres(state, load)
    steam_pres = get_steam_pres(state) * (inv_norm.loc['主蒸汽压力']['max'] - inv_norm.loc['主蒸汽压力']['min']) + inv_norm.loc['主蒸汽压力']['min']
    # neg_pressure = get_neg_pres(state) * (inv_norm.loc['炉膛负压']['max'] - inv_norm.loc['炉膛负压']['min']) + inv_norm.loc['炉膛负压']['min']

    # cost 1, 负荷:lim_load ~ limload+25
    if len(state.shape) == 1:
        if load - lim_load > 25:
            cost_load = np.abs(load - lim_load - 25) / 10
        elif load < lim_load:
            cost_load = 1
        else:
            cost_load = 0
    else:
        cost_load = np.zeros([len(state)])
        cost_load[load-lim_load > 25] = np.abs(load - lim_load - 25)[load-lim_load > 25] / 10
        cost_load[load-lim_load < 0] = 1


    # else:
    #     if diff < 0.01:
    #         return 0
    #     elif diff < 0.1:
    #         return 0.2
    #     elif diff < 0.5:
    #         return 0.5
    #     else:
    #         return 1

    # cost 2, 主蒸汽温度:569-10 ~ 569+5
    if len(state.shape) == 1:
        if steam_temp > 569+10:
            cost_steam_temp = np.abs(steam_temp - 569-10) / 10
        elif steam_temp < 569-10:
            cost_steam_temp = np.abs(steam_temp - 569+10) / 10
        else:
            cost_steam_temp = 0
    else:
        cost_steam_temp = np.zeros([len(state)])
        cost_steam_temp[steam_temp > 569+10] = np.abs(steam_temp - 569-10)[steam_temp > 569+10] / 10
        cost_steam_temp[steam_temp < 569-10] = np.abs(steam_temp - 569+10)[steam_temp < 569-10] / 10

    # cost 3, 主蒸汽压力:given_pres-0.5 ~ given_pres+0.5
    if len(state.shape) == 1:
        if steam_pres > given_steam_pres+1:
            cost_steam_pres = np.abs(steam_pres - given_steam_pres-1) / 5
        elif steam_pres < given_steam_pres-1:
            cost_steam_pres = np.abs(steam_pres - given_steam_pres+1) / 5
        else:
            cost_steam_pres = 0
    else:
        cost_steam_pres = np.zeros([len(state)])
        cost_steam_pres[steam_pres > given_steam_pres+1] = np.abs(steam_pres - given_steam_pres-1)[steam_pres > given_steam_pres+1] / 5
        cost_steam_pres[steam_pres < given_steam_pres-1] = np.abs(steam_pres - given_steam_pres+1)[steam_pres < given_steam_pres-1] / 5


    return 1/3*cost_load + 1/3*cost_steam_temp + 1/3*cost_steam_pres


def compute_done(state):
    return False


def convert_to_tuple(batch):
    outer = batch[:, OUTER_START_POS: OUTER_START_POS + OUTER_SIZE]
    state_with_outer = batch[:, OUTER_START_POS: STATE_START_POS + STATE_SIZE]
    action = batch[:, ACTION_START_POS: ACTION_START_POS + ACTION_SIZE]
    new_state = batch[:, NEW_STATE_START_POS: NEW_STATE_START_POS + STATE_SIZE]
    new_state_with_outer = np.concatenate([outer, new_state], axis=1)
    done = batch[:, -1]
    return (state_with_outer, action, new_state_with_outer, done)



def restrictive_action(action, episode):
    action_histogram = np.array(pd.read_csv('../Simulator/data/action_histogram.csv', header=None)).astype('float')
    threshold = action_histogram[:, -1]
    noise_weight = 1

    if episode % 100 == 0 and episode > 0:
        noise_weight *= 0.99

    # print('value'+str(self.df[np.arange(len(x[:-1])).astype('int'), (x[:-1] * 20).astype('int')]))
    action_distri = np.array(action_histogram[np.arange(len(action)).astype('int'), (action * 20).astype('int')[:]] > threshold).astype('int')
    for i in range(100):
        if len(np.where(action_distri == 0)[0]) > 0:
            unsatisfied_index = np.where(action_distri == 0)[0]
            # print(f'unsatisfied index {unsatisfied_index}')
            #       f'unsatisfied action {actions[unsatisfied_index]})')
            random_noise = np.random.normal(np.zeros(len(unsatisfied_index)),
                                            (0.1 + 0.01 * i) * np.ones(len(unsatisfied_index)),
                                            len(unsatisfied_index))
            action[unsatisfied_index] += random_noise * noise_weight
            # print(f'fixed actions {actions[unsatisfied_index]}'
        else:
            # print(32 + np.where(action_distri[32:] == 0)[0])
            # print('find action within '+str(i)+' times')
            break

    # if len(np.where(action_distri == 0)[0]) > 0:
    #     unsatisfied_index = np.where(action_distri == 0)[0]
    #     print(f'Break! dissatisfied actions is {len(np.where(action_distri == 0)[0])}, '
    #           f'index: {unsatisfied_index}, value: {action[unsatisfied_index]},')
    #     break

    return action
