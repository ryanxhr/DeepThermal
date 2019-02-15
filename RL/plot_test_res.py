# -*- coding: utf-8 -*-
"""
Created on Thu May 24 17:47:40 2018

test_result画图

@author: xuhaoran
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
# from keras.models import load_model
import matplotlib as mpl


def sub_plot_fuc(row, col, plot_num, name, true, opt, color_true, color_opt, legend_fontsize=10, title_fontsize=10):
    plt.subplot(row, col, plot_num)
    plt.plot(opt, color_opt, label='opt')
    plt.plot(true, color_true, label='real')
    plt.legend(fontsize=legend_fontsize)  # 显示图例
    # plt.ylabel('mg/m^3', fontsize=20)
    plt.title(name, fontsize=title_fontsize)


# action
#给煤量: 6个
def action_plot(action_name, action_data, action_opt):
    figure = plt.figure(figsize=(8, 6))
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    index = 1
    for name in action_name[1:7]:
        sub_plot_fuc(row=2, col=3, plot_num=index, name=name, true=action_data[name], opt=action_opt[name],\
                     color_true='thistle', color_opt='royalblue', legend_fontsize=8, title_fontsize=13)
        index += 1


    #磨煤机: 26个
    figure = plt.figure(figsize=(10, 8))
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    index = 1
    for name in action_name[7:32]:
        sub_plot_fuc(row=5, col=5, plot_num=index, name=name, true=action_data[name], opt=action_opt[name],\
                     color_true='thistle', color_opt='royalblue', legend_fontsize=8, title_fontsize=10)
        index += 1


    #CF挡板: 48个
    figure = plt.figure(figsize=(10, 8))
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    index = 1
    for name in action_name[32:44]:
        sub_plot_fuc(row=3, col=4, plot_num=index, name=name, true=action_data[name], opt=action_opt[name],\
                     color_true='thistle', color_opt='royalblue', legend_fontsize=6, title_fontsize=8)
        index += 1


    #其他: 16个
    figure = plt.figure(figsize=(10, 8))
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    index = 1
    for name in action_name[44:]:
        sub_plot_fuc(row=2, col=4, plot_num=index, name=name, true=action_data[name], opt=action_opt[name],\
                     color_true='thistle', color_opt='royalblue', legend_fontsize=8, title_fontsize=11)
        index += 1

    plt.show()
    # plt.savefig('result/figure/figure_action_64.png', bbox_inches='tight')# 图片紧凑显示



# # state
def state_plot(state_data, next_state_opt):
    figure_state = plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    opt_effi = next_state_opt['1号机组锅炉效率'][:-1]
    init_effi = state_data['1号机组锅炉效率'][0]
    opt_effi = np.concatenate([[init_effi], opt_effi])
    sub_plot_fuc(row=5, col=1, plot_num=1, name='Efficiency', true=state_data['1号机组锅炉效率'], opt=opt_effi,
                 color_true='thistle', color_opt='red')
    plt.xlabel('time (min)', fontsize=10)
    plt.ylabel('%', fontsize=10)

    opt_nox = next_state_opt['SCR反应器入口Nox含量'][:-1]
    init_nox = state_data['SCR反应器入口Nox含量'][0]
    opt_nox = np.concatenate([[init_nox], opt_nox])
    sub_plot_fuc(row=5, col=1, plot_num=2, name='Nox', true=state_data['SCR反应器入口Nox含量'], opt=opt_nox,
                 color_true='thistle', color_opt='red')
    plt.xlabel('time (min)', fontsize=10)
    plt.ylabel('mg/m^3', fontsize=10)

    opt_load = next_state_opt['#1机组锅炉负荷'][:-1]
    init_load = state_data['#1机组锅炉负荷'][0]
    opt_load = np.concatenate([[init_load], opt_load])
    sub_plot_fuc(row=5, col=1, plot_num=3, name='Load', true=state_data['#1机组锅炉负荷'], opt=opt_load,
                 color_true='thistle', color_opt='red')
    plt.xlabel('time (min)', fontsize=10)
    plt.ylabel('MW', fontsize=10)

    opt_steam_temp = next_state_opt['锅炉主蒸汽温度'][:-1]
    init_steam_temp = state_data['锅炉主蒸汽温度'][0]
    opt_steam_temp = np.concatenate([[init_steam_temp], opt_steam_temp])
    sub_plot_fuc(row=5, col=1, plot_num=4, name='Steam_temp', true=state_data['锅炉主蒸汽温度'], opt=opt_steam_temp,
                 color_true='thistle', color_opt='red')
    plt.xlabel('time (min)', fontsize=10)
    plt.ylabel('ssd', fontsize=10)

    opt_steam_pres = next_state_opt['主蒸汽压力'][:-1]
    init_steam_pres = state_data['主蒸汽压力'][0]
    opt_steam_pres = np.concatenate([[init_steam_pres], opt_steam_pres])
    sub_plot_fuc(row=5, col=1, plot_num=5, name='Steam_pres', true=state_data['主蒸汽压力'], opt=opt_steam_pres,
                 color_true='thistle', color_opt='red')
    plt.xlabel('time (min)', fontsize=10)
    plt.ylabel('KPa', fontsize=10)

    plt.show()
    # plt.savefig('result/figure/figure_state.png', bbox_inches='tight')# 图片紧凑显示


def plot_all():
    # custom_font = mpl.font_manager.FontProperties(fname='C:/Windows/Fonts/STXIHEI.TTF')
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


    action_data = pd.read_csv('results/test_actor/test_result_action_data.csv')
    action_opt = pd.read_csv('results/test_actor/test_result_action_opt.csv')
    state_data = pd.read_csv('results/test_actor/test_result_state_data.csv')
    next_state_opt = pd.read_csv('results/test_actor/test_result_next_state_opt.csv')

    action_name = action_data.columns
    state_name = state_data.columns

    sns.set_style("white")

    state_plot(state_data, next_state_opt)
    action_plot(action_name, action_data, action_opt)


if __name__ == '__main__':
    plot_all()
