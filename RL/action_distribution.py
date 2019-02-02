#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:27:41 2019

@author: xhr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
df=pd.read_csv('/Users/xhr/PycharmProjects/Boiler/Simulator/data/replay_buffer.csv', index_col='date').values
#data_df=np.array(df).astype('float')
length=len(df)
print(df.shape)
result=np.zeros((51,21)).astype('float')
for i in range(58,109):
    result[i-58,:20]=plt.hist(df[:,i],20)[0]/length
    plt.close()
    sortedvalue=np.sort(result[i-58,:20])[::-1]
    # print(sortedvalue)
    for j in range(1,20):
        if sum(sortedvalue[:j])>0.97:
            # print(sum(sortedvalue[:j]))
            result[i - 58, 20] =sortedvalue[j]
            break
print(result.shape)
# meanvalue=np.mean(result,axis=1).reshape(95,1)
# stdvalue=np.std(result,axis=1).reshape(95,1)
# print(meanvalue.shape)
# print(stdvalue.shape)
# all=np.concatenate([result,meanvalue,stdvalue],axis=1)
# print(all.shape)




#csv_write = csv.writer(
#    open('D:/data/origin/data_20190114/action_histogram_0.97_v2.csv', 'w',newline=''), dialect='excel')
#for row in result:
#    csv_write.writerow(row)