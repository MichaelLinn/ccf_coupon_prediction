# -*- coding:utf-8 -*- 
"""
Created on '2016/10/9' '13:59'

@author: 'michael"  
"""

import numpy as np
import pandas as pd
import datetime
import pymysql

offline_filename = "../../data/ccf_offline_stage1_train.csv"
online_filename = "../../data/ccf_online_stage1_train.csv"

def load_offline_data(filename = offline_filename):
    """
    load the data_offline
    convert string_time into timestamp
    """
    header = ['User_id','Merchant_id','Coupon_id',
              'Discount_rate','Distance','Date_received','Date']
    data_offline = pd.read_csv(filename,names = header)
    # print(data_offline)
    return data_offline



def load_online_data(filename = online_filename):
    """
    load the data_online
    convert string_time into timestamp
    :param filename:
    :return: data_online
    """
    header = ['User_id','Merchant_id','Action','Coupon_id',
              'Discount_rate','Date_received','Date']
    data_online = pd.read_csv(filename,names = header)
    return data_online

data_online = load_online_data()
print(len(data_online))