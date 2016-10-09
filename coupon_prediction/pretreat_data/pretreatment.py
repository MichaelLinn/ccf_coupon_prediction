# -*- coding:utf-8 -*- 
"""
Created on '2016/10/9' '13:59'

@author: 'michael"  


Action: 0 点击， 1购买，2领取优惠券
Coupon_id: null表示无优惠券消费，此时Discount_rate和Date_received字段无意义。
       “fixed”表示该交易是限时低价活动。
Discount_rate: x \in [0,1]代表折扣率；x:y表示满x减y；“fixed”表示低价限时优惠；
Date_received: 领取优惠券日期
Date: 消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用；
                如果Date!=null & Coupon_id = null，则表示普通消费日期；
                如果Date!=null & Coupon_id != null，则表示用优惠券消费日期；

Coupon_id : null表示无优惠券消费，此时Discount_id和Date_received字段无意义
Discount_rate : 优惠率 x \in [0,1]代表折扣率; x:y 表示满x减y。单位是元
Distance : 	user经常活动的地点离该merchant的最近门店距离是x*500米（如果是连锁店，则取最近的一家门店），
            x\in[0,10]；null表示无此信息，0表示低于500米，10表示大于5公里；
Date_received : 领取优惠券日期
Date : 消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本；
       如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null，则表示用优惠券消费日期，即正样本；
"""

import numpy as np
import pandas as pd
import datetime
import pymysql

offline_filename = "../../data/ccf_offline_stage1_train.csv"
online_filename = "../../data/ccf_online_stage1_train.csv"
test_filename = "../../data/ccf_offline_stage1_test.csv"

def load_offline_data(filename = offline_filename):
    """
    load the data_offline
    convert string_time into timestamp
    """
    header = ['User_id','Merchant_id','Coupon_id',
              'Discount_rate','Distance','Date_received','Date']
    data_offline = pd.read_csv(filename,names = header)
    rdate_list = []
    udate_list = []
    for i in range(len(data_offline['Date_received'])):
        if data_offline['Date_received'][i] != "null":
            rdate_list.append(datetime.datetime.strptime(data_offline['Date_received'][i],"%Y%m%d"))
        else:
            rdate_list.append('null')
        if data_offline['Date'] != "null":
            udate_list.append(datetime.datetime.strptime(data_offline['Date'][i],"%Y%m%d"))
        else:
            udate_list.append("null")
    data_offline['Date_received'] = np.array(rdate_list)
    data_offline['date'] = np.array(udate_list)
    print(data_offline)
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

def load_test_data(filename = test_filename):
    """
    load the test data
    :param filename:
    :return: data_test
    """
    header = ['User_id','Merchant_id','Coupon_id',
              'Discount_rate','Distance','Date_received']
    test_data = pd.read_csv(filename,names = header)
    return test_data


def count_coupon_day(Date_received,Date_used):
    day_beforeUsed = []
    for i in range(len(Date_received)):
        if Date_received[i] != "null" and Date_used[i] != "null":
            day_beforeUsed.append(Date_used[i] - Date_received[i])
    day_sorted = sorted(day_beforeUsed,reverse = True)
    print(day_sorted)

load_offline_data()