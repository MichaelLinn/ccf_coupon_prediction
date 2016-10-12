# -*- coding:utf-8 -*- 
"""
Created on '2016/10/9' '13:59'

@author: 'michael"  

#online_data
Action: 0 点击， 1购买，2领取优惠券
Coupon_id: null表示无优惠券消费，此时Discount_rate和Date_received字段无意义。
       “fixed” 表示该交易是限时低价活动。
Discount_rate: x \in [0,1]代表折扣率；x:y表示满x减y；“fixed”表示低价限时优惠；
Date_received: 领取优惠券日期
Date: 消费日期：  如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用；
                如果Date!=null & Coupon_id = null，则表示普通消费日期；
                如果Date!=null & Coupon_id != null，则表示用优惠券消费日期；

#offline_data
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
import pickle


offline_filename = "../../data/ccf_offline_stage1_train.csv"
online_filename = "../../data/ccf_online_stage1_train.csv"
test_filename = "../../data/ccf_offline_stage1_test.csv"

offline_train_data_pklfilename = "./offline_train_data.pkl"
washed_labeled_data_pklfilename = "./washed_labeled_data.pkl"

def load_offline_data(filename = offline_filename):
    """
    load the data_offline
    convert string_time into datetime
    and make the data persistent as a *.pkl
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
        if data_offline['Date'][i] != "null":
            udate_list.append(datetime.datetime.strptime(data_offline['Date'][i],"%Y%m%d"))
        else:
            udate_list.append("null")
    data_offline['Date_received'] = np.array(rdate_list)
    data_offline['Date'] = np.array(udate_list)
    pickle._dump(data_offline,open("offline_train_data.pkl","wb"))
    return data_offline

def load_online_data(filename = online_filename):
    """
    load the data_online
    convert string_time into datetime
    and make the data persistent as a *.pkl
    :param filename:
    :return: data_online
    """
    header = ['User_id','Merchant_id','Action','Coupon_id',
              'Discount_rate','Date_received','Date']
    data_online = pd.read_csv(filename,names = header)

    return data_online

def load_test_data(filename = test_filename):
    """
    load the test data and make the data persistent as a *.pkl
    :param filename:
    :return: data_test
    """
    header = ['User_id','Merchant_id','Coupon_id',
              'Discount_rate','Distance','Date_received']
    test_data = pd.read_csv(filename,names = header)
    pickle._dump(test_data,open("./pretreat_test_data/test_data.pkl","wb"))
    print(test_data)

def count_coupon_day(pklfiname = offline_train_data_pklfilename):
    offline_data = pickle.load(open(pklfiname,"rb"))
    day_beforeUsed = []
    for i in range(len(offline_data)):
        if offline_data['Date_received'][i] != "null" and offline_data['Date'][i] != "null":
            day_beforeUsed.append(offline_data['Date'][i] - offline_data['Date_received'][i])
    day_sorted = np.array(sorted(day_beforeUsed,reverse = True))
    pickle._dump(day_sorted,open("offline_coupon_using_day.pkl","wb"))
    below15_days = len(day_sorted[day_sorted <= datetime.timedelta(15)])
    print(below15_days)

def random_test_toCSV(test_data):
    result_random = []
    for i in range(len(test_data)):
        result_random.append(np.random.random())
    result = np.array(result_random)
    test_data['result'] = result
    test_data.to_csv("test_result.csv",index = False , header = False ,
                     columns = ['User_id','Coupon_id','Date_received','result'])

def label_offline_data(pklfilename = offline_train_data_pklfilename):
    """
    delete the irrelevant data and label the effective data
    convert the distance_str into distance_int
    # Date : 消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本；
             如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null，则表示用优惠券消费日期，即正样本；
    :param pklfilename:
    :return:
    """
    offline_data = pickle.load(open(pklfilename,"rb"))
    labels = []
    for i in range(len(offline_data)):
        if offline_data['Coupon_id'][i] == "null":
            labels.append(0)
        else:
            if offline_data['Date'][i] != "null":
                labels.append(1)
            else:
                labels.append(-1)
    offline_data['Label'] = np.array(labels)
    pickle._dump(offline_data,open("labeled_offline_train_data.pkl","wb"))
    washed_labeled_data = pd.DataFrame(offline_data[offline_data['Label'] != 0])
    washed_labeled_data.to_csv("tem.csv",header=True,index=False)
    washed_labeled_data = pd.read_csv("tem.csv")
    # print(washed_labeled_data)
    # Warning: the index of dataframe cannot be modified,and the slices of the dataframe is not the copy of the dataframe
    #          though we delete a row of the dataframe, we cannot change the indices

    pickle._dump(washed_labeled_data,open("washed_labeled_data.pkl","wb"))


# label_offline_data()
# labeled_data = pickle.load(open("labeled_offline_train_data.pkl","rb"))
# washed_labeled_data = labeled_data[labeled_data['Label'] != 0]
# pickle._dump(washed_labeled_data,open("washed_labeled_data.pkl","wb"))

"""
# def dispose_null_distance(washed_labeled_data):
# there is an ideal that using logistic model to predict the null value
"""

def convert_distance_int(pklfilename = washed_labeled_data_pklfilename,targert_filename=washed_labeled_data_pklfilename):
    washed_labeled_data = pickle.load(open(pklfilename,"rb"))
    dis_list = []
    for i in range(len(washed_labeled_data)):
        if str(washed_labeled_data['Distance'][i]) != "null":
            dis_list.append(int(washed_labeled_data['Distance'][i]))
        else:
            dis_list.append(12)
    washed_labeled_data['Distance'] = np.array(dis_list)
    pickle._dump(washed_labeled_data,open(targert_filename,"wb"))

def convert_discount_num(pklfilename = washed_labeled_data_pklfilename,targert_filename = washed_labeled_data_pklfilename):
    """
    conver the '*:*' into float number
    create two feature: one is total money ,the other is discount money
    for example '50:10',total money is 50 , discount money is 10
    :param  pklfilename:
    :return:
    """
    upper_limit = []
    allowance = []
    discount_rate = []
    washed_data = pickle.load(open(pklfilename,"rb"))
    for i in range(len(washed_data)):
        if washed_data['Discount_rate'][i].find(':') != -1:
            tem = washed_data['Discount_rate'][i].split(':')
            upper_limit.append(int(tem[0]))
            allowance.append(int(tem[1]))
            discount_rate.append((1 - int(tem[1])*1.0/int(tem[0])))
        else:
            upper_limit.append(100)
            rate = 1 - float( washed_data['Discount_rate'][i] )
            allowance.append(int(100*1.0*rate))
            discount_rate.append(float(washed_data['Discount_rate'][i]))
    washed_data['Discount_rate'] = np.array(discount_rate)
    washed_data['Upper_limit'] = np.array(upper_limit)
    washed_data['Allowance'] = np.array(allowance)
    # print(washed_data)
    pickle._dump(washed_data,open(targert_filename,"wb"))

# the training features are "Discount_rate" , "Upper_limit" , "Allowance", "Distance"
# the class set is {-1,1}
# file "washed_labeled_data.pkl" is the after pretreatment data set, whose type is pandas.Dataframe

def convert_pkl_toCSV(pklfilename = washed_labeled_data_pklfilename):
    washed_data = pickle.load(open(pklfilename,"rb"))
    washed_data.to_csv("washed_labeled_data.csv",header = True,index= False)


# convert_pkl_toCSV()



# label_offline_data()
test_pklfilename = "./pretreat_test_data/test_data.pkl"
targert_filename = "test_dis_data.pkl"
# convert_distance_int(test_pklfilename,targert_filename)
# convert_discount_num(targert_filename,targert_filename)

test_data = pickle.load(open(targert_filename,"rb"))
























# days_used_received = pickle.load(open("offline_coupon_using_day.pkl","rb"))
# print(len(days_used_received) ," : " , len(days_used_received[days_used_received <= datetime.timedelta(15)]))

# print(washed_labeled_data)

# convert_discount_num()
# washed_labeled_data = label_offline_data(washed_labeled_data_pklfilename)
# convert_distance_int()

# label_offline_data()
# washed_labeled_data = pickle.load(open(washed_labeled_data_pklfilename, "rb"))
# if washed_labeled_data['Distance'][0] != "null":
# print(washed_labeled_data)
# pd = pd.DataFrame(washed_labeled_data)
# washed_labeled_data.to_csv("test.csv",index=False,header = False)
# data = pd.read_csv("test.csv")

# random_test_toCSV(load_test_data())
# test_data = load_test_data()
# pickle._dump(test_data,open("test_data.pkl","wb"))
# tem_data = pickle.load(open("offline_train_data.pkl","rb"))

"""
washed_labeled_data = pickle.load(open(washed_labeled_data_pklfilename,"rb"))
tem = []
for i in range(len(washed_labeled_data)):
    if washed_labeled_data['Distance'][i] == -1:
        tem.append(12)
    else:
        tem.append(washed_labeled_data['Distance'][i])
washed_labeled_data['Distance'] = np.array(tem)
print(washed_labeled_data[washed_labeled_data['Distance'] > 10])
pickle._dump(washed_labeled_data,open(washed_labeled_data_pklfilename,"wb"))
"""