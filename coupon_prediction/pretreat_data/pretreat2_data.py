# -*- coding:utf-8 -*- 
"""
Created on '2016/10/11' '12:30'

@author: 'michael"  
"""
import numpy as np
import pandas as pd
import pickle
import datetime
import random

offline_data_pklfilename = "offline_train_data.pkl"
pre_data_pklfilename = "1pre_data.pkl"
disc_dispose_filename = "disc_dispose_data.pkl"
labeled_data_pklfilename = "labeled_data.pkl"

def convert_distance_feature(pklfilename = offline_data_pklfilename,target_filename = offline_data_pklfilename):
    train_data = pickle.load(open(pklfilename,"rb"))
    dis_list = []
    for i in range(len(train_data)):
        if train_data['Distance'][i] == 'null':
            dis_list.append('12')
        else:
            dis_list.append(train_data['Distance'][i])
    train_data['Distance'] = np.array(dis_list)
    # print(train_data)
    pickle._dump(train_data,open(target_filename,"wb"))


def dispose_Discount_rate(pklfilename = pre_data_pklfilename,target_filename=pre_data_pklfilename):
    discount_dic = {}
    train_data = pickle.load(open(pklfilename,"rb"))
    discount_list = np.array(train_data['Discount_rate'])
    discount_name = np.unique(discount_list)
    # print(len(discount_name))
    for i in range(len(discount_name)):
        discount_dic[discount_name[i]] = str(i)
    # print(sorted(discount_dic.items(),key = lambda x:x[1]))
    discount_category = []
    for i in range(len(train_data)):
        discount_category.append(discount_dic[train_data['Discount_rate'][i]])
    train_data['Discount_rate'] = pd.Series(discount_category,dtype= "category")
    # print(train_data['Discount_rate'])
    pickle._dump(train_data,open(target_filename,"wb"))

def label_data(pklfilename = disc_dispose_filename):
    """
    Date : 消费日期：如果Date=null & Coupon_id != null，该记录表示领取优惠券但没有使用，即负样本；
           如果Date!=null & Coupon_id = null，则表示普通消费日期；如果Date!=null & Coupon_id != null，则表示用优惠券消费日期，即正样本；

    label:
            3: positive sample
            0: negative sample
            1: normal consumption sample
            2: more than 15 days sample
    :param pklfilename:
    :return:
    """
    train_data = pickle.load(open(pklfilename,"rb"))
    data_label = []
    for i in range(len(train_data)):
        if train_data['Coupon_id'][i] != "null" and train_data['Date'][i] == 'null':
            data_label.append('-1')
        else:
            if train_data['Coupon_id'][i] == "null" and train_data['Date'][i] != 'null':
                data_label.append('3')
            else:
                if (train_data['Date'][i] - train_data['Date_received'][i]) > datetime.timedelta(15):
                    data_label.append('2')
                else:
                    data_label.append('1')
    train_data['Label'] = pd.Series(data_label,dtype="category")
    print(train_data['Label'])
    pickle._dump(train_data,open("labeled_data.pkl","wb"))

def k_fold_sampling(pklfilename=labeled_data_pklfilename,target_fold = ""):
    train_data = pickle.load(open(pklfilename,"rb"))
    positive_data = train_data[train_data['Label'] == 1]
    negative_data = train_data[train_data['Label'] == -1]
    sum_p = len(positive_data)
    sum_n = len(negative_data)
    p_index = positive_data.index
    n_index = negative_data.index
    single_fold_num = int(sum_n/13.9)
    #print(single_fold_num)
    tem_nindex_list = n_index.tolist()
    for i in range(13):
        print(len(tem_nindex_list))
        single_fold_index = random.sample(list(tem_nindex_list),single_fold_num)
        tem_index = single_fold_index + p_index.tolist()
        select_data_toCSV(tem_index,train_data,i,target_fold)
        tem_nindex_list = np.setdiff1d(np.array(tem_nindex_list),np.array(single_fold_index))

def select_data_toCSV(index_list,train_data,i,csv_foldname="./k_fold_own_data/"):
    selected_train_data = train_data.ix[index_list,:]
    selected_train_data.to_csv("%s%d_fold_train_data.csv"%(csv_foldname,i),index=False)


# test_data = pickle.load(open("./pretreat_test_data/test_data.pkl","rb"))
# filename = "./pretreat_test_data/test_data_dis.pkl"
# target_filename = "./pretreat_test_data/test_data_discount.pkl"
# convert_distance_feature(filename,target_filename)
# dispose_Discount_rate(filename,target_filename)
# test_data = pickle.load(open("./pretreat_test_data/test_data_discount.pkl","rb"))

# test_data.to_csv("./pretreat_test_data/test_data_final.csv",index=False)

train_own_data = pickle.load(open("./washed_labeled_data.pkl","rb"))
filename = "./washed_labeled_data.pkl"
foldname = "./k_fold_own_data/"
k_fold_sampling(filename,foldname)














# dispose_Discount_rate()
# label_data()
# wdata = pickle.load(open("labeled_data.pkl","rb"))
# print(wdata)