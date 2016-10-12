# -*- coding:utf-8 -*- 
"""
Created on '2016/9/9' '10:32'

@author: 'michael"  


this adaboost model based on the decision stump,
the decision trump classifies the data using only one feature at one time
"""

import numpy as np
import pandas as pd
import pickle

class adaBoost:

    def __init__(self):
        self.dataMat = np.matrix([[1., 2.1,50], [2., 1.1,50], [1.3, 1.,100], [1., 1.,100], [2., 1.,50]])
        self.classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    def loadSimpData(self):
        dataMat = np.matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        return dataMat,classLabels



    def stumpDecisionTree(self,dataMat,dimen, thresholdVal, thresholdIneq):
        retArray = np.ones((np.shape(dataMat)[0],1))   # nunmber of the data(vector) in dataMatrix
        if thresholdIneq == 'lt':                       # two kinds of the inequality type
            retArray[dataMat[:,dimen] <= thresholdVal] = -1.0
            # print(dataMat[:,dimen].T , thresholdIneq , thresholdVal)
            # print((dataMat[:,dimen] <= thresholdVal).T)
        else:
            retArray[dataMat[:,dimen] > thresholdVal] = -1.0
            # print(dataMat[:, dimen].T, " ", thresholdVal)
            # print((dataMat[:, dimen] <= thresholdVal).T)
        return retArray

    def buildStump(self,dataArr,classLabels,D):  # D is a vector of the data's weight
        dataMatrix = np.mat(dataArr)
        labelMat = np.mat(classLabels).T
        m,n = np.shape(dataMatrix)
        numSteps = 100.0
        bestStump = {}
        bestClassEst = np.mat(np.zeros((m,1)))
        minError = np.inf
        for i in range(n):
            rangeMin = dataMatrix[:,i].min()
            rangeMax = dataMatrix[:,i].max()
            stepSize = (rangeMax - rangeMin)/numSteps
            for j in range(-1,int(numSteps) + 1):
                for inequal in ['lt','gt']:
                    thresholdVal = (rangeMin + float(j) * stepSize)
                    predictedVals = self.stumpDecisionTree(dataMatrix,i,thresholdVal,inequal)
                    # print("Predict value:" , predictedVals.T)
                    errArr = np.mat(np.ones((m,1)))
                    errArr[predictedVals == labelMat] = 0   # set 0 to the vector which is classified correctly
                    # print(predictedVals.T," ",labelMat.T)
                    weightedError = D.T * errArr
                    # print("split: dim %d, threshold value %.2f ,threshold inequal: %s, the weighted error is %.3f" %(i,thresholdVal,inequal,weightedError))
                    if weightedError < minError:
                        minError = weightedError
                        bestClassEst = predictedVals.copy()
                        bestStump['dimension'] = i
                        bestStump['inequal'] = inequal
                        bestStump['threshold'] = thresholdVal
        return bestStump,minError,bestClassEst

    def adaBoostTrainDecisionStump(self,dataArr,classLabels,numInt=200):
        weakDecisionStumpArr = []
        m = np.shape(dataArr)[0]
        weight = np.mat(np.ones((m,1))/m)     # init the weight of the data.Normally, we set the initial weight is 1/n
        aggressionClassEst = np.mat(np.zeros((m,1)))
        for i in range(numInt): # classEst == class estimation
            bestStump,error,classEst = self.buildStump(dataArr,classLabels,weight) # weight is a vector of the data's weight
            # print("D: ",weight.T)
            alpha = float(0.5 * np.log((1.0 - error)/max(error , 1e-16)))   # alpha is the weighted of the weak classifier
            bestStump['alpha'] = alpha
            weakDecisionStumpArr.append(bestStump)
            exponent = np.multiply(-1* alpha * np.mat(classLabels).T , classEst) # calculte the exponent [- alpha * Y * Gm(X)]
            print("classEst ：",classEst.T)
            weight = np.multiply(weight,np.exp(exponent)) # update the weight of the data, w_m = e^[- alpha * Y * Gm(X)]
            weight = weight/weight.sum()  # D.sum() == Z_m (Normalized Factor) which makes sure the D_(m+1) can be a probability distribution
            # give every estimated class vector (the classified result of the weak classifier) a weight
            aggressionClassEst += alpha*classEst
            print("aggression classEst: ",aggressionClassEst.T)
            # aggressionClassError = np.multiply(np.sign(aggressionClassEst) != np.mat(classLabels).T, np.ones((m,1)))
            # errorRate = aggressionClassError.sum()/m
            errorRate = (np.sign(aggressionClassEst) != np.mat(classLabels).T).sum()/m # calculate the error classification
            # errorRate = np.dot((np.sign(aggressionClassEst) != np.mat(classLabels).T).T,np.ones((m,1)))/m
            print("total error: ",errorRate,"\n")
            if errorRate == 0:
                break
        return weakDecisionStumpArr



    def adaClassify(self,data,adaBoostModel):
        dataMat = np.mat(data)
        aggClassEst = np.mat(np.zeros((dataMat.shape[0],1)))
        for i in range(len(adaBoostModel)):
            classLabels = self.stumpDecisionTree(dataMat,adaBoostModel[i]['dimension'],adaBoostModel[i]['threshold'],adaBoostModel[i]['inequal'])
            aggClassEst += classLabels*adaBoostModel[i]['alpha']
        # print(aggClassEst)
        predictVals = self.sigmoid_function(aggClassEst.tolist())
        return predictVals
        # return aggClassEst

    def load_train_data(self,i):
        """
        training example:np.matrix([[1., 2.1, 50], [2., 1.1, 50], [1.3, 1., 100], [1., 1., 100], [2., 1., 50]])
        training feature:"Discount_rate" , "Upper_limit" , "Allowance", "Distance"
        :return:
        """

        train_data_csv_file = "./k_fold_own_data/%d_fold_train_data.csv"%i
        train_dataframe = pd.read_csv(train_data_csv_file)
        train_data = []
        data_label = []
        for i in range(len(train_dataframe)):
            train_data.append([train_dataframe['Discount_rate'][i],train_dataframe['Upper_limit'][i],
                               train_dataframe['Allowance'][i],train_dataframe['Distance'][i]])
            data_label.append(train_dataframe['Label'][i])
        train_dataMat = np.matrix(train_data)
        # print(train_dataMat)
        # print(data_label)
        return train_dataMat,data_label

    def load_test_data(self):
        test_data_csv_file = "./adaboost_result/test_data.csv"
        test_data = pd.read_csv(test_data_csv_file)
        #print(test_data)
        test_data_list = []
        for i in range(len(test_data)):
            test_data_list.append([test_data['Discount_rate'][i],test_data['Upper_limit'][i],
                                   test_data['Allowance'][i],test_data['Distance'][i]])
        return test_data_list


    def sigmoid_function(self,data):
        result = []
        for i in range(len(data)):
            tem = 1.0 / (1.0 + np.e**(-(data[i][0])))
            result.append(tem)
        return result

    def bulid_csv_file(self,result_list, test_data_csv_filename = "../../data/ccf_offline_stage1_test.csv" ):
        probability = np.array(result_list)
        header = ['User_id', 'Merchant_id', 'Coupon_id',
                  'Discount_rate', 'Distance', 'Date_received']
        test_data = pd.read_csv(test_data_csv_filename,names=header)
        test_data["Prob"] = probability
        test_data.to_csv("./adaboost_result/test_result_13_fold.csv", index=False, header=False,
                         columns=['User_id', 'Coupon_id', 'Date_received', 'Prob'])



    def build_k_fold_models(self,num_fold = 13):
        for i in range(num_fold):
            train_dataMat,data_label = self.load_train_data(i)
            adaBoostModel = self.adaBoostTrainDecisionStump(train_dataMat,data_label)
            pickle._dump(adaBoostModel, open("./adaboost_result/adaboost_model%d.pkl" % i, "wb"))



    def converge_k_fold_predict_vals(self,num_fold = 13):
        test_data = self.load_test_data()
        sum_predict_vals = np.zeros((len(test_data)))
        for i in range(num_fold):
            adaBoostModel = pickle.load(open("./adaboost_result/adaboost_model%d.pkl"%i,"rb"))
            tem_result = self.adaClassify(test_data,adaBoostModel)
            sum_predict_vals += np.array(tem_result)
        result = sum_predict_vals/num_fold
        return result.tolist()


"""
training example:np.matrix([[1., 2.1, 50], [2., 1.1, 50], [1.3, 1., 100], [1., 1., 100], [2., 1., 50]])
training feature:"Discount_rate" , "Upper_limit" , "Allowance", "Distance"
"""
ab = adaBoost()
# test_data_list = ab.load_test_data()
# num_fold = 13
# i = 12
train_dataMat,data_label = ab.load_train_data(2)
adaBoostModel = ab.adaBoostTrainDecisionStump(train_dataMat,data_label)
# pickle._dump(adaBoostModel,open("./adaboost_result/adaboost_model%d.pkl"%i,"wb"))


#train_dataMat,data_label = ab.load_data()
#adaBoostTrees = ab.adaBoostTrainDecisionStump(train_dataMat,data_label)
#pickle._dump(adaBoostTrees,open("./adaboost_result/test2.pkl","wb"))
#ad_trees = pickle.load(open("./adaboost_result/adaboost_model0.pkl","rb"))

#test_data = ab.load_test_data()
# print(test_data)
#result = ab.adaClassify(test_data,ad_trees)
#ab.bulid_csv_file(result)
"""
ab = adaBoost()
result = ab.converge_k_fold_predict_vals()
print(result)
ab.bulid_csv_file(result)
"""




"""

def main():
    ab = adaBoost()
    weight = np.mat(np.ones((5,1))/5)
    ab.buildStump(ab.dataMat,ab.classLabels,weight)
    adaBoostTrees = ab.adaBoostTrainDecisionStump(ab.dataMat,ab.classLabels,9)
    arr = np.array([1.3,1.0,50])
    preditVal = ab.classify(adaBoostTrees,arr)
    print("predictVal:" , preditVal)


if __name__ == '__main__':
    main()

def classify(self,adaboostModel,arr_single):
    classLabel = []
    # n_row,n_col = arr_single.shape
    for i in range(len(adaboostModel)):
        inequal = adaboostModel[i]['inequal']
        if inequal == 'lt':
            if arr_single[adaboostModel[i]['dimension']] <= adaboostModel[i]['threshold']:
                classLabel.append(-1)
            else:
                classLabel.append(1)
        else:
            if arr_single[adaboostModel[i]['dimension']] >= adaboostModel[i]['threshold']:
                classLabel.append(-1)
            else:
                classLabel.append(1)
        classLabel[i] = classLabel[i] * adaboostModel[i]['alpha']
    predictValue = np.sign(sum(classLabel))
    return predictValue
"""