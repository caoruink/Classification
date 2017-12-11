# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
from sklearn.model_selection import KFold
from sklearn.neighbors import kde


class DataSet(object):
    """
    1.读取文件中的数据集，
    2.按照需要拆成训练集、验证集、测试集，
    3.用最大似然估计得出正态分布的参数，用似然率测事规则给出分类性能结果，
    4.使用平滑核函数估计数据分布，用似然率测事规则给出分类性能结果，
    5.是用朴素贝叶斯方法分类
    6.使用最近邻决策方法分类
    """
    def __init__(self, filename):
        """
        读取数据
        :param filename: 读取数据的文件名称
        """
        self.__data_set = np.loadtxt(filename, delimiter=',')
        self.X = self.__data_set[:, 0: 4]       # 特征
        self.Y = self.__data_set[:, 4]          # 目标
        self.__num_split = 10                   # 交叉验证的数目

    def classify(self):
        kf = KFold(n_splits=self.__num_split, shuffle=True)
        # 区分训练集和测试集，做十折交叉验证，训练集得出三个模型参数，测试集按照模型的参数用似然率规则决断
        error_ml = []
        error_kde = []
        error_nb = []
        error_knn = []
        for train_index, test_index in kf.split(self.X):
            x_train, x_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]
            error_ml.append(self.maximum_likelihood_estimation(x_train, x_test, y_train, y_test))
            error_kde.append(self.smoothing_kernel_function(x_train, x_test, y_train, y_test))
            error_nb.append(self.naive_bayes(x_train, x_test, y_train, y_test))
            error_knn.append(self.nearest_neighbor_decision(x_train, x_test, y_train, y_test))
        print("the mean error rate of the maximum likelihood rate test is ", np.mean(error_ml))
        print("the mean error rate of the smoothing kernel function is ", np.mean(error_kde))
        print("the mean error rate of the naive bayes is ", np.mean(error_nb))
        print("the mean error rate of the nearest_neighbor_decision is ", np.mean(error_knn))

    def maximum_likelihood_estimation(self, x_train, x_test, y_train, y_test):
        """
        最大似然估计
        :return:
        """
        # 训练集得出三个模型参数，测试集按照模型的参数用似然率规则决断
        # 根据训练集估计三个模型的参数，并返回
        model1, model2, model3 = self.cal_parameter(x_train, y_train)
        # 用似然率测试规则给出分类结果并记录
        return self.likelihood_rate(model1, model2, model3, x_test, y_test)

    def likelihood_rate(self, model1, model2, model3, x_test, y_test):
        """
        利用似然率测试规则分类
        :return: 错误率
        """
        # 错误的个数
        num_error = 0
        # 对样本中的每个向量进行判断，选取概率最大的标签最为决策的标签
        for each in np.arange(0, len(x_test)):
            decide_class = None
            prob1 = self.posterior_probability(model1, x_test[each, :])
            prob2 = self.posterior_probability(model2, x_test[each, :])
            prob3 = self.posterior_probability(model3, x_test[each, :])
            if max([prob1, prob2, prob3]) == prob1:
                decide_class = 1
            if max([prob1, prob2, prob3]) == prob2:
                decide_class = 2
            if max([prob1, prob2, prob3]) == prob3:
                decide_class = 3
            if not self.__equal(decide_class, y_test[each]):
                num_error += 1
        # 计算错误率
        error_rate = num_error / len(x_test)
        # print("the error rate of the likelihood-rate test is ", error_rate)
        return error_rate

    @staticmethod
    def cal_parameter(x_train, y_train):
        """
        估计每个类别的高斯分布的参数
        :param x_train:训练集的数据
        :param y_train:训练集的标签，按照标签分类
        :return:三个模型的均值和协方差
        """
        # 分别提取三个类别的数据
        index1 = []
        index2 = []
        index3 = []
        for each in np.arange(0, len(y_train)):
            if y_train[each] == 1:
                index1.append(each)
            if y_train[each] == 2:
                index2.append(each)
            if y_train[each] == 3:
                index3.append(each)
        data1_train = x_train[index1, :]
        data2_train = x_train[index2, :]
        data3_train = x_train[index3, :]
        # 估计模型参数
        mean1 = np.mean(data1_train.T, 1)
        cov1 = np.cov(data1_train.T)
        mean2 = np.mean(data2_train.T, 1)
        cov2 = np.cov(data2_train.T)
        mean3 = np.mean(data3_train.T, 1)
        cov3 = np.cov(data3_train.T)
        model1 = {'mean': mean1, 'cov': cov1}
        model2 = {'mean': mean2, 'cov': cov2}
        model3 = {'mean': mean3, 'cov': cov3}
        return model1, model2, model3

    @staticmethod
    def posterior_probability(model, x):
        """
        计算后验概率，这里省去了常数项，因为常数在作比较时可以约掉
        :param model: 正态分布模型的参数
        :param x: 向量
        :return: 向量的后验概率
        """
        return (1/np.sqrt(np.linalg.det(model['cov']))) * np.exp(-(1 / 2) * mat((x - model['mean'])) *
                                                                 mat(model['cov']).I * mat((x - model['mean'])).T)

    @staticmethod
    def __equal(x, y):
        """
        判断利用分类规则决策的结果于原结果是否一样
        :param x: 决策的类别
        :param y: 原结果
        :return: 一样，返回真，否则返回假
        """
        if x == y:
            return True
        return False

    def smoothing_kernel_function(self, x_train, x_test, y_train, y_test):
        # 使用高斯核函数进行估计
        # 交叉验证h的取值
        # error_new = []
        # kf1 = KFold(n_splits=self.__num_split, shuffle=True)
        # for train_index_new, val_index in kf1.split(x_train):
        #     x_train_new, x_val = x_train[train_index_new], x_train[val_index]
        #     y_train_new, y_val = y_train[train_index_new], y_train[val_index]
        #     model1_new, model2_new, model3_new = self.kernel_gaussian(x_train_new, y_train_new)
        #     error_new.append(self.likelihood_rate_gaussian(model1_new, model2_new, model3_new, x_val, y_val))
        # print("the mean validation error rate of the smoothing_kernel_function is ", np.mean(error_new))
        # 根据训练集估计三个模型，并返回
        model1, model2, model3 = self.kernel_gaussian(x_train, y_train)
        # 用似然率测试规则给出分类结果并返回
        return self.likelihood_rate_gaussian(model1, model2, model3, x_test, y_test)

    def likelihood_rate_gaussian(self, model1, model2, model3, x_test, y_test):
        """
        利用似然率测试规则分类
        :return: 错误率
        """
        # 错误的个数
        num_error = 0
        # 对样本中的每个向量进行判断
        for each in np.arange(0, len(x_test)):
            # 计算每个样本属于三个类别的概率
            prob1 = np.exp(model1.score_samples(x_test[each, :].reshape(1, -1)))
            prob2 = np.exp(model2.score_samples(x_test[each, :].reshape(1, -1)))
            prob3 = np.exp(model3.score_samples(x_test[each, :].reshape(1, -1)))
            if max([prob1, prob2, prob3]) == prob1:
                decide_class = 1
            elif max([prob1, prob2, prob3]) == prob2:
                decide_class = 2
            else:
                decide_class = 3
            if not self.__equal(decide_class, y_test[each]):
                num_error += 1
        # 计算错误率
        error_rate = num_error / len(x_test)
        # print("the error rate of the smoothing_kernel_function is ", error_rate)
        return error_rate

    @staticmethod
    def kernel_gaussian(x_train, y_train):
        # 分别提取三个类别的数据
        index1 = []
        index2 = []
        index3 = []
        for each in np.arange(0, len(y_train)):
            if y_train[each] == 1:
                index1.append(each)
            if y_train[each] == 2:
                index2.append(each)
            if y_train[each] == 3:
                index3.append(each)
        data1_train = x_train[index1, :]
        data2_train = x_train[index2, :]
        data3_train = x_train[index3, :]
        # 估计模型
        clf1 = kde.KernelDensity(kernel='gaussian', bandwidth=0.4).fit(data1_train)
        clf2 = kde.KernelDensity(kernel='gaussian', bandwidth=0.4).fit(data2_train)
        clf3 = kde.KernelDensity(kernel='gaussian', bandwidth=0.4).fit(data3_train)
        return clf1, clf2, clf3

    def naive_bayes(self, x_train, x_test, y_train, y_test):
        # 朴素贝叶斯
        model1, model2, model3 = self.nb_model(x_train, y_train)
        # 计算测试集所属每种类别的概率，并得出错误率
        return self.nb_decide(model1, model2, model3, x_test, y_test)

    def nb_model(self, x_train, y_train):
        # 分别提取三个类别的数据
        index1 = []
        index2 = []
        index3 = []
        for each in np.arange(0, len(y_train)):
            if y_train[each] == 1:
                index1.append(each)
            if y_train[each] == 2:
                index2.append(each)
            if y_train[each] == 3:
                index3.append(each)
        data1_train = x_train[index1, :]
        data2_train = x_train[index2, :]
        data3_train = x_train[index3, :]
        # 估计模型, 估计每一维正态分布的参数
        model1 = self.nb_gaussian_kernel(data1_train)
        model2 = self.nb_gaussian_kernel(data2_train)
        model3 = self.nb_gaussian_kernel(data3_train)
        return model1, model2, model3

    def nb_decide(self, model1, model2, model3, x_test, y_test):
        # 错误的个数
        num_error = 0
        # 对样本中的每个向量进行判断
        for each in np.arange(0, len(x_test)):
            prob1 = self.nb_probability(model1, x_test[each, :])
            prob2 = self.nb_probability(model2, x_test[each, :])
            prob3 = self.nb_probability(model3, x_test[each, :])
            if max([prob1, prob2, prob3]) == prob1:
                decide_class = 1
            elif max([prob1, prob2, prob3]) == prob2:
                decide_class = 2
            else:
                decide_class = 3
            if not self.__equal(decide_class, y_test[each]):
                num_error += 1
        # 计算错误率
        # print(num_error)
        error_rate = num_error / len(x_test)
        # print("the error rate of the naive bayes is ", error_rate)
        return error_rate

    @staticmethod
    def nb_gaussian_kernel(data):
        mean_data = []
        cov_data = []
        for i in np.arange(0, 4):
            mean_data.append(np.mean(data[:, i]))
            cov_data.append(np.cov(data[:, i]))
        return {'mean': mean_data, 'cov': cov_data}

    def nb_probability(self, model, x):
        """
        计算多变量似然函数
        :param model: 正态分布的均值和方差
        :param x: 样本
        :return: 似然函数值
        """
        p = 1
        for i in np.arange(0, 4):
            p *= self.gaussian(model['mean'][i], model['cov'][i], x[i])
        return p

    @staticmethod
    def gaussian(mean_model, cov_model, x):
        """
        计算一维正态分布的概率
        :param mean_model: 均值
        :param cov_model: 方差
        :param x: 样本
        :return: 概率
        """
        return 1/(np.sqrt(2 * np.pi) * cov_model) * np.exp(-np.square(x - mean_model)/(2 * np.square(cov_model)))

    def nearest_neighbor_decision(self, x_train, x_test, y_train, y_test):
        k = 6
        # kf_val = KFold(n_splits=self.__num_split, shuffle=True)
        # # 把训练集再拆成训练集和验证集，求最优的k
        # for train_new_index, val_index in kf.split(x_train):
        #     x_train_new, x_val = x_train[train_new_index], x_train[val_index]
        #     y_train_new, y_val = y_train[train_new_index], y_train[val_index]
        # 根据训练集确定距离，利用最近邻决策确定结果
        num_error = 0
        for index_test in np.arange(0, len(x_test)):
            prob = self.find_k_nearest_point(k, x_train, y_train, x_test[index_test])
            # decide_class = None
            if max(prob) == prob[0]:
                decide_class = 1
            elif max(prob) == prob[1]:
                decide_class = 2
            else:
                decide_class = 3
            if not self.__equal(decide_class, y_test[index_test]):
                num_error += 1
        # 求平均的分类正确率
        # print("the error rate of the nearest_neighbor_decision is", num_error / len(x_test))
        return num_error / len(x_test)

    def find_k_nearest_point(self, k, x_train, y_train, x_test):
        # 计算k个与样本点最近的距离，记录标签
        distance = []
        for index_sam_train in np.arange(0, len(x_train)):
            temp_dis = {'dis': 0, 'label': 1}
            temp_dis['dis'] = self.__cal_distance(x_train[index_sam_train], x_test)
            temp_dis['label'] = y_train[index_sam_train]
            distance.append(temp_dis)
        distance = sorted(distance, key=lambda x: x['dis'])
        found_label = distance[0:k]
        # 分别记录每个标签的个数
        num_label1 = 0
        num_label2 = 0
        num_label3 = 0
        for each in found_label:
            if each['label'] == 1:
                num_label1 += 1
            elif each['label'] == 2:
                num_label2 += 1
            else:
                num_label3 += 1
        return [num_label1/k, num_label2/k, num_label3/k]

    @staticmethod
    def __cal_distance(x, y):
        """
        计算向量之间的欧氏距离
        :param x:四维矢量
        :param y:四维矢量
        :return:欧氏距离
        """
        return np.sqrt(np.square(x[0] - y[0]) + np.square(x[1] - y[1]) + np.square(x[2] - y[2]) + np.square(x[3] - y[3]))


if __name__ == '__main__':
    data_set = DataSet("HWData3.csv")
    data_set.classify()
