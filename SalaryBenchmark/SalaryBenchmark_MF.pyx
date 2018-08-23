from __future__ import (absolute_import, division, print_function,unicode_literals)
from surprise import AlgoBase
from surprise import PredictionImpossible
cimport numpy as np 
import numpy as np
from six.moves import range


'''
This file defines two classes used to predict salaries by two Matrix Factorization based methods-- SVD, NMF.
Both of the two methods utilized job-similarity, company-similarity, time-similarity, location-similarity regularizations
We initialize model by setting the common used parameters in MF-based models. 
We need to pass trainset, aux_pu, aux_qi (or aux_pu_num,aux_pu_denom,aux_qi_num,aux_qi_denom)to "train" function to train the model. trainset is pre-defined by Surprise--another package
used for matrix factorization. aux_pu, aux_qi (or aux_pu_num,aux_pu_denom,aux_qi_num,aux_qi_denom) is prepared by SalaryBenchmark_MF_SideMatrix firstly.
"estimate" function used for predictions.

Authors: Qingxin Meng(xinmeng320@gmail.com)
Date:    2018/01/18
'''



class SalaryBenchmark_SVD(AlgoBase):
    def __init__(self, n_factors=5, n_epochs=50, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005, reg_all=.02,
                 lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.verbose = verbose

        AlgoBase.__init__(self)

    def train(self, trainset, aux_pu,aux_qi):

        AlgoBase.train(self, trainset)
        self.sgd(trainset, aux_pu,aux_qi)

    def sgd(self, trainset, aux_pu,aux_qi):

        # user and item factors
        cdef np.ndarray[np.double_t, ndim = 2] pu, qi

        # user and item biases
        cdef np.ndarray[np.double_t] bu, bi
        cdef int u, i, f
        cdef double r, est, l, dot, err

        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double global_mean = trainset.global_mean
        cdef double current_rmse, current_mae
        cdef np.ndarray[np.double_t, ndim = 2]  auxiliary_pu, auxiliary_qi


        # Randomly initialize user and item factors
        pu = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_users, self.n_factors))
        qi = np.random.normal(self.init_mean, self.init_std_dev,
                              (trainset.n_items, self.n_factors))

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        self.min_rating = trainset.rating_scale[0]
        self.max_rating = trainset.rating_scale[1]
        train_rmse_list = []
        train_mae_list = []

        if not self.biased:
            global_mean = 0


        for current_epoch in range(self.n_epochs):
            auxiliary_pu = aux_pu.dot(pu) - reg_pu * pu
            auxiliary_qi = aux_qi.dot(qi) - reg_qi * qi
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            current_rmse = 0.0
            current_mae = 0.0
            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                current_rmse += err ** 2
                current_mae += abs(err)
                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # compute numerators and denominators
                for f in range(self.n_factors):
                    puf = auxiliary_pu[u, f]
                    qif = auxiliary_qi[i, f]
                    pu[u, f] += self.lr_pu * (err * qi[i, f] + puf)
                    qi[i, f] += self.lr_qi * (err * pu[u, f] + qif)

            current_rmse = np.sqrt(current_rmse / trainset.n_ratings)
            train_rmse_list.append(current_rmse)
            current_mae = current_mae / trainset.n_ratings
            train_mae_list.append(current_mae)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.train_rmse_list = train_rmse_list
        self.train_mae_list = train_mae_list

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])


        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])


            else:
                raise PredictionImpossible('User and item are unkown.')

        return est

class SalaryBenchmark_NMF(AlgoBase):
    def __init__(self, n_factors=5, n_epochs=50, biased=True, reg_pu=.06,
                 reg_qi=.06, reg_bu=.02, reg_bi=.02, reg_su=5e-5,reg_si=5e-5,reg_t=1e-5,
                 lr_bu=.005, lr_bi=.005,init_low=0, init_high=1, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.init_low = init_low
        self.init_high = init_high
        self.verbose = verbose

        if self.init_low < 0:
            raise ValueError('init_low should be greater than zero')

        AlgoBase.__init__(self)

    def train(self, trainset, aux_pu_num,aux_pu_denom,aux_qi_num,aux_qi_denom):

        AlgoBase.train(self, trainset)
        self.sgd(trainset, aux_pu_num,aux_pu_denom,aux_qi_num,aux_qi_denom)

    def sgd(self, trainset,aux_pu_num,aux_pu_denom,aux_qi_num,aux_qi_denom):

        # user and item factors
        cdef np.ndarray[np.double_t, ndim = 2] pu,qi

        # user and item biases
        cdef np.ndarray[np.double_t] bu,bi

        # auxiliary matrices used in optimization process
        cdef np.ndarray[np.double_t, ndim = 2] user_num,user_denom,item_num,item_denom
        cdef np.ndarray[np.double_t, ndim = 2] auxiliary_pu_num,auxiliary_pu_denom,auxiliary_qi_num,auxiliary_qi_denom
        cdef int u, i, f
        cdef double r, est, l, dot, err
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double global_mean = self.trainset.global_mean

        # Randomly initialize user and item factors
        pu = np.random.uniform(self.init_low, self.init_high,
                               size=(trainset.n_users, self.n_factors))
        qi = np.random.uniform(self.init_low, self.init_high,
                               size=(trainset.n_items, self.n_factors))

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        train_rmse_list = []
        train_mae_list = []

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            # (re)initialize nums and denoms to zero
            user_num = np.zeros((trainset.n_users, self.n_factors))
            user_denom = np.zeros((trainset.n_users, self.n_factors))
            item_num = np.zeros((trainset.n_items, self.n_factors))
            item_denom = np.zeros((trainset.n_items, self.n_factors))

            auxiliary_pu_num=aux_pu_num.dot(pu)
            auxiliary_pu_denom=aux_pu_denom.dot(pu)+reg_pu*pu
            auxiliary_qi_num=aux_qi_num.dot(qi)
            auxiliary_qi_denom=aux_qi_denom.dot(qi)+reg_qi*qi

            current_rmse = 0.0
            current_mae = 0.0

            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                est = global_mean + bu[u] + bi[i] + dot
                err = r - est

                current_rmse += err ** 2
                current_mae += abs(err)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # compute numerators and denominators
                for f in range(self.n_factors):
                    user_num[u, f] += qi[i, f] * r
                    user_denom[u, f] += qi[i, f] * est
                    item_num[i, f] += pu[u, f] * r
                    item_denom[i, f] += pu[u, f] * est

            # Update user factors
            for u in trainset.all_users():
                n_ratings = len(trainset.ur[u])
                for f in range(self.n_factors):
                    user_num[u, f] +=n_ratings*auxiliary_pu_num[u,f]
                    user_denom[u, f] += n_ratings * auxiliary_pu_denom[u,f]
                    pu[u, f] *= user_num[u, f] / user_denom[u, f]

            # Update item factors
            for i in trainset.all_items():
                n_ratings = len(trainset.ir[i])
                for f in range(self.n_factors):
                    item_num[i, f] +=n_ratings*auxiliary_qi_num[i,f]
                    item_denom[i, f] += n_ratings * auxiliary_qi_denom[i,f]
                    qi[i, f] *= item_num[i, f] / item_denom[i, f]

            current_rmse = np.sqrt(current_rmse / trainset.n_ratings)
            train_rmse_list.append(current_rmse)
            current_mae = current_mae / trainset.n_ratings
            train_mae_list.append(current_mae)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.train_rmse_list = train_rmse_list
        self.train_mae_list = train_mae_list

    def estimate(self, u, i):

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unkown.')

        return est