from __future__ import (absolute_import, division, print_function,unicode_literals)
import numpy as np
cimport numpy as np
from six.moves import range
import math

'''
This file define two functions return matrices used by SalaryBenchmark_Matrix Factorization.

Authors: Qingxin Meng(xinmeng320@gmail.com)
Date:    2018/01/18 
'''


def SalaryBenchamrk_SVD_SideMatrix(trainset,user_sim,item_sim,reg_su=5e-5,reg_si=5e-5,reg_t=5e-5,reg_l=5e-5,b=2.0,knn=10):

	# This function return matrices used in SalaryBenchmark_SVD based methods.
	# trainset: trainset contain salaries of 'time-specific job and location specific company' combinations. The format of trainset is pre-defined
	# by Surprise. Surprise is a git_hub package for matrix factorization.
	# user_sim: it is a triangulation matrix describing pair-wise time-specific job similarities.
	# item_sim: it is a triangulation matrix describing pair-wise location-specific company similarities.
	# reg_su: parameter control job similarity related regularization.
	# reg_si: parameter control company similarity related regularization.
	# reg_t: parameter control time similarity related regularization.
	# reg_l: parameter control location similarity related regularization.
	# b: a parameter for time similarity
	# knn: a parameter control the action scope of time and location similarity regularization.
    cdef int num_user,num_item,i,j,delta_t,count_i
    cdef np.ndarray[np.double_t, ndim = 2] su, si, D_su, D_si, t, D_t, l, D_l, aux_pu, aux_qi
    cdef np.ndarray[np.int_t,ndim = 2 ] top_c, top_j
    cdef np.ndarray[np.int_t] row,col, row_inner, col_inner


    num_user = trainset.n_users
    num_item = trainset.n_items

    su = np.zeros((num_user, num_user))
    row_inner = np.repeat(range(num_user), num_user)
    col_inner = np.tile(range(num_user), num_user)
    user_list=map(trainset.to_raw_uid, range(num_user))
    temp_list=np.array(user_list)//5
    row = np.repeat(temp_list, num_user)
    col = np.tile(temp_list, num_user)
    su[row_inner, col_inner] = user_sim[row, col]


    si = np.zeros((num_item, num_item))
    row_inner = np.repeat(range(num_item), num_item)
    col_inner = np.tile(range(num_item), num_item)
    item_list=map(trainset.to_raw_iid, range(num_item))
    temp_list=np.array(item_list)//5
    row = np.repeat(temp_list, num_item)
    col = np.tile(temp_list, num_item)
    si[row_inner, col_inner] = item_sim[row, col]

    D_su = np.zeros((num_user,num_user))
    np.fill_diagonal(D_su,su.sum(1))
    D_si = np.zeros((num_item,num_item))
    np.fill_diagonal(D_si,si.sum(1))

    top_j=np.argsort(-user_sim,axis=1)[:,:knn]
    top_c=np.argsort(-item_sim,axis=1)[:,:knn]



    D_t=np.zeros((num_user,num_user))
    t=np.zeros((num_user,num_user))
    for i in range(num_user):
        for j in range(num_user):
            if (i!=j) and ((user_list[j]//5) in top_j[user_list[i]//5]):
                delta_t=abs((user_list[i]%5)-(user_list[j]%5))
                # if not delta_t <0 :
                t[i,j]=math.exp(-b*delta_t)          
    np.fill_diagonal(t,1)
    np.fill_diagonal(D_t,t.sum(1))
       

       
    avenue=np.array([9240.0,8962.0,7409.0,8315.0,7330.0])
    D_l=np.zeros((num_item,num_item))
    l=np.zeros((num_item,num_item))
    for i in range(num_item):
        for j in range(num_item):
            if (i!=j) and ((item_list[j]//5) in top_c[item_list[i]//5]):
                l[i,j]=1-abs(avenue[(item_list[i]%5)]-avenue[(item_list[j]%5)])/max(avenue[(item_list[i]%5)],avenue[(item_list[j]%5)])
    np.fill_diagonal(l,1)
    np.fill_diagonal(D_l,l.sum(1))





    aux_pu=reg_su*(su-D_su)+reg_t*(t-D_t)
    aux_qi=reg_si*(si-D_si)+reg_l*(l-D_l)


    return aux_pu,aux_qi

# the function below is deprecated.

def SlaryBenchmark_NMF_SideMatrix(trainset,user_sim,item_sim,reg_su=5e-5,reg_si=5e-5,reg_t=5e-5,reg_l=5e-5,b=2.0):
	# This function return matrices used in SalaryBenchmark_NMF based methods.
	# trainset: trainset contain salaries of 'time-specific job and location specific company' combinations. The format of trainset is pre-defined
	# by Surprise. Surprise is a git_hub package for matrix factorization.
	# user_sim: it is a triangulation matrix describing pair-wise time-specific job similarities.
	# item_sim: it is a triangulation matrix describing pair-wise location-specific company similarities.
	# reg_su: parameter control job similarity related regularization.
	# reg_si: parameter control company similarity related regularization.
	# reg_t: parameter control time similarity related regularization.
	# reg_l: parameter control location similarity related regularization.
	# b: a parameter for time similarity
    cdef int num_user,num_item,i,j,delta_t
    cdef np.ndarray[np.double_t, ndim = 2] su, si, D_su, D_si, t, D_t, l, D_l, aux_pu_num, aux_pu_denom, aux_qi_num, aux_qi_denom
    cdef np.ndarray[np.int_t] row,col, row_inner, col_inner

    num_user = trainset.n_users
    num_item = trainset.n_items

    su = np.zeros((num_user, num_user))
    row_inner = np.repeat(range(num_user), num_user)
    col_inner = np.tile(range(num_user), num_user)
    user_list=map(trainset.to_raw_uid, range(num_user))
    temp_list=np.array(user_list)//5
    row = np.repeat(temp_list, num_user)
    col = np.tile(temp_list, num_user)
    su[row_inner, col_inner] = user_sim[row, col]


    si = np.zeros((num_item, num_item))
    row_inner = np.repeat(range(num_item), num_item)
    col_inner = np.tile(range(num_item), num_item)
    item_list=map(trainset.to_raw_iid, range(num_item))
    temp_list=np.array(item_list)//5
    row = np.repeat(temp_list, num_item)
    col = np.tile(temp_list, num_item)
    si[row_inner, col_inner] = item_sim[row, col]

    D_su = np.zeros((num_user,num_user))
    np.fill_diagonal(D_su,su.sum(1))
    D_si = np.zeros((num_item,num_item))
    np.fill_diagonal(D_si,si.sum(1))

    avenue=np.array([9240.0,8962.0,7409.0,8315.0,7330.0])
    D_t=np.zeros((num_user,num_user))
    t=np.zeros((num_user,num_user))
    for i in range(num_user):
        for j in range(num_user):
            if (i!=j) and (user_list[i]//5 == user_list[j]//5):
                delta_t=(user_list[i]%5)-(user_list[j]%5)
                if not delta_t <0 :
                    t[i,j]=math.exp(-b*delta_t)          
    np.fill_diagonal(t,1)
    np.fill_diagonal(t,t.sum(1))
       

    D_l=np.zeros((num_item,num_item))
    l=np.zeros((num_item,num_item))
    for i in range(num_item):
        for j in range(num_item):
            if (i!=j) and (item_list[i]//5==item_list[j]//5):
                l[i,j]=np.divide(avenue[(item_list[i]%5)],avenue[(item_list[j]%5)])
    np.fill_diagonal(l,1)
    np.fill_diagonal(D_l,l.sum(1))

    aux_pu_num=reg_su*su+reg_t*t
    aux_pu_denom=reg_su*D_su+reg_t*D_t
    aux_qi_num=reg_si*si+reg_l*l
    aux_qi_denom=reg_si*D_si+reg_l*D_l


    return aux_pu_num,aux_pu_denom,aux_qi_num,aux_qi_denom

 

