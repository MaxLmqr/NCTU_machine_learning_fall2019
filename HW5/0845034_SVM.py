import numpy as np
import sys
sys.path.insert(0, 'libsvm-3.24/python')
sys.path.insert(0, 'libsvm-3.24/tools')
from svmutil import *
import grid
from scipy.spatial.distance import *


Y_train = open('Y_train.csv','r').read().splitlines()
for i in range(len(Y_train)):
    Y_train[i] = float(Y_train[i])
Y_train=np.asarray(Y_train)

X_train = open('X_train.csv','r').read().splitlines()
for i in range(len(X_train)):
    X_train[i] = X_train[i].split(',')
    for j in range(len(X_train[i])):
        X_train[i][j] = float(X_train[i][j])
X_train = np.asarray(X_train)


Y_test = open('Y_test.csv','r').read().splitlines()
for i in range(len(Y_test)):
    Y_test[i] = float(Y_test[i])
Y_test=np.asarray(Y_test)

X_test = open('X_test.csv','r').read().splitlines()
for i in range(len(X_test)):
    X_test[i] = X_test[i].split(',')
    for j in range(len(X_test[i])):
        X_test[i][j] = float(X_test[i][j])
X_test = np.asarray(X_test)


# cs = [1e-4,1e-3,1e-2,1e-1,1e0]
# for c in cs:
#     prob = svm_problem(Y_train,X_train)
#     param = svm_parameter('-t 0 -c '+str(c))
#     m_linear = svm_train(prob,param)
#     print("Linear Model result with c = {}:".format(c))
#     p_label, p_acc, p_val = svm_predict(Y_test,X_test, m_linear)
#     # p_acc contains : accuracy, mean squared error and squared correlation coefficient
# deg = [2,3,4,5]
# cs = [10,100,1000,10000]
# for d in deg:
#     for c in cs:
#         param = svm_parameter('-t 1 -c '+str(c)+' -d '+str(d))
#         m_poly = svm_train(prob,param)
#         print("Polynomial Model result with c = {} and d = {} :".format(c,d))
#         p_label, p_acc, p_val = svm_predict(Y_test,X_test, m_poly)

# gammas = [1e-3,1e-2,0.1]
# cs = [1e1,1e2,1e3,1e4]
# for g in gammas:
#     for c in cs:
#         param = svm_parameter('-t 2 -c '+str(c)+' -g '+str(g))
#         m_rbf = svm_train(prob,param)
#         print("RBF Model result for c = {} and g = {} :".format(c,g))
#         p_label, p_acc, p_val = svm_predict(Y_test,X_test, m_rbf)



def linear_kernel(x,y,c):
    res = x@y.T + c
    return res

def rbf_kernel(x,y,gamma):
    temp = cdist(x,y,'euclidean')
    return np.exp(-gamma*temp)


cs=[1e-4,1e-2,1,1e2]
gammas = [1e-3,1e-2,0.1]

for c in cs:
    for gamma in gammas:
        K = linear_kernel(X_train,X_train,c)
        K += rbf_kernel(X_train,X_train,gamma)
        K/=2
        KK = linear_kernel(X_test,X_test,c)
        KK += rbf_kernel(X_test,X_test,gamma)
        K/=2
        model = svm_train(Y_train,K,'-t 4')
        e,f,g = svm_predict(Y_test,KK,model)

