import numpy as np
import random
import matplotlib.pyplot as plt
import sys

def gaussian_data_generator(m,s):
    """
        Function to generate data from a gaussian distribution with mean m and variance s
    """
    # Construct N(m,s)
    A = 10000
    N = [0]*A
    n=10
    for i in range(n):
        N += np.random.uniform(0,1,A)
    N = (N-[n/2]*A)/((1/12**0.5)*n**0.5)
    X = [m]*A + s**0.5*N
    return X

def gen_data(n,mx,vx,my,vy):
    """
        Uses the gaussian data generator to generate random points (x,y), with x and y belonging to 2 different
        gaussian distribution
    """
    Nx = gaussian_data_generator(mx,vx)
    Ny = gaussian_data_generator(my,vy)

    D = []
    for i in range(n):
        D.append((random.choice(Nx),random.choice(Ny)))

    return D

def cost_function(x,y):
    """
        Compute the cost with cross entropy function for a classification problem.
    """
    err = 0
    epsilon = 1e-5
    n = len(x)
    for i in range(n):
        if y[i]==1:
            err += -np.log(x[i]+epsilon)
        else:
            err += -np.log(1-x[i]+epsilon)
    return err


N=50
cas = 1
# CAS 1
if cas ==1:
    mx1, my1 = 1,1
    mx2, my2 = 10,10
    vx1, vy1 = 2,2
    vx2, vy2 = 2,2

# CAS 2
if cas == 2:
    mx1, my1 = 1,1
    mx2, my2 = 3,3
    vx1, vy1 = 2,2
    vx2, vy2 = 4,4

# Generate two sets of data with the corresponding means and variance for x's and y's
D1 = gen_data(N,mx1,vx1,my1,vy1)
D2 = gen_data(N,mx2,vx2,my2,vy2)
# Generate the label associated with the sets of data we just generated
label = np.asarray([0 for x in range(N)] + [1 for x in range(N)])
label = label.reshape((1,2*N))

# Data array
A = np.ones((3,2*N))
X = [x[0] for x in D1] + [x[0] for x in D2]
Y = [x[1] for x in D1] + [x[1] for x in D2]
A[1,:] = X
A[2,:] = Y

# A is the design matrix

W = np.random.randn(3,1)*1e-3

# W is the coefficient matrix

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_grad(output, label, A):
    n = len(label)
    G = A@(output-label).T/n
    return G

def update_W(W,grad, alpha, method):
    if method=='gradient':
        W  = W - alpha*grad
    else:
        W = W - alpha*grad[1]@grad[0]
    return W

def compute_hessian(output,A):
    R = np.diag([y*(1-y) for y in output])
    H = A@R@A.T
    return H


def logistic_regression(A,W,label, method='gradient'):
    """
        Logistic regression using gradient method or newton's method
    """
    learning_rate = 0.001
    costs = []
    #randomize here because we will call several time the logistic regression
    W = np.random.randn(3,1)*1
    i=0
    conv = 1
    while conv>1e-5:
        
        # propagation
        pre_output  = W.T@A
        output = sigmoid(W.T@A)     # Number between 0 and 1, size 1x100

        # compute the actual cost
        cost = cost_function(output[0],label[0])
        costs.append(cost)
        if i>1:
            conv = np.abs(costs[-1]-costs[-2])

        # compute the gradient of w
        gradW = compute_grad(output, label, A)

        if method == 'newton':
            # Compute hessian matrix
            hessW = compute_hessian(output[0],A)

            # If hessian matrix is invertible, then update with newton method
            if np.linalg.cond(hessW) < 1/sys.float_info.epsilon:
                Hinv = np.linalg.inv(hessW)
                W = update_W(W,[gradW,Hinv],learning_rate,'newton')
            # Else we use gradient method
            else:
                # update W
                W = update_W(W,gradW, learning_rate,'gradient')
        else:
            W = update_W(W,gradW, learning_rate,'gradient')

        i+=1
        # if i%1000 == 0:
        #     print("Iteration n°",i," : ",cost)
    print("Nb iteration : ",i)
    return costs,W

# Results of gradient descent
costs_grad, Wout = logistic_regression(A,W,label,'gradient')
output = sigmoid(Wout.T@A).T # Array size 1x100 with numbers close to 0 or 1
predictions = [1 if output[i]>0.5 else 0 for i in range(len(output))]
index1,index0 = [],[]
grad_confusion_matrix = {'TP':0, 'FP':0, 'TN': 0, 'FN': 0}
for i in range(len(predictions)):
    if predictions[i] == 1:
        index1.append(i)
        if label.T[i] == 1:
            grad_confusion_matrix['TP'] += 1
        else:
            grad_confusion_matrix['FP'] += 1
    else:
        index0.append(i)
        if label.T[i] == 0:
            grad_confusion_matrix['TN'] += 1
        else:
            grad_confusion_matrix['FN'] += 1
sensitivity = grad_confusion_matrix['TP']/(grad_confusion_matrix['TP']+grad_confusion_matrix['FN'])
specificity = grad_confusion_matrix['TN']/(grad_confusion_matrix['TN']+grad_confusion_matrix['FP'])
print("Gradient Method : \n")
print("W : ",Wout.T)
print("Confusion Matrix : \n")
print("\t\t Predict Cluster 1 \t Predict Cluster 2")
print("Is Cluster 1 \t\t", grad_confusion_matrix['TP'], "\t\t\t", grad_confusion_matrix['FN'])
print("Is Cluster 2 \t\t", grad_confusion_matrix['FP'], "\t\t\t", grad_confusion_matrix['TN'])
print("Sensitivity : ",sensitivity)
print("Specificity : ",specificity)
print("\n\n")

# Results of Newton's method
costs_newton, Wout = logistic_regression(A,W,label,'newton')
output = sigmoid(Wout.T@A).T # Array size 1x100 with numbers close to 0 or 1
predictions = [1 if output[i]>0.5 else 0 for i in range(len(output))]
index1_newton,index0_newton = [],[]
newton_confusion_matrix = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
for i in range(len(predictions)):
    if predictions[i] == 1:
        index1_newton.append(i)
        if label.T[i] == 1:
            newton_confusion_matrix['TP'] += 1
        else:
            newton_confusion_matrix['FP'] += 1
    else:
        index0_newton.append(i)

        if label.T[i] == 0:
            newton_confusion_matrix['TN'] += 1
        else:
            newton_confusion_matrix['FN'] += 1
sensitivity = newton_confusion_matrix['TP']/(newton_confusion_matrix['TP']+newton_confusion_matrix['FN'])
specificity = newton_confusion_matrix['TN']/(newton_confusion_matrix['TN']+newton_confusion_matrix['FP'])
print("Newton's Method : \n")
print("W : ",Wout.T)
print("Confusion Matrix : \n")
print("\t\t Predict Cluster 1 \t Predict Cluster 2")
print("Is Cluster 1 \t\t", newton_confusion_matrix['TP'], "\t\t\t", newton_confusion_matrix['FN'])
print("Is Cluster 2 \t\t", newton_confusion_matrix['FP'], "\t\t\t", newton_confusion_matrix['TN'])
print("Sensitivity : ",sensitivity)
print("Specificity : ",specificity)



fig, axs = plt.subplots(2,3)
fig.tight_layout() 
# Truth
axs[0][0].plot([x[0] for x in D1], [x[1] for x in D1], 'ro')
axs[0][0].plot([x[0] for x in D2], [x[1] for x in D2], 'bo')
axs[0][0].set_title('Ground Truth')

D = D1+D2
# Gradient Descent
axs[0][1].plot([D[i][0] for i in index0], [D[i][1] for i in index0], 'ro')
axs[0][1].plot([D[i][0] for i in index1], [D[i][1] for i in index1], 'bo')
axs[0][1].set_title('Gradient Descent')


# Newton Method
axs[0][2].plot([D[i][0] for i in index0_newton], [D[i][1] for i in index0_newton], 'ro')
axs[0][2].plot([D[i][0] for i in index1_newton], [D[i][1] for i in index1_newton], 'bo')
axs[0][2].set_title('Newton\'s Method')

axs[1][0].set_visible(False)

# Display cost function
X_cost = [i for i in range(len(costs_grad))]
axs[1][1].plot(X_cost,costs_grad)
axs[1][1].set_title('Cost function for Gradient Descent')

# Display cost function for newton's method
X_cost = [i for i in range(len(costs_newton))]
axs[1][2].plot(X_cost,costs_newton)
axs[1][2].set_title('Cost function for Newton\'s method')
plt.show()




