import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import *

np.warnings.filterwarnings('ignore')

# Read the input file
data = open('input.data','r').read().splitlines()
for i in range(len(data)):
    data[i] = data[i].split()
    data[i][0] = float(data[i][0])
    data[i][1] = float(data[i][1])
data = np.asarray(data)

# Def kernel function
def kernel(x1,x2,param=[1,1,1]):
    """
        rational quadratic kernel, 3 parameters : sigma,alpha,l
    """
    l,sigma,alpha = param
    temp = cdist(x1,x2,'euclidean')
    return sigma**2*(1+temp/(2*alpha*l**2))**(-alpha)

def posterior_predictive(X_s, X_train, Y_train, param):
    '''  
        Compute posterior from known parameters
    '''
    cov = kernel(X_train, X_train, param) + 1e-8 * np.eye(len(X_train))
    cov_train = kernel(X_train, X_s, param)
    K = kernel(X_s, X_s, param) + 1e-8 * np.eye(len(X_s))
    K_inv = np.linalg.inv(cov)
    
    mu_s = cov_train.T.dot(K_inv).dot(Y_train)

    cov_s = K - cov_train.T.dot(K_inv).dot(cov_train)
    
    return mu_s, cov_s

def log_likelihood(param):
    K = kernel(X_train, X_train,param) + 1e-5*np.eye(len(X_train))
    L = np.linalg.cholesky(K)
    return np.sum(np.log(np.diagonal(L))) + \
            0.5 * Y_train.T.dot(np.linalg.lstsq(L.T, np.linalg.lstsq(L, Y_train)[0])[0]) + \
            0.5 * len(X_train) * np.log(2*np.pi)

def plot_GP(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()   # unfold data
    mu = mu.ravel() # unfold data
    uncertainty = 1.96 * np.sqrt(np.diag(cov))  # Compute the uncertainty based on variance
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()



X_train = data[:,0].reshape(-1,1)
Y_train = data[:,1]

# STEP 1 : Build prior.
# Our training data are between -50 and 50, so we'll take -60 and 60 to build our prior
X = np.linspace(-60,60,50).reshape(-1,1)
# Build the mean vector and covariance matrix of our prior points
mu = np.zeros(X.shape)
cov = kernel(X,X)   # Covariance is based on the kernel

# Compute and plot samples
samples = np.random.multivariate_normal(mu.ravel(),cov,2)

plot_GP(mu,cov,X,samples=samples)

# param = [l,sigma,alpha] are the parameters of the kernel
param=[3,3,3]
mu_posterior,cov_posterior = posterior_predictive(X,X_train,Y_train,param)
samples_post = np.random.multivariate_normal(mu_posterior.ravel(),cov_posterior,2)
plt.figure()
plot_GP(mu_posterior,cov_posterior,X,X_train,Y_train,samples=samples_post)

rez = minimize(log_likelihood,param,bounds=((1e-5,None),(1e-5,None),(1e-5,None)))
param = rez['x']

mu_posterior,cov_posterior = posterior_predictive(X,X_train,Y_train,param)
samples_post = np.random.multivariate_normal(mu_posterior.ravel(),cov_posterior,2)
plt.figure()
plot_GP(mu_posterior,cov_posterior,X,X_train,Y_train,samples=samples_post)



