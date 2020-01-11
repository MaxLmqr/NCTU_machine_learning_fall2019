import struct as st
import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from math import factorial as fac


def beta(x,a,b):
    return x**(a-1)*(1-x)**(b-1)*(gamma(a+b))/(gamma(a)*gamma(b))

def mle_binomial(data):
    N = len(data)
    X = 0
    for x in data:
        if x=='1':
            X+=1
    p = X/N
    return X,N,p

def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


def binomial_likelihood(X,n,p):
    return binomial(n,X)*p**(X)*(1-p)**(n-X)


def beta_posterior(m,N,a,b):
    new_a = m+a
    new_b = N-m+b
    return new_a, new_b

def main():
    i=1
    a=0
    b=0
    # Read datas
    file = open('testfile.txt','r')
    case = file.read().splitlines()
    for x in case:
        print('Case ',i,' : ',int(x))
        i +=1
        X,N,p = mle_binomial(x)
        print('Likelihood : ',binomial_likelihood(X,N,p))
        print('Beta prior: a = ',a,' b = ',b)
        a,b = beta_posterior(X,N,a,b)
        print('Beta posterior: a = ',a,' b = ',b)



if __name__ == '__main__':
    main()