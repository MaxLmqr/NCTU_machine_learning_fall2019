import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

# Returns a list of number, picking one randomly inside this list follows a gaussian distribution
# of mean m and variance s
def gaussian_data_generator(m,s):
    # Construct N(m,s)
    A = 10000
    N = [0]*A
    n=10
    for i in range(n):
        N += np.random.uniform(0,1,A)
    N = (N-[n/2]*A)/((1/12**0.5)*n**0.5)
    X = [m]*A + s**0.5*N
    return X



# Polynomial basis linear model data generator
# returns one random x chosen between -1 and 1, and the y corresponding to w*x with w arbitrarly chosen.
def linear_mode_data_generator(n,a,w):
    e = gaussian_data_generator(0,a)
    x = np.random.uniform(-1,1)
    y = sum([w[i]*x**i for i in range(n)]) + random.choice(e)
    return x,y

# Print current mean and variance
def sequential_estimator(m,s):
    i = 1
    x = gaussian_data_generator(m,s)
    mean = random.choice(x)
    variance = 0
    while i<50:
        X = random.choice(x)
        new_mean = (i*mean+X)/(i+1)
        variance = ((i-1)/i)*variance + (X-mean)**2/(i)
        mean = new_mean
        i += 1
        print("Data point: ",X)
        print("Mean : ",mean,"\tVariance : ",variance)

sequential_estimator(3,5)


# Compute predictve mean and variance of a new input X, imagine a vertical gaussian 
def compute_prediction(new_mean_posterior,X_unknown,lambda_new):
    # Compute predictive distribution parameters
    mean_predictive = np.dot(np.transpose(new_mean_posterior),X_unknown)
    variance_predictive = 1/a + np.dot(np.dot(np.transpose(X_unknown),np.linalg.inv(lambda_new)),X_unknown)
    return mean_predictive,variance_predictive

# Return vectors ready to be displayed, a vector Y corresponding to the mean and a variance to compute
# the 2 other curves to plot
def compute_display_prediction(new_mean_posterior,lambda_new,X_truth):
    Y = []
    variance = []
    for x in X_truth:
        X_unknown = [x**i for i in range(n)]
        mean_predictive,variance_predictive = compute_prediction(new_mean_posterior,X_unknown,lambda_new)
        Y.append(mean_predictive)
        variance.append(variance_predictive)
    Y = np.asarray(Y)
    variance = np.asarray(variance)
    return Y,variance


def baysian_linear_regression(b,n,a,w):
    # Ground truth datas
    X_truth = np.linspace(-3,3,1000)
    w_truth = [1,2,3,4]
    Z_truth = np.asarray([X_truth**i for i in range(len(w))])
    Y_truth = np.dot(w,Z_truth)

    # Create empty lists for plots and convergence criteria 
    X_visualization = []
    Y_visualization = []
    X_design = []
    Y_design = []
    mean_posterior = [0]*n
    new_mean_posterior = [1]*n
    i=1

    while sum(abs(np.asarray(new_mean_posterior)-np.array(mean_posterior)))>1e-5:    
        x,y = linear_mode_data_generator(n,1/a,w)
        X_visualization.append(x)
        Y_visualization.append(y)
        X_design.append([x**i for i in range(n)])
        Y_design.append(y)
        X_unknown = np.asarray([x**i for i in range(n)])
        X = np.asarray(X_design)
        Y = np.asarray(Y_design)

        # Record old mean to check convergence
        mean_posterior = new_mean_posterior

        # Compute posterior parameters
        lambda_new = a*np.dot(np.transpose(X),X)+b*np.eye(np.shape(X)[1])
        new_mean_posterior = a*np.dot(np.dot(np.linalg.inv(lambda_new),np.transpose(X)),Y)
        
        # Compute predictive distribution parameters
        mean_predictive,variance_predictive = compute_prediction(new_mean_posterior,X_unknown,lambda_new)
        
        # Print Results
        if i%100==0:
            print("New data point : (",x,",",y,")")
            print("Mean Posterior : ",new_mean_posterior)
            print("\nPosterior Variance : \n",np.linalg.inv(lambda_new))        
            print("\n\nPredictive disribution ~ N(",mean_predictive,",",variance_predictive,")\n\n")
        i+=1

        if i==10:
            Y_ten_incomes,variance_ten_incomes = compute_display_prediction(new_mean_posterior,lambda_new,X_truth)
        if i==50:
            Y_fifty_incomes,variance_fifty_incomes = compute_display_prediction(new_mean_posterior,lambda_new,X_truth)
            

    # Predictive result
    Y_predictive,variance_predictive_list = compute_display_prediction(new_mean_posterior,lambda_new,X_truth)
    



    fig, ax = plt.subplots(nrows=2, ncols=2)
    plt.tight_layout(1.5)
    # First figure
    ax[0][0].set_title('Ground Truth')
    ax[0][0].set_ylim([-15,25])
    ax[0][0].set_xlim([-2,2])
    ax[0][0].plot(X_truth,Y_truth)
    ax[0][0].plot(X_truth,Y_truth+1/a,'r')
    ax[0][0].plot(X_truth,Y_truth-1/a,'r')

    # Second Figure
    ax[0][1].set_title('Predict Result')
    ax[0][1].set_ylim([-15,25])
    ax[0][1].set_xlim([-2,2])
    ax[0][1].plot(X_truth,Y_predictive)
    ax[0][1].plot(X_truth,Y_predictive+variance_predictive_list,'r')
    ax[0][1].plot(X_truth,Y_predictive-variance_predictive_list,'r')
    ax[0][1].plot(X_visualization,Y_visualization, 'b+')

    # Third Figure
    ax[1][0].set_title('After 10 Incomes')
    ax[1][0].set_ylim([-15,22])
    ax[1][0].set_xlim([-1.5,1.5])
    ax[1][0].plot(X_truth,Y_ten_incomes)
    ax[1][0].plot(X_truth,Y_ten_incomes+variance_ten_incomes,'r')
    ax[1][0].plot(X_truth,Y_ten_incomes-variance_ten_incomes,'r')
    ax[1][0].plot(X_visualization[:10],Y[:10],'b+')

    # Fourth Figure
    ax[1][1].set_title('After 50 Incomes')
    ax[1][1].set_ylim([-15,22])
    ax[1][1].set_xlim([-1.5,1.5])
    ax[1][1].plot(X_truth,Y_fifty_incomes)
    ax[1][1].plot(X_truth,Y_fifty_incomes+variance_fifty_incomes,'r')
    ax[1][1].plot(X_truth,Y_fifty_incomes-variance_fifty_incomes,'r')
    ax[1][1].plot(X_visualization[:50],Y[:50],'b+')

    plt.show()


print("Choose case number : ")
print("Case n°1 : \ta=1\tb=1\tn=4\tw=[1,2,3,4]")
print("Case n°2 : \ta=1\tb=100\tn=4\tw=[1,2,3,4]")
print("Case n°3 : \ta=3\tb=1\tn=3\tw=[1,2,3]")
switch = int(input())
if switch==1:
    a=1
    b=1
    n=4
    w=[1,2,3,4]
    baysian_linear_regression(b,n,a,w)
elif switch==2:
    a=1
    b=100
    n=4
    w=[1,2,3,4]
    baysian_linear_regression(b,n,a,w)
elif switch==3:
    a=1/3
    b=1
    n=3
    w=[1,2,3]
    baysian_linear_regression(b,n,a,w)

# N.B : a est une précision, pas une variance a = 1/variance
