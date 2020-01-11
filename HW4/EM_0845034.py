import struct as st
import numpy as np
import math
from tqdm import tqdm 
import matplotlib.pyplot as plt
import warnings
from munkres import Munkres, print_matrix

filename = {'images': 'train-images.idx3-ubyte',
            'labels': 'train-labels.idx1-ubyte',
            'test_images': 't10k-images.idx3-ubyte',
            'test_labels': 't10k-labels.idx1-ubyte'}

# Read in binary mode the different files
train_imagesfile = open(filename['images'], 'rb')
train_labels = open(filename['labels'], 'rb')
test_imagesfile = open(filename['test_images'], 'rb')
test_labels = open(filename['test_labels'], 'rb')

# Go to the offset correponding to the magic number
train_imagesfile.seek(0)
train_labels.seek(0)
test_imagesfile.seek(0)
test_labels.seek(0)
# Unpack the magic number
st.unpack('>4B', train_imagesfile.read(4))
st.unpack('>4B', train_labels.read(4))
st.unpack('>4B', test_imagesfile.read(4))
st.unpack('>4B', test_labels.read(4))

# Number of images, rows and column in train and test set
train_nImg = st.unpack('>I', train_imagesfile.read(4))[0]
train_nR = st.unpack('>I', train_imagesfile.read(4))[0]
train_nC = st.unpack('>I', train_imagesfile.read(4))[0]

test_nImg = st.unpack('>I', test_imagesfile.read(4))[0]
test_nR = st.unpack('>I', test_imagesfile.read(4))[0]
test_nC = st.unpack('>I', test_imagesfile.read(4))[0]

# Number of image in train_label set and test_label set
train_label_nImg = st.unpack('>I', train_labels.read(4))[0]
test_label_nImg = st.unpack('>I', test_labels.read(4))[0]


# Load images data from the file into an images_array
nBytesTotal = train_nImg*train_nR*train_nC
images_array = np.asarray(st.unpack('>'+'B'*nBytesTotal,
    train_imagesfile.read(nBytesTotal))).reshape((train_nImg,train_nR,train_nC))

nBytesTotalTest = test_nImg*test_nR*test_nC
test_images_array = np.asarray(st.unpack('>'+'B'*nBytesTotalTest,
    test_imagesfile.read(nBytesTotalTest))).reshape((test_nImg,test_nR,test_nC))

# Load label data into label_array
label_array = np.asarray(st.unpack('>'+'B'*train_label_nImg,train_labels.read(train_label_nImg)))
test_label_array = np.asarray(st.unpack('>'+'B'*test_label_nImg, test_labels.read(test_label_nImg)))

train_imagesfile.close()
train_labels.close()
test_imagesfile.close()
test_labels.close()

############## DATA LOADED ###############################################

# def p1(training_images,parameters, mixing_coefficient,K):
#     res = np.empty((training_images.shape[0],K))
#     for k in range(K):
#         temp = 1
#         temp *= mixing_coefficient[k]
#         temp2 = parameters[k]

def loglikelihood(mixing_coefficient,training_images,parameters):
    temp = training_images@np.log(parameters+epsilon)
    temp2 = (1-training_images)@np.log(1-parameters+epsilon)
    result=-(np.log(mixing_coefficient.T)+temp+temp2) # Dim : 60000x10
    return np.sum(result)


def expectationStep(mixing_coefficient, training_images, parameters):
    """
        Do the expectation step of the EM algorithm. This compute the probability of each
        training image to belong to each class, in a nbTrainingImage x nbClass matrix
    """
    warnings.filterwarnings("ignore")
    temp = training_images@np.log(parameters+(parameters==0))
    temp2 = (1-training_images)@np.log(1-parameters+((1-parameters)==0))
    result=(np.log(mixing_coefficient.T)+temp+temp2) # Dim : 60000x10
    ll = -np.sum(result)
    result = np.exp(result)
    z = result/np.sum(result,axis=1).reshape((-1,1))
    z[np.isnan(z)] = 0
    z[np.isneginf(z)] = 0

    warnings.resetwarnings()
    return z, ll


def maximizationStep(z,images,parameters,mixing_coefficient):
    """
        Update the parameters of each pixel, and the mixing coefficients of each class
    """
    N = z.sum(axis=0)
    parameters = (1/N)*(images.T@z)
    mixing_coefficient = N/N.sum()
    return parameters,mixing_coefficient


def guessLabel(z, label, K):
    """
        Function to determine which label corresponds to which class. Initially the EM algorithm just
        build 10 clusters from the data. Then we want to know to which label (from 0 to 9) corresponds
        to which cluster. Uses the hungarian algorithm to assign.
    """
    bible = np.zeros((K,K))
    for i in range(z.shape[0]):
        y = list(z[i])
        bible[label[i]][y.index(max(y))]+=1
    bible_minimization = np.max(bible,axis=1)-bible
    bible_list = [list(x) for x in bible_minimization]
    # resultat = {i:bible[i].index(max(bible[i])) for i in range(K)}
    m = Munkres()
    resultat = m.compute(bible_list)
    return resultat, bible

def buildConfusionmatrix(re,bible,K):
    """
        Build the confusion matrix of each class
    """
    confusions = {i:{} for i in range(K)}
    total = np.sum(bible)
    for j in range(K):
        TP = int(bible[j][re[j][1]])
        FP = int(sum([bible[i][re[j][1]] for i in range(K) if i!=re[j][1] ]))
        TN = int(sum([sum(bible[k]) for k in range(K) if k!=j]) - sum([bible[i][re[j][1]] for i in range(K) if i!=re[j][1] ]))
        FN = int(sum([bible[j][k] for k in range(K) if k!= re[j][1]]))
        confusions[j] = {'TP':TP, 'FP': FP, 'TN':TN, 'FN':FN}
    return confusions

def displayConfusion(confusion, class_n):
    """
        Function to display the confusion matrix with other indicator such as sensitivity, specificity
    """
    sensitivity = confusion['TP']/(confusion['TP']+confusion['FN'])
    specificity = confusion['TN']/(confusion['TN']+confusion['FP'])
    print("Confusion Matrix : \n")
    print("\t\t Predict ",class_n,"\t Predict not ",class_n)
    print("Is", class_n, "\t\t", confusion['TP'], "\t\t\t", confusion['FN'])
    print("Is not ",class_n, "\t", confusion['FP'], "\t\t\t", confusion['TN'])
    print("Sensitivity : ",sensitivity)
    print("Specificity : ",specificity)
    print("\n\n")

def computeError(bible,re,K):
    """
        Compute the error rate of the entire set
    """
    mistakes = 0
    for i in range(K):
        mistakes+= sum([bible[i][j] for j in range(K) if j!=re[i][1]])
    return mistakes/np.sum(bible)

############################## PARAMETERS RANDOM DEFINITION ################################
K = 10  # Number of classes
bin_size = 784  # Size of each training element
epsilon = 1e-5

parameters = np.random.uniform(0.25,0.75,(784,10))        # Taille 784x10
parameters = parameters/np.sum(parameters,axis=1).reshape((-1,1))
mixing_coefficient = (1/10)*np.ones((1,10)).T             # Taille 10x1

# Transforme directement toutes les images en des vecteurs dont les composantes sont binaires.
training_images = (images_array>127).astype(int)    # 60000x784
training_images = training_images.reshape((train_nImg,-1))

converged = False
indic = 0
i=0
while not converged:
    z,ll = expectationStep(mixing_coefficient,training_images,parameters)
    parameters,mixing_coefficient = maximizationStep(z,training_images,parameters,mixing_coefficient)
    if abs(ll-indic)<(ll/100):
        converged = True
    if i%10==0:
        print("Ite n°",i)
        print(ll)
    indic = ll
    i+=1
re,bible = guessLabel(z,label_array,K)
confusions_matrix = buildConfusionmatrix(re, bible, K)
print(re, "\n\n")
for i in range(K):
    displayConfusion(confusions_matrix[i],i)
    
err = computeError(bible,re,K)
print("Total iteration to converge : ",i)
print("Total error rate :",err)

# imaginationnum = {i:parameters[:,i].reshape((28,28)) for i in range(10)}
# fig,ax = plt.subplots(2,5)
# fig.tight_layout()
# for j in range(10):
#     ax[j//5][j%5].imshow(imaginationnum[j])