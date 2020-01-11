import struct as st
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

#### SWITCH ####
print("Choose mode : \n0. Discrete Mode\n1. Continuous Mode")
switch = 0      # 0 : discrete mode
                # 1 : continuous mode

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
magic_train_imagesfile = st.unpack('>4B', train_imagesfile.read(4))
magic_train_labels = st.unpack('>4B', train_labels.read(4))
magic_test_imagesfile = st.unpack('>4B', test_imagesfile.read(4))
magic_test_labels = st.unpack('>4B', test_labels.read(4))

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
                                    train_imagesfile.read(nBytesTotal))).reshape((train_nImg, train_nR, train_nC))

nBytesTotalTest = test_nImg*test_nR*test_nC
test_images_array = np.asarray(st.unpack('>'+'B'*nBytesTotalTest,
                                    test_imagesfile.read(nBytesTotalTest))).reshape((test_nImg,test_nR,test_nC))

# Load label data into label_array
label_array = np.asarray(st.unpack('>'+'B'*train_label_nImg,train_labels.read(train_label_nImg)))
test_label_array = np.asarray(st.unpack('>'+'B'*test_label_nImg, test_labels.read(test_label_nImg)))


############## DATA LOADED ###############################################



# Function to display the 'imagination of number'
def imageDisplay(image):
    display_image = image.copy()
    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            if image[i][j] >= 128:
                display_image[i][j] = 1
            else:
                display_image[i][j] = 0
    print(display_image)

# Create a bins array [x1, ...., x32] for an image
def create_bins_array(image):
    bins_array = [0]*32
    for x in np.nditer(image):
        bins_array[int(math.floor(x/8))] += 1
    return bins_array

# Helper function to construct the frequency table
# of xi's value for each i and each label
def add_frequency_table(bins,label):
    for i in range(len(bins)):
        label_bins[label][bins[i]][i] += 1

# Discrete prediction. With an image as parameter, returns the probability of the image to
# belong to each label
def discrete_predicition(image):
    # Create empty array which will contain the probability
    # for the image to belong to the class i
    predictions = [0]* 10
    # Create the array of bins for the unknown image
    unknown_bin = create_bins_array(image)
    for i in range(len(predictions)):
        frequencies = [0] * 32
        for j in range(len(frequencies)):
            # The '+1' is useful to avoid a frequency = 0
            frequencies[j] = np.log((label_bins[i][unknown_bin[j]][j]+1)/(num_labels[i]+1))
        sum_freq = 0
        for x in frequencies:
            sum_freq += x
        predictions[i] = np.log(prior[i])+sum_freq
    normalization = 1/sum(predictions)
    predictions = [normalization*x for x in predictions]
    return predictions
        
# Setup usefull variable to compute probabilities for the discrete mode
label_bins = {i:np.zeros((784,32)) for i in range(10)}
num_labels = np.zeros(10)

# Discrete mode
# label_bins, num_labels, prior = discrete_mode()
if switch == 0:
    # Loop over every image in the training set to generate a table that easily compute frequencies
    for i in range(train_nImg):
        # Create the bins array corresponding to image i
        temp_bins = create_bins_array(images_array[i])
        # Add the new bin array data to the corresponding label
        add_frequency_table(temp_bins,label_array[i])
        num_labels[label_array[i]]+=1

    # Create prior function.
    # It's the ratio of the number of images labelled i with the total number of images.
    prior = [0]*10
    for i in range(10):
        prior[i] = num_labels[i]/train_nImg

    # Loop over test set and print number of correct answers
    correct_answers = 0
    for i in tqdm(range(test_nImg)):
        p = discrete_predicition(test_images_array[i])
        # I took the min because of the log, it makes the value negative and the normalzation
        # makes it positive again but it's now the smallest value
        if (p.index(min(p))) == test_label_array[i]:
            correct_answers += 1
    print("Right answers : ", correct_answers)
    print("Number of images tested : ", test_nImg)
    error_rate = (test_nImg - correct_answers)/test_nImg*100
    print("Error rate : ", error_rate, "%")

# Continuous mode
def gaussian(x,mu,sigma):
    return (1/(2*np.pi*(sigma))**0.5)*np.exp((-1/(2*(sigma)))*(x-mu)**2)

def log_gaussian(x,mu,sigma):
    return -1/2*np.log(2*np.pi*sigma) -(x-mu)**2/(2*sigma)

def continuous_prediction(image,mu,sigma):
    predictions = [0]*10

    for i in range(len(predictions)):
        for j in range(28):
            for k in range(28):
                if sigma[i][j][k] != 0:
                    predictions[i] += log_gaussian(image[j][k],mu[i][j][k],sigma[i][j][k])
                else:
                    predictions[i] += log_gaussian(image[j][k],0,1)
        predictions[i] += np.log(prior[i])
    normalization = 1/sum(predictions)
    predictions = [normalization*x for x in predictions]
    return predictions    
    
if switch == 1:
    mu = np.zeros((10,28,28))
    sigma = np.zeros((10,28,28))
    for i in tqdm(range(train_nImg)):
        num_labels[label_array[i]]+=1
        for k in range(28):
            for h in range(28):
                mu[label_array[i]][k][h] += images_array[i][k][h]

    for i in range(10):
        for j in range(28):
            for k in range(28):
                mu[i][j][k] = mu[i][j][k] / num_labels[i]

    for i in tqdm(range(train_nImg)):
        for k in range(28):
            for h in range(28):
                sigma[label_array[i]][k][h] += (images_array[i][k][h]-mu[label_array[i]][k][h])**2/(num_labels[label_array[i]])
    
    prior = [0]*10
    for i in range(10):
        prior[i] = num_labels[i]/train_nImg

    correct_answers = 0
    for i in tqdm(range(test_nImg)):
        p = continuous_prediction(test_images_array[i],mu,sigma)
        # I took the min because of the log, it makes the value negative and the normalzation
        # makes it positive again but it's now the smallest value
        if p.index(min(p)) == test_label_array[i]:
            correct_answers += 1
    print("Right answers : ", correct_answers)
    print("Number of images tested : ", test_nImg)
    error_rate = (test_nImg - correct_answers)/test_nImg*100
    print("Error rate : ", error_rate, "%")

#### Close FILES ####
train_imagesfile.close()
train_labels.close()
test_imagesfile.close()
test_labels.close()
    
