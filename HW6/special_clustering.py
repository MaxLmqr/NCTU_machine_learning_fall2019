import matplotlib.pyplot as plt
import numpy as np
import time
import imageio
from scipy.spatial.distance import *

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


def read_input(image):
    im = imageio.imread(image)
    im=im.reshape((-1,3))
    return im

def compute_rbf_kernel(image,gamma1,gamma2):
    kernel = np.zeros((image.shape[0],image.shape[0]))
    temp=[]
    for i in range(100):
      for j in range(100):
        temp.append([i,j])
    temp = np.asarray(temp)
    temp2 = np.exp(-gamma1*cdist(temp,temp))
    color = cdist(image,image)
    kernel = np.multiply(temp2,np.exp(-gamma2*color))
    return kernel

def initialization(k, data):
	initialize_method = "k-means++"
	print("Initialization Method: {}".format(initialize_method))
	previous_classification = np.zeros([1500], np.int)
	if initialize_method == "randomly generate":
		means = np.random.rand(k, 2)
	elif initialize_method == "randomly assign":
		temp = np.random.randint(low=0, high=data.shape[0], size=k)
		means = np.zeros([k, 2], dtype=np.float32)
		for i in range(0, k):
			means[i,:] = data[temp[i],:]
	elif initialize_method == "k-means++":
		means = np.zeros([k, 2], dtype=np.float32)
		temp = np.random.randint(low=0, high=data.shape[0], size=1, dtype=np.int)
		means[0,:] = data[temp,:]
		temp = np.zeros(data.shape[0], dtype=np.float32)
		for i in range(0, data.shape[0]):
			temp[i] = np.linalg.norm(data[i,:] - means[0,:])
		temp = temp / temp.sum()
		temp = np.random.choice(data.shape[0], 1, p=temp)
		means[1,:] = data[temp,:]
	return means, np.asarray(previous_classification), 1 # 1 for iteration


def classify(data, means):
	classification = np.zeros([data.shape[0]], dtype=int)
	for i in range(0, data.shape[0]):
		temp = np.zeros([means.shape[0]], dtype=np.float32) # temp size: k
		for j in range(0, means.shape[0]):
			delta = abs(np.subtract(data[i,:], means[j,:]))
			temp[j] = np.square(delta).sum(axis=0)
		classification[i] = np.argmin(temp)
	return classification

def calculate_error(classification, previous_classification):
	error = 0
	for i in range(0, classification.shape[0]):
		error += np.absolute(classification[i] - previous_classification[i])
	return error

def update(data, means, classification):
	means = np.zeros(means.shape, dtype=np.float32)
	count = np.zeros(means.shape, dtype=np.int)
	one = np.ones(means.shape[1], dtype=np.int)
	for i in range(0, data.shape[0]):
		means[classification[i]] += data[i]
		count[classification[i]] += one
	for i in range(0, means.shape[0]):
		if count[i][0] == 0:
			count[i] += one
	return np.true_divide(means, count)

def draw(k, data, classification, iteration, dataset):
	color = iter(plt.cm.rainbow(np.linspace(0, 1, k)))
	plt.clf()
	plt.imshow(classification.reshape((100,100)),cmap='gray')
	title = "Spectral-Clustering Iteration-" + str(iteration)
	plt.suptitle(title)
	plt.show()

def draw_eigen_space(k, data, classification):
	color = iter(plt.cm.rainbow(np.linspace(0, 1, k)))
	plt.clf()
	title = "Spectral-Clustering in Eigen-Space"
	plt.suptitle(title)
	for i in range(0, k):
		col = next(color)
		for j in range(0, data.shape[0]):
			if classification[j] == i:
				plt.scatter(data[j][0], data[j][1], s=8, c=col)
	plt.savefig("./Screenshots/Spectral-Clustering/moon/" + title + ".png")
	plt.show()

def k_means(k, raw_data, data):
	# k is the number of cluster
	means, previous_classification, iteration = initialization(k, data) # means size: k*2 previous_classification: 3000
	classification = classify(data, means) # classification: 3000
	error = calculate_error(classification, previous_classification)
	draw(k, raw_data, classification, iteration, dataset)
	while(True):
		iteration += 1
		means = update(data, means, classification)
		previous_classification = classification
		classification = classify(data, means)
		error = calculate_error(classification, previous_classification)
		draw(k, raw_data, classification, iteration, dataset)
		print(error)
		if error < 5:
			break
	draw(k, raw_data, classification, iteration, dataset)
	print("Elapsed Time: {}".format(time.time() - start_time))
	print("Iterations to coverged: {}".format(str(iteration)))
	return classification

if __name__ == "__main__":
    start_time = time.time()
    k = 2
    dataset = "image1.png"
    print("Dataset: {}".format(dataset))
    data = read_input(dataset)
    Weight = compute_rbf_kernel(data,0.1,0.01) # Weight size: 3000 * 3000
    Degree = np.diag(np.sum(Weight, axis=1))
    print("Compute Laplacien...")
    Laplacian = Degree - Weight
    print("Compute eigenvectors and eigenvalues...")
    eigen_values, eigen_vectors = np.linalg.eig(Laplacian)
    idx = np.argsort(eigen_values)
    eigen_vectors = eigen_vectors[:,idx]
    U = (eigen_vectors[:,:k+1])[:,1:]
    print('Start classification...')
    classification = k_means(k, data, U)