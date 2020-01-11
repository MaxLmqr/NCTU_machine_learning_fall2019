import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy.spatial.distance import *
from tqdm import tqdm

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

def initialization(data, k):
    """
        Function to initialize the means of the different clusters
        and the classification first step.
    """
    n = data.shape[0]
    initialize_method = "2"
    means = np.random.rand(k,3)*255
    if initialize_method == "1":    # RANDOM INIT
        classif_prec = np.random.randint(k, size=n)
    elif initialize_method == "2":      # Structured init : every 2 columns
        classif_prec = []
        for i in range(n):
            if i % 2 == 1:
                classif_prec.append(0)
            else:
                classif_prec.append(1)
        classif_prec = np.asarray(classif_prec)
    elif initialize_method == "3":  # More accurate method, based on the random mean
        classif_prec = np.zeros(n, dtype=np.int)
        temp = np.zeros(n)
        null_vector = np.zeros([1, 3])
        for i in range(0, n):
            temp[i] = np.linalg.norm(data[i,:] - null_vector[0,:])
        mean_temp = np.mean(temp)
        for i in range(0, n):
            if temp[i] >= mean_temp:
                classif_prec[i] = 0
            else:
                classif_prec[i] = 1
    return means, np.asarray(classif_prec)

def deuxieme_terme(data, kernel_data, classification, data_number, cluster_number, k):
	result = 0
	number_in_cluster = 0
	for i in range(0, data.shape[0]):
		if classification[i] == cluster_number:
			number_in_cluster += 1
	if number_in_cluster == 0:
		number_in_cluster = 1
	for i in range(0, data.shape[0]):
		if classification[i] == cluster_number:
			result += kernel_data[data_number][i]
	return -2 * (result / number_in_cluster)

def troisieme_terme(kernel_data, classification, k):
    """
        Function to compute the third term of the euclidean distance 
        from a point to the center of the different clusters.
    """
    temp = np.zeros(k)
    temp1 = np.zeros(k)
    for i in range(0, classification.shape[0]):
        temp[classification[i]] += 1
	for i in range(0, k):
		for p in range(0, kernel_data.shape[0]):
			for q in range(p + 1, kernel_data.shape[1]):
				if classification[p] == i and classification[q] == i:
					temp1[i] += kernel_data[p,q]
	for i in range(0, k):
		if temp[i] == 0:
			temp[i] = 1
		temp1[i] /= (temp[i] ** 2)
	return temp1

def classifier(data, kernel_data, means, classification):
    """
    Attribute a cluster to every pixel of an image, based on the kernel results
    """
    temp_classification = np.zeros([data.shape[0]], dtype=np.int)
    third_term = troisieme_terme(kernel_data, classification, means.shape[0])
    for i in tqdm(range(0, data.shape[0])):
        temp = np.zeros([means.shape[0]], dtype=np.float32) # temp size: k
        for j in range(0, means.shape[0]):
            temp[j] = deuxieme_terme(data, kernel_data, classification, i, j, means.shape[0]) + third_term[j]
        temp_classification[i] = np.argmin(temp)
    return temp_classification

def nb_erreur(classification, classif_prec):
    """
        Count the number of pixels that have change the cluster between an iteration
    """
    error = 0
    for i in range(0, classification.shape[0]):
        error += np.absolute(classification[i] - classif_prec[i])
    return error

def update(data, means, classification):
    """
        Update the means of the different clusters.
    """
    means = np.zeros(means.shape)
    count = np.zeros(means.shape)
    one_vector = np.ones(means.shape[1])
    for i in range(0, classification.shape[0]):
        means[classification[i]] += data[i]
        count[classification[i]] += one_vector
    return np.true_divide(means, count)
 
def display_clusters(k, data, means, classification, iteration, filename):
  title = "Kernel-K-Means Iteration-" + str(iteration)
  plt.clf()
  plt.suptitle(title)
  plt.imshow(classification.reshape((100,100)),cmap='gray')
  plt.show()


def kkmeans(data, kernel_data, filename):
    k = 2                                               # cluster number
    means, classif_prec = initialization(data, k)       # INITIALIZE everything
    iteration, erreur_prec = 1, 0
    display_clusters(k, data, means, classif_prec, iteration, filename) # display the inital assignment of clusters
    classification = classifier(data, kernel_data, means, classif_prec)
    error = nb_erreur(classification, classif_prec)
    for i in range(50): # Limit to 50 iteration, after that if it did not converge already, it will stop.
        display_clusters(k, data, means, classification, iteration, filename)
        iteration += 1
        classif_prec = classification
        classification = classifier(data, kernel_data, means, classification)
        error = nb_erreur(classification, classif_prec)
        print(error)
        if error == erreur_prec:
            break
        erreur_prec = error
    means = update(data, means, classification) # update the clusters mean to have the final centroids
    display_clusters(k, data, means, classification, iteration, filename)   # display the final assignment.


if __name__ == "__main__":
    filename = "image1.png"
    data = read_input(filename)
    kernel_data = compute_rbf_kernel(data, gamma1=0.1, gamma2=0.01)
    kkmeans(data, kernel_data, filename)