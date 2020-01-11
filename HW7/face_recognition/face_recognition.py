from scipy.spatial import distance
from skimage import io
import os
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

path = {"Training":"Yale_Face_Database/Training/",
        "Test":"Yale_Face_Database/Testing/"}


def create_data_set(data_path): 
        """ 
        Load the data based on a data path. 
        It returns a 3 dimension vector X containing the images, and a label vector containing the filenames
        Be careful : They are not sorted as in the directory.
        """
        images_path = [ os.path.join(data_path, item)  for item in  os.listdir(data_path) ]
        image_data = []
        image_labels = [item for item in os.listdir(data_path)]

        for i,im_path in enumerate(images_path):
                im = io.imread(im_path,as_gray=True)
                im = resize(im,(im.shape[0]//3,im.shape[1]//3), anti_aliasing=True)
                image_data.append(im)

        X = np.array(image_data).astype(np.float32())
        return X, image_labels


def compute_mean(X,Y):
        mean = np.zeros((15,X.shape[1]*X.shape[2]))
        for i in range(X.shape[0]):
                mean[Y[i]] += X[i].reshape((-1))/9
        global_mean = np.mean(mean,axis=0)
        return mean, global_mean


def knn(X_test, X_train, Y_train,k=20):
    """
    K neareste neighbors algorithm : 
        For every points, we first compute its distance to every other points.
        Then we take K closest neighbors, we check their class, and predict the most appeared class for the new point.
    """
    predictions = []
    for current_test in X_test:        # Loop over all the examples
        distances = []
        for current_train in X_train:
            distances.append(distance.euclidean(current_test.reshape((-1)),current_train.reshape((-1))))
        min_liste = n_small_element(distances,k)
        closest_classes = [Y_train[distances.index(i)] for i in min_liste]
        predictions.append(max(set(closest_classes),key=closest_classes.count))
    return predictions


def n_small_element(L,n):
    """
    Helper function to get the K smallest elements of the distance list created in KNN.
    """
    ele = []
    myList = list(np.copy(L)) # To avoid modifying our list with which we call the function.
    for i in range(n):
        ele.append(min(myList))
        myList.remove(min(myList))
    return ele


# FIRST, load the data and create the class vector Y_train. Same for Y_test in order to get the accuracy.
X_train, label_train = create_data_set(path["Training"])
Y_train = [int(x[7:9])-1 for x in label_train]
X_test, label_test = create_data_set(path["Test"])
Y_test = [int(x[7:9])-1 for x in label_test]
# Then compute the mean and global mean which are used in both PCA and LDA.
mean, global_mean = compute_mean(X_train,Y_train)

"""
    X_train         a 3D_tensor containing the training images
    label_train     the filenames of the training images
    Y_train         a class for the images, starting from 0 to 14
    Same for the test variable.
"""
method = 'PCA'
# Comment or uncomment whether you want to use basic LDA or kernel LDA.
if method=='LDA':
    import LDA_eigenfaces
    # eigenvectors = LDA_eigenfaces.LDA(X_train,Y_train)
    eigenvectors = LDA_eigenfaces.K_LDA(X_train,Y_train,kernel='RQ')

if method=='PCA':
    import PCA_eigenfaces
    eigenvectors = PCA_eigenfaces.PCA(X_train,Y_train,'RQ')

# We now have the eigenvectors to reduce the dimension.
# PS : 25 eigenvectors.
# Now let's project our data into low_dimension space
X_train_reduced = np.matmul(X_train.reshape((X_train.shape[0],-1))-global_mean,eigenvectors)
# Our X_reduced is of dimension (number of training ex) by (number of eigenvectors)
X_test_reduced = np.matmul(X_test.reshape((X_test.shape[0],-1))-global_mean,eigenvectors)

# Then check our results on the testing set.
res = []
ks = [i for i in range(1,26)]
for k in ks:
    pred = knn(X_test_reduced,X_train_reduced,Y_train,k=k)
    correct = 0
    for i in range(len(Y_test)):
        if Y_test[i]==pred[i]:
            correct+=1
    print("Correct : ",correct)
    res.append(correct)
res = np.asarray(res)/30*100
plt.plot(ks,res,'or')
plt.title('Accuracy for different K-nn with Kernel LDA (RQ)')
plt.xlabel('K')
plt.ylabel('Accurcay')

# K=6 seems to be the best to have best results with both LDA and PCA.
"""
    The following snippet is what I've used to get the reconstructed images, and to save them and their corresponding original into a folder.
"""
# random_indexes = np.random.randint(low=0,high=X_train.shape[0],size=10)
# reconstructed_images = []
# for i in random_indexes:
#     projected_image = np.matmul(X_train[i].reshape((-1)),eigenvectors)
#     temp = global_mean.reshape((X_train.shape[1],X_train.shape[2]))+np.matmul(projected_image,eigenvectors.transpose()).reshape((X_train.shape[1],X_train.shape[2]))
#     io.imsave("Results_linear/rec_"+label_train[i],temp)
#     io.imsave("Results_linear/original_"+label_train[i],X_train[i])
#     reconstructed_images.append(temp)