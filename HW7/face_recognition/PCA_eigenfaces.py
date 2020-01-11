from skimage import io
import os
import numpy as np
from skimage.transform import resize
from scipy.spatial import distance


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
                im = resize(im,(im.shape[0]//2,im.shape[1]//2), anti_aliasing=True)
                image_data.append(im)


        X = np.array(image_data).astype('float32')
        return X, image_labels

def compute_mean(X,Y):
        """
        Takes parameter X containing the images as a tensor, and a vector Y which contains the label of each training example.
        Then reshape the images into a vector, to compute two means : One is the mean of each class, and global mean which is the 
        mean of all the training examples.
        """
        mean = np.zeros((15,X.shape[1]*X.shape[2]))
        for i in range(X.shape[0]):
                mean[Y[i]] += X[i].reshape((-1))/9
        global_mean = np.mean(mean,axis=0)
        return mean, global_mean

def get_feature_vectors(covariance):
        """
        Return the feature vectors associated with the 25th highest eigenvalues
        """
        eigen_values, eigen_vectors = np.linalg.eigh(covariance)
        idx = eigen_values.argsort()[::-1]
        return eigen_vectors[:,idx][:,:25]


def rbf_kernel(x,y,gamma):
        temp = distance.cdist(x,y,'euclidean')
        return np.exp(-gamma*temp)
def linear_kernel(x,y,c):
        res = x@y.T + c
        return res
def rq_kernel(x1,x2,param=[1,1,1]):
        """
                rational quadratic kernel, 3 parameters : sigma,alpha,l
        """
        l,sigma,alpha = param
        temp = distance.cdist(x1,x2,'euclidean')
        return sigma**2*(1+temp/(2*alpha*l**2))**(-alpha)

def PCA(X_train,Y_train,kernel='None'):
        """
        This function is made to be used when the file is imported in another function.
        Return the eigenvectors from PCA, right now 25 eigenvectors.
        CAn be modified by changing the above function.
        """
        print("Computing mean and global mean ...")
        mean, global_mean = compute_mean(X_train,Y_train)
        print("Done.")
        print("Computing covariance matrix ...")
        x = X_train.reshape((X_train.shape[0],-1))-global_mean
        if kernel=='linear':
                covariance = linear_kernel(x.T,x.T,1)
        if kernel=='RQ':
                covariance = rq_kernel(x.T,x.T)
        if kernel=='RBF':
                covariance =  rbf_kernel(x.T,x.T,0.1)
        if kernel =='None':
                covariance = np.cov((X_train.reshape((X_train.shape[0],-1))-global_mean).transpose()) # 11155 by 11155
        print("Done.")
        print("Computing feature vectors ...")
        feature_vectors = get_feature_vectors(covariance) # 11155 by 25
        print("Done.")
        return feature_vectors

if __name__=="__main__":
        ###################  PCA ##############################
        X_train, label_train = create_data_set(path["Training"])
        Y_train = [int(x[7:9])-1 for x in label_train]
        X_test, label_test = create_data_set(path["Test"])
        mean, global_mean = compute_mean(X_train,Y_train)
        covariance = np.cov((X_train.reshape((X_train.shape[0],-1))-global_mean).transpose()) # 11155 by 11155
        feature_vectors = get_feature_vectors(covariance) # 11155 by 25
        random_indexes = np.random.randint(low=0,high=X_train.shape[0],size=10)
        reconstructed_images = []
        for i in random_indexes:
                projected_image = np.matmul(X_train[i].reshape((-1)),feature_vectors)
                temp = global_mean.reshape((115,97))+np.matmul(projected_image,feature_vectors.transpose()).reshape((115,97))
                io.imsave("Results/rec_"+label_train[i],temp)
                io.imsave("Results/original_"+label_train[i],X_train[i])
                reconstructed_images.append(temp)



 