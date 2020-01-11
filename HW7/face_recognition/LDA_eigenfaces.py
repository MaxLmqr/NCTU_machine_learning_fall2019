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
                im = resize(im,(im.shape[0]//3,im.shape[1]//3), anti_aliasing=True)
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

def get_feature_vectors(w_c_mat, b_w_mat):
        """
        Takes as parameter the within class matrix and the between clas matrix, and the compute the eigenvectors related to the 
        right equation. 
        Return the feature vectors associated with the 25th highest eigenvalues
        """
        eigen_values, eigen_vectors = np.linalg.eigh(np.matmul(np.linalg.pinv(w_c_mat),b_w_mat))
        idx = eigen_values.argsort()[::-1]
        return eigen_vectors[:,idx][:,:25]

def within_class_matrix(x,y, mean):
        """
        Compute the within class scatter matrix.
        """
        w_c_mat = np.zeros([x.shape[1]*x.shape[2], x.shape[1]*x.shape[2]], dtype=np.float32)
        for i in range(0, x.shape[0]):
                temp = np.subtract(x[i].reshape((-1)), mean[y[i]])
                temp = temp.reshape((-1,1))
                w_c_mat += np.matmul(temp, temp.transpose())
        return w_c_mat

def between_class_matrix(mean, global_mean):
        """
        Compute the between class scatter matrix
        """
        b_c_mat = np.zeros([mean.shape[1], mean.shape[1]], dtype=np.float32)
        for i in range(0, 9):
                temp = np.subtract(mean[i], global_mean).reshape(mean.shape[1], 1)
                b_c_mat += np.matmul(temp, temp.transpose())
                b_c_mat *= 9
        return b_c_mat

def LDA(X_train,Y_train):
        """
        Return the feature eigenvectors, right now it is 25 eigenvectors. Can be changed by
        modifying the get_feature_vector function.
        """
        print('Computing means...')
        mean, global_mean = compute_mean(X_train,Y_train)
        print('Done.')
        temp = X_train.reshape((X_train.shape[0],-1))-global_mean
        X_train = temp.reshape((X_train.shape))
        print('Computing within class matrix ...')
        w_c_mat = within_class_matrix(x=X_train,y=Y_train,mean=mean)
        print('Done.')
        print('Computing between class matrix ...')
        b_c_mat = between_class_matrix(mean=mean, global_mean=global_mean)
        print('Done.')
        print('Computing feature vectors ...')
        feature_vectors = get_feature_vectors(w_c_mat,b_c_mat)
        print('Done.')
        return feature_vectors

# Define a few kernels

def rbf_kernel(x,y,gamma):
        temp = distance.cdist(x,y)
        return np.exp(-gamma*temp)

def linear_kernel(x,y,c):
        res = x@y.T + c
        return res

def rq_kernel(x1,x2,param=[1,1,1]):
        """
        rational quadratic kernel, 3 parameters : sigma,alpha,l
        """
        l,sigma,alpha = param
        temp = distance.cdist(x1,x2)
        return sigma**2*(1+temp/(2*alpha*l**2))**(-alpha)


def K_LDA(X_train,Y_train,kernel='RBF'):
        """
                Kernel LDA, instead of the basic within class matrix and the between class matrix,
                we compute two corresponding matrix based on the kernel we first compute.
        """
        print('Computing means...')
        mean, global_mean = compute_mean(X_train,Y_train)
        print('Done.')
        x = X_train.reshape((X_train.shape[0],-1))-global_mean
        if kernel=='RBF':
                K = rbf_kernel(x.T,x.T,gamma=0.1)
        if kernel=='linear':
                K = linear_kernel(x.T,x.T,1)
        if kernel == 'RQ':
                K = rq_kernel(x.T,x.T)

        index = {i:[] for i in range(15)}
        for i in range(len(Y_train)):
                index[Y_train[i]].append(i)

        Ks = {i:[] for i in range(15)}
        for i in K:
                for key,val in index.items():
                        temp = []
                        for h in val:
                                temp.append(i[h])
                        Ks[key].append(np.array(temp))
        for key in Ks.keys():
                Ks[key] = np.asarray(Ks[key])

        A = np.identity(9) - ((1/float(9)) * np.ones((9,9)))

        print('Compute within class matrix ...')
        # calculate within class scatter matrix N
        N = np.zeros(K.shape)
        for value in Ks.values():
                temp = np.dot(A,value.T)
                temp = np.dot(value, temp)
                N += temp
        print('Done.')

        print('Compute between class matrix ...')
        # calculate M1 and M2
        M = {i:[] for i in range(15)}
        for key,value in Ks.items():
                for i in range(len(value)):
                        M[key].append(np.sum(value[i])/float(9))
        for key in M:
                M[key] = np.asarray(M[key])

        Mstar = []
        for i in range(5005):
                Mstar.append(np.sum(value[i])/float(9*15))
        Mstar = np.asarray(Mstar)
        
        finalM = np.zeros((5005,5005))
        for i in range(15):
                finalM += 9*np.outer((M[i]-Mstar),(M[i]-Mstar).T)
        print('Done.')

        w_c_mat = N
        b_c_mat = finalM
        print('Compute feature vectors ...')
        feature_vectors = get_feature_vectors(w_c_mat,b_c_mat) # 11155 by 25
        print('Done.')
        return feature_vectors



if __name__=="__main__":
        ###################  LDA ##############################
        X_train, label_train = create_data_set(path["Training"])
        Y_train = [int(x[7:9])-1 for x in label_train]
        X_test, label_test = create_data_set(path["Test"])
        mean, global_mean = compute_mean(X_train,Y_train)
        temp = X_train.reshape((X_train.shape[0],-1))-global_mean
        X_train = temp.reshape((X_train.shape))
        w_c_mat = within_class_matrix(x=X_train,y=Y_train,mean=mean)
        b_c_mat = between_class_matrix(mean=mean, global_mean=global_mean)
        feature_vectors = get_feature_vectors(w_c_mat,b_c_mat) # 11155 by 25
        random_indexes = np.random.randint(low=0,high=X_train.shape[0],size=10)
        reconstructed_images = []
        for i in random_indexes:
                projected_image = np.matmul(X_train[i].reshape((-1)),feature_vectors)
                temp = global_mean.reshape((X_train.shape[1],X_train.shape[2]))+np.matmul(projected_image,feature_vectors.transpose()).reshape((X_train.shape[1],X_train.shape[2]))
                io.imsave("Results_LDA/rec_"+label_train[i],temp)
                io.imsave("Results_LDA/original_"+label_train[i],X_train[i]+global_mean.reshape((X_train.shape[1],X_train.shape[2])))
                reconstructed_images.append(temp)



