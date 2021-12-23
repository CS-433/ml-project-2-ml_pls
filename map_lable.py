import numpy as np



def label_mapping(method1,method2):

    means1=method1.means_ # centroids with complete data
    means2=method2.means_ # centroids with train set
    nb_centroids = len(means1)  
    error_matrix=np.zeros([nb_centroids,nb_centroids])
    for index1,centroid1 in enumerate(means1):
        for index2,centroid2 in enumerate(means2):
            error_matrix[index1,index2]=np.linalg.norm(centroid1-centroid2)
    filler = 2*np.max(error_matrix)
    #print("error matrix")
    #print(error_matrix)
    true_label = np.zeros([1,nb_centroids])
    for i in range(nb_centroids):
        min = np.min(error_matrix)
        ind = np.where(error_matrix == min)
        row = ind[0][0]
        col = ind[1][0]
        #print(row,col)
        true_label[0,row] = col
        error_matrix[row,:] = filler
        error_matrix[:,col] = filler
        #print("filling process")
        #print(error_matrix)
    true_label=np.reshape(true_label,(np.shape(true_label)[1],))
    return true_label

