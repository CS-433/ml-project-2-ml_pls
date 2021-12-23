
import numpy as np
from mlinsights.mlmodel import KMeansL1L2
from sklearn import mixture
import sklearn as sklearn






def stability(nb_iter,nb_cl,X,method):
    '''
    This function wants to check the stability of a method of clustering, meaning how many points change of cluster 
    at each call of the method. 
    To do so, we call the method two times and check how many point stay in the same cluster. We iterate the following 
    for a large number. Since, the label may be changed at between two models (which are in the same group), we do an other function which give us the mapping between labels.
    
    Imput  : The number of iterations nb_iteration, the method of clustering = method, the number of clusters = nb_cl 
           and the matrix X to apply the clustering method
    
    Output : How many points did not change clusters for each iteration. 
    '''
    f=[]
    index_cl_change=[]
    for j in range(nb_iter):
        a=0
        if method=='EM' :
            model1=Gaussian_Mixture(nb_cl,X,0.21)
            pred1=model1.predict(X)
            model2=Gaussian_Mixture(nb_cl,X,0.21)
            pred2=model2.predict(X)
        if method=='Kmeans_L1' :
            Kmeans_L1 = KMeansL1L2(nb_cl, norm = 'L1')
            model1 = Kmeans_L1.fit(X)
            pred1 = model1.predict(X)
            model2 = Kmeans_L1.fit(X)
            pred2 = model2.predict(X)

                
        true_label=label_mapping(model1,model2,method)
        k=np.unique(true_label)
        index_predlab=[]
        for label in true_label :
            index_predlab.append(np.where(pred2==label))
        
        for label,i in enumerate(k) :
            pred2[index_predlab[int(i)]]=label*np.ones((len(index_predlab[int(i)]),1))
        a=np.count_nonzero(pred2-pred1)
        
        f.append(a)
        if a>=6 :
            index_cl_change=np.where(pred2-pred1!=0)
        
    return f,index_cl_change,pred1


#gives mapping from the train set to the complete data

def label_mapping(model1,model2,method):
    if method=='EM':
        means1=model1.means_ # centroids with complete data
        means2=model2.means_ # centroids with train set
    else :
        means1=model1.cluster_centers_
        means2=model2.cluster_centers_
    nb_centroids = len(means1)  
    error_matrix=np.zeros([nb_centroids,nb_centroids])
    for index1,centroid1 in enumerate(means1):
        for index2,centroid2 in enumerate(means2):
            if method=='EM':
                error_matrix[index1,index2]=np.linalg.norm(centroid1-centroid2)
            if method=='Kmeans_L1' :
                error_matrix[index1,index2]=np.linalg.norm(centroid1-centroid2,ord=1)

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
   
    
def Gaussian_Mixture(nb_cl,X,thresh):
    sil=0
    while (sil<thresh):
        clf = mixture.GaussianMixture(n_components=nb_cl, covariance_type="full").fit(X)
        pred=clf.predict(X)
        sil=sklearn.metrics.silhouette_score(X, pred)
    return clf