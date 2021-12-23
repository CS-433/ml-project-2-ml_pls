import numpy as np
from sklearn import mixture
import sklearn as sklearn
from mlinsights.mlmodel import KMeansL1L2





#Choose the variables depneding on silhouette and davies_bouldin_score
def variablefamil_select(X,method,Variables,nb, subsets):
    '''
    Imput : the features, the method of clustering, the number of clusters and different subsets of the features
          (represented by the indexes of the features)
    Ouput : We apply the method of clustering on each subset and compute the silhouette score and davies bouldin score 
           to give a score to the clusters that we obtain. 
           return the two scores and the indexes of the subset 
    
    '''
    Criteria=np.zeros((len(subsets),2))
    for k in range(len(subsets)):
        X_res=X[:,subsets[k]] #Takes the subset that we need of features
        if method=="EM":
            clf = mixture.GaussianMixture(n_components=nb, covariance_type="full").fit(X_res)
            pred=clf.predict(X_res)
        if method == "Kmeans_L1":
            Kmeans_L1 = KMeansL1L2(nb, norm = 'L1')
            model = Kmeans_L1.fit(X_res)
            pred = model.predict(X_res)
            
        score=np.array([sklearn.metrics.davies_bouldin_score(X_res,pred ),sklearn.metrics.silhouette_score(X_res, pred)])
        Criteria[k,:]=score 
    return Criteria,subsets



def numbercluster_select(X,method, Variables, nb_max):
    '''
    Aim    : Call the previous function  variablefamil_select to find the optimal number of clusters with the best
             score for each subset of feature.
    Imput  : The X =feature matrix, variable = the index of the 4groups of the feature matrix, 
             method =the method of clustering and the nb_max = number maximal of clusters.
    Output : For each number in the range of nb_max, we output the scores of the method on each subset possible of X
             with index corresponding to the indexes in Variables.
    '''
    scores = []
    subsets=[]
    for i in range(len(Variables[0])):
        for j in range(len(Variables[3])):
            subsets.append(np.array([Variables[0][i],2,3,Variables[3][j]])) #define the indexes of the subsets possible
    for n in range(2,nb_max): 
        score,_ = variablefamil_select(X, method,Variables,n,subsets) #call the function 
        scores.append(score) #add for each number of clusters, the scores of the method
    return scores, subsets 



def findbest_score(score) : 
    ''' 
    This function takes the different score that we obtain and return the best ones. 
    Output : best value for the davis bouldin score 
             best number of clusters  that correspond to the previous value
             best value for the silhouette score
             best number of clusters  that correspond to the previous value
    (tell why we did that ? )
    '''
    best = np.zeros((len(score),2))
    for n in range(len(score)):
        score_best_db=min((score[n])[:,0])
        score_best_sil=max((score[n])[:,1])
        best[n,:]=np.array([score_best_db,score_best_sil])
     
    Best_nb_cluster_db_score=min(best[:,0])
    best_nb_cl_db=np.argmin(best[:,0])
    best_nb_cl_sil=np.argmax(best[:,1])
    Best_nb_cluster_sil_score=max(best[:,1])

    
    return Best_nb_cluster_db_score,Best_nb_cluster_sil_score,best_nb_cl_db,best_nb_cl_sil
