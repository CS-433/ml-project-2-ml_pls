import scipy.stats as stats
import numpy as np





def t_test_symptome(label,symptome):
    '''
    Check if for a given symptome 
    '''
    k=np.unique(label)
    G=[]
    print(type(k))
    test_total=np.zeros((len(k),len(k)))
    pval_total=test_total
    for i in k:
        index=np.where(label==i)
        G.append(index)
    for i in k:
        for j in range(i+1,len(k)):
            #the ith row of the j col corresond to the t-test between the cluster i and cluster j
            ttest,pval=stats.ttest_ind(symptome[G[i]],symptome[G[j]],equal_var=False)
            test_total[i,j]=ttest
            pval_total[i,j]=pval
            
    return  test_total,pval_total






def symptom_anova(labels,symptom):
    lab = np.unique(labels)
    clusters = []
    for i in lab:
        index = np.where(labels==i)
        clusters.append(index)
    fvalue, pvalue = stats.f_oneway(symptom[clusters[0]],symptom[clusters[1]],symptom[clusters[2]])
    return fvalue,pvalue




def calculate_mean_of_label(data_mean,pred):
    '''Input: data_mean and pred:
        data_mean represents an array which corresponds to the value of a specific
        features of our data that we want to compare according the different prediction we have
    Output: the mean of each class 
    
    '''
    #mean of each label
    mean=[]
    #Number of label
    k=np.unique(pred)

    for i in k:
        mean.append(np.mean(data_mean[np.where(pred==i)]))
    return mean