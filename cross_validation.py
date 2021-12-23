from stability_function import*
import numpy as np
from sklearn.model_selection import KFold
from score import*



def cross_validation(X_res1,nb_iter,sil):
    clf1=Gaussian_Mixture(3,X_res1,sil)
    pred=clf1.predict(X_res1)
    cv = KFold(n_splits=8, random_state=3, shuffle=True)

    for i  in range(nb_iter):
        a=0
        index_err=[]    
        for train_index , test_index in cv.split(X_res1):
            #X_train, X_test=X[train_index,:],X[test_index,:]
            X_train, X_test=X_res1[train_index,:],X_res1[test_index,:]
            clf2 =Gaussian_Mixture(3,X_train,sil)            
            #clf2=Gaussian_Mixture(3,X_res1,0.11)

            #model = Kmeans_L1.fit(X_res1)

            pred_test=clf2.predict(X_test)
            pred_train=clf2.predict(X_train)
            pred_all=[]
            pred_all.extend(pred_test)
            pred_all.extend(pred_train)
            pred_all=np.asarray(pred_all)
            true_label=label_mapping(clf1,clf2,'EM')
            pred0=np.where(pred_test==true_label[0])
            pred1=np.where(pred_test==true_label[1])
            pred2=np.where(pred_test==true_label[2])
            pred_test[pred0]=np.zeros((len(pred0),1))
            pred_test[pred1]=np.ones((len(pred1),1))
            pred_test[pred2]=2*np.ones((len(pred2),1))
            a+=np.count_nonzero(pred_test-pred[test_index])
            ind=np.where(pred_test-pred[test_index])
            inde_err=np.asarray(test_index[ind])
            for index in inde_err:
                index_err.append(index)
            #if a>20:
           # break

        print('Total of errors after running for the 8 folds :',a)
    return index_err,pred

