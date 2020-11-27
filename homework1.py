#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:19:41 2020

@author: lindaludovisi


"""

from sklearn.datasets import load_wine
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#import seaborn as sns



#
#sklearn_to_dataframe()
#
#This function transforms a sklearn dataset into a pandas dataframe
#The input parameter is the original sklearn dataset
#This function simply returns a dataframe
def sklearn_to_dataframe(data):
    dataframe = pd.DataFrame(data.data, columns=data.feature_names)
    dataframe['target'] = pd.Series(data.target)
    return dataframe
    


#
#plot_classifier()
#
#This function plots the boundaries of a classifier   
def plot_classifier(clf, X, y):
    h = .02  # step size in the mesh
    #color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #x_min, x_max = X.alcohol.min() - 1 , X.alcohol.max() + 1
    #y_min, y_max = X.malic_acid.min() - 1 , X.malic_acid.max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.pcolormesh(xx, yy, Z , cmap=cmap_light) #light 

    # Plot also the data points 
    y_list = y["target"].tolist()
    np.array(y_list)
    
    plt.scatter(X[:, 0] , X[:, 1] , c=y_list, cmap=cmap_bold, #bold
                edgecolor='k', s=20)
    #plt.scatter(X.alcohol , X.malic_acid , c=y_list, cmap=cmap_bold, #bold
    #            edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    return 0
  
    

def main():
    # 1. Load the wine dataset as a dataframe
    data = load_wine()
    raw_df = sklearn_to_dataframe(data)
    raw_df.info()
        
    # 2. Select the first two attributes for a 2D representation of the image
    df = raw_df.iloc[:,:2]
    target = raw_df[["target"]]
    
    # 3. Randomly split data into train, validation and test sets in 
    #    proportion 5:2:3
    X = df
    y = target
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_train_raw, y_train, test_size=2/7, 
                                                        random_state= 42)  
    
    
    #create an instance of StandardScaler 
    scaler = StandardScaler()
    #train the scaler on X_train
    scaler.fit(X_train_raw)
    #transform X_train, X_val, X_test
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    """
        K Nearest Neighbors
    """
    
    # 4. now that we have train/validation/test , we perform KNearestNeighbors 
    k_list = [1,3,5,7]
    acc_list = []
    best_acc = 0
   
    for k in k_list :        
        # create an instance of K Nearest Neighbours Classifier and fit the data
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_val)
        
        #accuracy score
        acc = accuracy_score(y_val, y_pred, normalize=True)
        print(f"The accuracy score with k = {k} is {acc}")
        
        #save the accuracy in acc_list
        acc_list.append(acc)
        
        #save the instance of the KNN classifier with the best accuracy
        if (acc > best_acc) :
            best_acc = acc
            best_clf = clf
            best_k = k
            
        #plot it!
        plot_classifier( clf, X_val, y_val)
        plt.title(f"3-Class classification with K= {k}")
        plt.show()

        
    # 5. Plot a graph showing how the accuracy on the validation set varies when changing K
    plt.style.use('seaborn-whitegrid')
    plt.plot(k_list, acc_list, '-ok', color='blue',
             markersize=7, linewidth=3,
             markerfacecolor='white',
             markeredgecolor='darkblue',
             markeredgewidth=2)
    plt.title(f"How accuracy varies when changing K parameter")
    plt.xlabel("K")
    plt.ylabel("accuracy")
    plt.show()
    
    
    # 7. Use the best value of K and evaluate the model on the test set.
    #    How well does it works?
    y_pred_knn = best_clf.predict(X_test)
    #accuracy score
    acc = accuracy_score(y_test, y_pred_knn, normalize=True)
    print(f"The accuracy score performed on the test set is {acc}")
    
    #plot it
    plot_classifier(best_clf, X_test, y_test)
    plt.title(f"3-Class classification with K= {best_k}")
    plt.show()


    """
        Linear SVM
    """
       
    # 8. Perform LinearSVM
    c_list = [0.001, 0.01, 0.1, 1, 10, 100,1000]
    acc_list = []
    best_acc = 0
    for c in c_list :
        svc = svm.LinearSVC( C=c)
        svc.fit(X_train, y_train.values.ravel())
        y_pred = svc.predict(X_val)
        
        #accuracy score
        acc = accuracy_score(y_val, y_pred, normalize=True) 
        print(f"The accuracy score with C = {c} is {acc}")
        
        #save the accuracy in acc_list
        acc_list.append(acc)
        
        #save the instance of the LinearSVC classifier with the best accuracy
        if (acc > best_acc) :
            best_acc = acc
            best_svc = svc
            best_c = c
       
        #plot it
        plot_classifier(svc, X_val, y_val)
        plt.title(f"3-Class classification using LinearSVC with C= {c}")
        plt.show()


    # 9. Plot a graph showing how the accuracy on the validation set varies when changing C
    plt.style.use('seaborn-whitegrid')
    num= [1,2,3,4,5,6,7]
    plt.plot( num, acc_list, '-ok', color='blue',
             markersize=7, linewidth=3,
             markerfacecolor='white',
             markeredgecolor='darkblue',
             markeredgewidth=2)
    plt.xticks(np.arange(1,8), ['0.001', '0.01', '0.1', '1', '10', '100','1000'] )
    plt.title(f"How accuracy varies when changing C parameter")
    plt.xlabel("C")
    plt.ylabel("accuracy")
    plt.show()

    # 11. Use the best value of C and evaluate the model on the test set.
    #     How well does it works?
    y_pred_svc = best_svc.predict(X_test)
    #accuracy score
    acc = accuracy_score(y_test, y_pred_svc, normalize=True)
    print(f"The accuracy score performed on the test set is {acc}")
    
    #plot it
    plot_classifier(best_svc, X_test, y_test)
    plt.title(f"3-Class classification using LinearSVC with C= {best_c}")
    plt.show()


    """
        SVM with RBF Kernel
    """

    # 12. Perform SVM with RBF kernel
    acc_list = []
    c_list = [0.001, 0.01, 0.1, 1, 10, 100,1000]
    best_acc = 0
    for c in c_list :
        rbf = svm.SVC(kernel = 'rbf', C=c)
        rbf.fit(X_train, y_train.values.ravel())
        y_pred = rbf.predict(X_val)
        
        #accuracy score
        acc = accuracy_score(y_val, y_pred, normalize=True) 
        print(f"The accuracy score with C = {c} is {acc}")
        
        #save the accuracy in acc_list
        acc_list.append(acc)
                
        #save the instance of the SVC(kernel = 'rbf') classifier with the best accuracy
        if (acc > best_acc) :
            best_acc = acc
            best_rbf = rbf
            best_c = c
       
        #plot it
        plot_classifier(rbf, X_val, y_val)
        plt.title(f"3-Class classification using SVC with RBF kernel with C= {c}")
        plt.show()

    #plot accuracy with different C 
    plt.style.use('seaborn-whitegrid')
    num= [1,2,3,4,5,6,7]
    plt.plot( num, acc_list, '-ok', color='blue',
             markersize=7, linewidth=3,
             markerfacecolor='white',
             markeredgecolor='darkblue',
             markeredgewidth=2)
    plt.xticks(np.arange(1,8), ['0.001', '0.01', '0.1', '1', '10', '100','1000'] )
    plt.title(f"How accuracy varies when changing C parameter")
    plt.xlabel("C")
    plt.ylabel("accuracy")
    plt.show()
    
    # 13. Use the best value of C and evaluate the model on the test set.
    #     How well does it works?
    y_pred_rbf = best_rbf.predict(X_test)

    acc = accuracy_score(y_test, y_pred_rbf, normalize=True)
    print(f"The accuracy score performed on the test set is {acc}")
    
    #plot it
    plot_classifier(best_rbf, X_test, y_test)
    plt.title(f"3-Class classification using SVC with RBF kernel with C= {best_c}")
    plt.show()

    # 15. Perform a grid search of the best parameters for an RBF kernel: we will
    # now tune both gamma and C at the same time. Select an appropriate
    # range for both parameters. Train the model and score it on the validation
    # set. Evaluate the best parameters on the test set. Plot the decision
    # boundaries.
    
    gamma_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    best_acc = 0 
    for c in c_list:
        for g in gamma_list:
            rbf_gamma = svm.SVC(kernel = 'rbf', C = c, gamma = g)
            rbf_gamma.fit(X_train, y_train.values.ravel())
            y_pred = rbf_gamma.predict(X_val)
            
            #accuracy score
            acc = accuracy_score(y_val, y_pred, normalize=True) 
            print(f"The accuracy score with C = {c} and gamma = {g} is {acc}")
        
            if (acc > best_acc) :
                best_acc = acc
                best_rbf_gamma = rbf_gamma
                best_c = c
                best_gamma = g
    
    #evaluate the model on the test set.
    y_pred_rbf_gamma = best_rbf_gamma.predict(X_test)

    acc = accuracy_score(y_test, y_pred_rbf_gamma, normalize=True)
    print(f"The accuracy score performed on the test set is {acc}")
    
    #plot it
    plot_classifier(best_rbf_gamma, X_test, y_test)
    plt.title(f"3-Class classification using SVC with RBF kernel with C = {best_c} and gamma = {best_gamma}")
    plt.show()
    
    
    
    # 16. Merge the training and validation split. You should now have 70% training
    #   and 30% test data.
    
    frames= [X_train_raw, X_val_raw]
    lab = [y_train, y_val]
    
    X_train_raw = pd.concat(frames)
    y_train = pd.concat(lab)
    
    #fit the Scaler on the "new" X_train_raw
    scaler.fit(X_train_raw)    
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 17. Repeat the grid search for gamma and C but this time perform 5-fold validation.   
    grid_svc = {'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                     'C': [0.001, 0.01, 0.1, 1, 10, 100,1000]} 
   
    gsCV = GridSearchCV(svm.SVC(), grid_svc, cv=5, verbose=3, n_jobs=-1)
    gsCV.fit(X_train, y_train.values.ravel())

    print()
    print(f"Best parameters set found on development set: {gsCV.best_params_}")
    print()
    print("The best parameters are %s with a score of %0.2f"
      % (gsCV.best_params_, gsCV.best_score_))


    # 18. Evaluate the parameters on the test set. Is the final score different? 
    y_pred_grid = gsCV.best_estimator_.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred_grid, normalize=True)
    print(f"The accuracy score performed on the test set is {acc}")
    
    #plot it
    plot_classifier( gsCV.best_estimator_ , X_test, y_test)
    plt.title(f"3-Class classification using SVC with RBF kernel with parameters : {gsCV.best_params_}")
    plt.show()
      
    
    


if __name__ == "__main__" :
    main()
    




