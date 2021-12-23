# Second Project of the EPFL Machine Learning Course (CS-433)

# General Information
This repository contains the code for project 2 of the Machine Learning course 2021 (CS-433).

## Team
The project is accomplished by team PLSteam with members: 

Leandre Castagna: @Defteggg \
Pascal  Epple   : @epplepascal \
Selima  Jaoua   : @salimajaoua

# Project presentation and methods used

## Presentation : 
The aim of this project is to explore if the heterogeneity of schizophrenia can be tackled by dividing the patients into subgroups using unsupervised learning. A clinical data set where 227 schizophrenia patients performed a battery of perceptual and cognitive tasks has been provided. The purpose is to build clusters in order to construct different classifications and use them to describe differences between groups of patients. To validate the subgroups,  polygenic risk scores and the symptoms subscales are provided in the data set.

## Data set :
The data set is not available as it contains highly confidential information. The computed results and plots are nonetheless present in the provided Jupyter notebook.

## Needed packages in order to run the code:

To run the code, one needs to download the latest version of Python.
Moreover, the following external libraries are used:
- NumPy
- Sklearn
- SciPy
- Pandas

## Clustering methods used: 
Three clustering methods were used: K-Means with the L1 norm, the Expectation Maximization (EM) algorithm and DBSCAN.

## Methods:
Our code is separated into 6 different .py files, with 3 .ipynb files which are consisting of the Main of our project and were all of our results can be seen, DBSCAN and Topological Clustering .
As the data set cannot be shared, the results are not reproducible. Contact us in case you would like to get further informations about some results, we are glad to answer your questions.

We hereby briefly introduce the 6 .py files and explain their utility:

- anova_ttest.py:
Functions to perform ANOVA and Welch's t-test on our data.
- Cross_validation.py:
Implements cross-validation as described in the final report (search for the report.pdf file in the repository for further information)
- data_import.py:
Data management : In this .py, we just create a function which download the data with the symptom and the output of the function returns a list with all the symptom and an other list with the name corresponding to each symptom
- map_label.py:
Reassigns the label of one cluster with the labels of another one. This method is necessary in order to compare two clusters, as they might be differently labeled depending on the initial value we launched the algorithm with.
- score.py:
Computes the Davis Bouldin and silhouette scores of a given cluster.
- Stability_function.py:
Functions to determine how the labelling is affected with respect to the initial value the clustering algorithm is launched with.









