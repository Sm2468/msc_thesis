# experimental


Please see the 'clustering_experiment' code for all the information about this project. 
 
# Problem Statement
In this Jupyter Notebook, we will apply a K-Means Clustering algorithm hierarchically on the output of a Classification and/or Regression algorithm. 
We will start with a small "proof-of-concept", where we apply K-Means clustering on the test data generated from a Wine dataset.
The purpose of this research is to investigate which errors occur disproportionately more in clusters comprising instances with demographic attributes, such as age, gender and income. This could indicate that the classification/regression algorithm underperforms for these groups, hereby indicating discriminating behaviour.

Furthermore, this study is inspired by the following paper: https://bit.ly/3g2pAmT (the Bias-Aware Hierarchical K-Means Clustering algorithm). 

### 1. Exploratory Data Analysis (EDA)

(+ Applying the K-Means Clustering
The third step is to apply K-Means Clustering on this dataset to eventually group similar errors into clusters. The cluster with the highest error rate will then be further investigated. We will first use the Elbow Method and the Silhouette Coefficient to select the number of clusters. )

### 2. Preprocessing 
We start with creating the dataset through applying a classification or regression algorithm on the data to generate the desired input for the clustering analysis. 
Our input for the clustering algorithm has the following components:
- The test data in a Pandas DataFrame: the instances with their features 
- The ground truth labels (for a classification model) or values (for a regression model)
- The predicted classes/values per instance
- The errors of the model per instance, which can be calculated from (truth label - predicted label)


### 3: Applying the Classification Algorithm  
We apply a simple classification model on the data to obtain the classification errors. Then, we add the errors as attribute to the dataset. Here, we use the RandomForestClassifier algorithm from SciKit. 

### 4: Bias-Aware Hierarchical K-Means 
Finally, we formulate a bias metric to calculate the bias per cluster and to identify the cluster(s) with the highest bias. Then, we will adapt this bias detection algorithm to further identify bias in the clusters.

### 5. Classification on Biased Clusters
