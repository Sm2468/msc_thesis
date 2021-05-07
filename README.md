# General Approach and Current Progress


Please see the 'clustering_experiment' code for all the information about this project. 
 
# Problem Statement
In this Jupyter Notebook, we will apply a K-Means Clustering algorithm hierarchically on the output of a Classification algorithm.
We will start with a small "proof-of-concept", where we apply K-Means clustering on the test data generated from a Wine dataset.
The purpose of this research is to investigate which errors occur disproportionately more in clusters comprising instances with demographic attributes, such as age, gender and income. This could indicate that the classification/regression algorithm underperforms for these groups, hereby indicating discriminating behaviour.

Furthermore, this study is inspired by the following paper: https://bit.ly/3g2pAmT (the Bias-Aware Hierarchical K-Means Clustering algorithm). 

### 1. Exploratory Data Analysis (EDA)
We begin by exploring the raw dataset that will be fed to the classification algorithm at the third step. The dataset distribution is investigated as well as visually inspected. 

### 2. Preprocessing 
Then, we preprocess the dataset by handling missing values and transforming categorical features into numerical values, possibly through one hot encoding.


### 3: Applying the Classification Algorithm  
We apply a simple classification model on the data to obtain the classification errors. Here, we use the RandomForestClassifier algorithm from SciKit. Then, we add the errors as a new feature to the dataset.
The generated results on the test set will serve as input for step 4. Thus, we export the generated results to a csv-file. 


### 4: Bias-Aware Hierarchical K-Means 
The Bias-Aware Hierarchical KMeans algorithm can now be applied on the classification results. 

The input for the clustering algorithm has the following components:
- The test data in a Pandas DataFrame: the instances with their features 
- The ground truth labels (for a classification model) or values (for a regression model)
- The predicted class of the classifier per instance
- The errors of the model per instance, which can be calculated from (truth label - predicted label) per instance

Finally, we formulate a bias metric to calculate the bias per cluster and to identify the cluster(s) with the highest bias. 
The bias metric is the following:
Bias of Cluster X = Performance Metric(Cluster X) - Performance Metric(all Clusters\Cluster X)

Where the performance metric is Macro F-score. This will later be replaced by the Weighted F-score.

The output of the Bias-Aware algorithm is/are the cluster(s) with the highest bias. This will we exported to a csv-file. 


### 5. Identifying Patterns in Biased Clusters
Several methods will be applied to identify the patterns in the biased clusters. We will inspect which features and which values of the features occurred most. 

-----

Updates/open issues:
- 