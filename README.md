# Hierarchical Bias-Aware Clustering (HBAC)


# Problem Statement
In this project, we will apply a K-Means Clustering algorithm hierarchically on the output of a classification algorithm to identify clusters with a high bias. The purpose of this research is to investigate which errors occur disproportionately more in clusters comprising instances with demographic attributes, such as age, gender and income. This could indicate that the classification algorithm underperforms, and therefore discriminates against these groups. 



We will start with a small "proof-of-concept", where we apply K-Means clustering on the test data generated from a Wine dataset.

### 1. Exploratory Data Analysis (EDA)
We begin by exploring the raw dataset which will be first fed to a classifier (which takes place at the third step) and then to the clustering algorithm (fourth step). The dataset distribution is investigated as well as visually inspected. 

### 2. Preprocessing 
Then, we preprocess the dataset by handling missing values and transforming categorical features into numerical values. One of the options we investigate is one-hot encoding the categorical features.


### 3: Applying the Classification Algorithm  
We apply a simple classification model on the data to obtain the classification errors. Here, we use the RandomForestClassifier algorithm from SciKit. Then, we add the errors as a new feature to the dataset.
The generated results on the test set will serve as input for step 4. Thus, we export the generated results to a csv-file. 

### 4: Bias-Aware Hierarchical K-Means (BAH-KM)
First, all the features, except for the predictions, ground truth labels and the errors, are scaled. After scaling, the Bias-Aware Hierarchical KMeans algorithm can be applied on the classification results. 

The input for the clustering algorithm has the following components:
- The test data in a Pandas DataFrame: the instances with their features 
- The ground truth labels (for a classification model) or values (for a regression model)
- The predicted class of the classifier per instance
- The errors of the model per instance, which can be calculated from (truth label - predicted label) per instance

We formulate a bias metric to calculate the bias per cluster and to identify the cluster(s) with the highest bias. 
The bias metric is the following:
Bias of Cluster X = Performance Metric(Cluster X) - Performance Metric(all Clusters\Cluster X)

Where the performance metric is Macro F-score. This will later be replaced by the Weighted F-score.

The output of the Bias-Aware algorithm is/are the cluster(s) with the highest bias. This will we exported to a csv-file. 


### 5. Identifying Patterns in Biased Clusters
Several methods will be applied to identify the patterns in the biased clusters. We will inspect which features and which values of the features occurred most. 


## Additional Sources
* The main inspiration behind this study: the Bias-Aware Hierarchical K-Means Clustering proposed by Misztal-Radecka and Indurkhya : https://bit.ly/3g2pAmT.
* An introduction on bias with 


 ![plot test 1](./Images/Bias_Workflow.jpg)