# Functions to calculate the F-score, mean F-score and the bias of each cluster. 
# Requires with predicted and true class columns, but the errors column is not needed)
# TODO solve the "float division by zero" error
# TODO replace macro F-score by weighted F-score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
def F_score(results, class_number):
    true_pos  = results.loc[(results["true_class"] == class_number) & (results["predicted_class"] == class_number)]
    true_neg  = results.loc[(results["true_class"] != class_number) & (results["predicted_class"] != class_number)]
    false_pos = results.loc[(results["true_class"] != class_number) & (results["predicted_class"] == class_number)]
    false_neg = results.loc[(results["true_class"] == class_number) & (results["predicted_class"] != class_number)]

    try:
        precision = len(true_pos)/(len(true_pos) + len(false_pos))
        recall    = len(true_pos)/(len(true_pos) + len(false_neg))
        f_score   = 2 * ((precision * recall)/(precision + recall))
    except ZeroDivisionError:
        return 0
        
    return f_score

# Calculating the macro average F-score --> will eventually be replaced with weighted F-score
def mean_f_score(results, method="macro"):
    classes = results['true_class'].unique()
    fscores_list = []
    for i in classes:
        class_i = F_score(results, i)
        fscores_list.append(class_i)

#    if (method == "weighted"):
#        mean_f_score = ...
    if (method == "macro"):
        mean_f_score = (sum(fscores_list))/len(classes)

    return(mean_f_score)

# Calculating the bias for each cluster
def calculate_bias(clustered_data, cluster_number):
    cluster_x = clustered_data.loc[clustered_data["clusters"] == cluster_number]
    f_cluster = mean_f_score(cluster_x)
    print("Mean F score of cluster " + str(cluster_number) + "=" + str(f_cluster))
    remaining_clusters = clustered_data.loc[clustered_data["clusters"] != cluster_number]
    f_remain = mean_f_score(remaining_clusters)
    print("Mean F score of remainder " + str(cluster_number) + "=" + str(f_remain))
    
# Bias definition: the lower this is, the higher the bias of cluster x is
    return f_remain - f_cluster



# Receives the data within one cluster to calculate the variance
def calculate_variance(data):
    # Obtain errors column
    errors_col = data['errors']
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(errors_col)/n
    # Squared deviation 
    deviations = [(x - mean) ** 2 for x in errors_col]
    # Variance
    variance = sum(deviations) / n
    return variance

def get_highest_var_cluster(data):
    clusters = data['clusters'].unique()
    highest_variance = 0
    best_cluster = None
    cluster_number = None
    for i in clusters:
        #print('this is i:', i)
        cluster_i = data[data['clusters'] == i]
        variance_cluster = calculate_variance(cluster_i)
        #print('variance cluster:', variance_cluster)
        #print('highest found variance:', highest_variance)

        if variance_cluster > highest_variance:
            highest_variance = variance_cluster
            best_cluster = cluster_i
            cluster_number = i
            print('the cluster with the highest variance:', cluster_number)

    return cluster_number


# plotting the cluster assignments to check whether the clusters make sense
# TODO convert clustering dimensions to PCA to plot on 2-dimensional axis 
def plot_clusters(data):
#     pca = PCA(n_components=2)
#     transformed = pd.DataFrame({"axis1": np.zeros(len(data)), "axis2" : np.zeros(len(data))})
#     transformed["axis1","axis2"] = pca.fit_transform(data.drop("clusters", axis=1), y=None)
#     transformed["clusters"] = data["clusters"]
    scatterplot = sns.scatterplot(data=data, x="Job", y="Age", hue="clusters")
    plt.show()
    
