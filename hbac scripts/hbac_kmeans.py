# Loading the libraries
from hbac_utils import *
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
import random
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import warnings
warnings.filterwarnings('ignore')


def hbac_kmeans(data, max_iter=30, show_plot=True):
    clus_model_kwargs = {
        "n_clusters": 2,
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
    }

    full_data = initialize_dataset(data)

    
    x = 0 # initial cluster number
    initial_bias = 0
    if show_plot:
        pca_plot(full_data, 'HBAC-KMeans', 0.6)
    variance_list = []
    average_accuracy = accuracy(full_data)
    minimal_splittable_cluster_size = round(0.05 * len(full_data))
    minimal_acceptable_cluster_size = round(0.03 * len(full_data))
    print("average_accuracy is: ", average_accuracy) 

    for i in range(1, max_iter):
        if i != 1:
            variance_list.append(calculate_variance(full_data))
        full_data['new_clusters'] = -1
        candidate_cluster = full_data.loc[full_data['clusters'] == x]

        if len(candidate_cluster) < minimal_splittable_cluster_size:
            x = get_random_cluster(full_data['clusters'])
            continue
        
        # Apply KMeans 
        kmeans_algo = KMeans(**clus_model_kwargs).fit(candidate_cluster.drop(['clusters', 'new_clusters', 'predicted_class', 'true_class', 'errors'], axis=1))
        
        candidate_cluster['new_clusters'] = pd.DataFrame(kmeans_algo.predict(candidate_cluster.drop(['clusters', 'new_clusters', 'predicted_class', 'true_class', 'errors'], axis=1)),index=candidate_cluster.index) 
        full_data['new_clusters'] = candidate_cluster['new_clusters'].combine_first(full_data['new_clusters'])

        max_discr_bias = get_max_negative_bias(full_data) # was get_max_abs_bias, but now it only finds the discriminated clusters
        min_new_size = get_min_cluster_size(full_data)
        
        if (max_discr_bias <= initial_bias) & (min_new_size > minimal_acceptable_cluster_size): #abs: >
            # Add new cluster
            print('adding a new cluster')
            n_cluster = max(full_data['clusters'])
            full_data['clusters'][full_data['new_clusters'] == 1] =  n_cluster + 1
            
            if show_plot:
                pca_plot(full_data, 'HBAC-KMeans', 0.6)
            x = get_next_cluster(full_data)
            initial_bias = max_discr_bias
        else:
            x = get_random_cluster(full_data['clusters'])
            
    print('MAX_ITER')
    print(variance_list)
    return full_data