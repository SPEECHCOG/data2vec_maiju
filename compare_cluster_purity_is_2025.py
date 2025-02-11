# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html



"""


import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, BisectingKMeans#, HDBSCAN
from scipy.spatial.distance import chebyshev
from sklearn_extra.cluster import CLARA#, KMedoids

if sys.platform == 'win32':
    # This is to avoid an error with MiniBatchKMeans on Windows
    os.environ["OMP_NUM_THREADS"] = '1'



def normalize_features(feats) -> np.ndarray:
    # Z-score normalization (zero mean, unit standard deviation)
    
    normalized = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    
    # Remove NaN values by converting them to zero
    normalized = np.nan_to_num(normalized)
    
    return normalized



def compute_cluster_purity(labels_clusters, ignore_single_sample_clusters=False):
    total_points = 0
    purity_sum = 0

    for cluster in labels_clusters:
        if ignore_single_sample_clusters and len(cluster) == 1:
            continue
        
        total_points += len(cluster)
        
        # Count the occurrences of each label in the cluster
        label_counts = {}
        for label in cluster:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # Find the most common label count
        try:
            max_label_count = max(label_counts.values())
        except ValueError:
            continue
        
        # Add the count of the most common label to the purity sum
        purity_sum += max_label_count

    # Calculate the overall purity
    purity = purity_sum / total_points if total_points > 0 else 0
    
    return purity



def compute_chebyshev_distance_matrix(X):
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = chebyshev(X[i], X[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            
    return distance_matrix



if __name__ == '__main__':
    
    #############################################################################################
    ################# These are the configuration settings for the experiments ##################
    #############################################################################################
    
    experiment_num = 1
    
    corpus = 'finnish_ser_dataset'
    
    label_type = 'discrete'
    
    # A flag for defining whether we want to normalize our data or not before model training
    norm_feats = True
    
    data_dir = './SER_paper_datasets'
    
    # The random seeds for the experiments
    random_seeds = [24, 48, 96, 52, 28]
    
    # The number of clusters
    num_clusters_list = np.array(list(range(50, 260, 10)) + [500, 750, 1000, 1500])
    
    # The different types of features that we will be using in the experiments
    feat_types = ['clustering', 'prosodic', 'exhubert', 'finnsentiment']
    
    # The directory and the name of the text file where the experiment results are saved
    results_save_dir = f'./clustering_results_{corpus}'
    name_of_result_textfile = f'clustering_test_{experiment_num}.txt'
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
    
    #############################################################################################
    ################################## We initialize our data ###################################
    #############################################################################################
    
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
    
    # All annotated samples (without any data splitting)
    valence_all = pd.read_csv(os.path.join(data_dir, f'{label_type}_valence_annotation_df.csv'))
    valence_all = valence_all.set_index("Unnamed: 0")
    arousal_all = pd.read_csv(os.path.join(data_dir, f'{label_type}_arousal_annotation_df.csv'))
    arousal_all = arousal_all.set_index("Unnamed: 0")
    
    # Get the indices of all items in the valence and arousal dataframes where annotation_mode is not 'nan'
    indices_valence_annotated = valence_all[valence_all['annotation_mode'].notna()].index.tolist()
    indices_arousal_annotated = arousal_all[arousal_all['annotation_mode'].notna()].index.tolist()
    
    # Just in case, we check that the indices are the same
    if not (np.array(indices_valence_annotated) == np.array(indices_arousal_annotated)).all():
        sys.exit('Lists are not equal!')
    
    # Using these indices, extract only these items from the dataframe
    valence_filtered = valence_all.loc[indices_valence_annotated]
    del valence_all
    
    # Same for arousal
    arousal_filtered = arousal_all.loc[indices_arousal_annotated]
    del arousal_all
    
    # We extract only the "annotation_mode" labels and convert the labels to Numpy
    valence_labels = valence_filtered['annotation_mode'].to_numpy()
    arousal_labels = arousal_filtered['annotation_mode'].to_numpy()
    
    # We extract the clustering features and the raw features for our training and testing data
    # Note: Clustering features are normalized!
    clustering_features_df = pd.read_csv(os.path.join(data_dir, 'clustering_features_df.csv'))
    raw_features_df = pd.read_csv(os.path.join(data_dir, 'raw_features_df.csv'))
    
    clustering_features_df_filtered = clustering_features_df.loc[indices_valence_annotated]
    clustering_feats = clustering_features_df_filtered.to_numpy()
    del clustering_features_df
    del clustering_features_df_filtered
    
    raw_features_df_filtered = raw_features_df.loc[indices_valence_annotated]
    
    # We take separately the eGeMAPS, ExHuBERT, and FinnSentiment features for the 12k annotated samples
    # Note: All are unnormalized! Prosodic features can be anything from approx. -100k to 100k, but
    # ExHuBERT and FinnSentiment features are both already in the range from 0.0 to 1.0 (since they are
    # model posteriors).
    prosodic_feats_column_names = raw_features_df.iloc[:, 1:89].columns.to_list()
    exhubert_column_names = raw_features_df.iloc[:, 103:109].columns.to_list()
    finnsentiment_column_names = raw_features_df.iloc[:, 109:].columns.to_list()
    del raw_features_df
    
    prosodic_feats = raw_features_df_filtered[prosodic_feats_column_names].to_numpy()
    exhubert_feats = raw_features_df_filtered[exhubert_column_names].to_numpy()
    finnsentiment_feats = raw_features_df_filtered[finnsentiment_column_names].to_numpy()
    
    if norm_feats:
        prosodic_feats = normalize_features(prosodic_feats)
        exhubert_feats = normalize_features(exhubert_feats)
        finnsentiment_feats = normalize_features(finnsentiment_feats)
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
    
    # Initialize the result text file
    file = open(os.path.join(results_save_dir, name_of_result_textfile), 'w')
    file.close()
    
    for feat_type in feat_types:
        if feat_type == 'clustering':
            features = clustering_feats
        elif feat_type == 'prosodic':
            features = prosodic_feats
        elif feat_type == 'exhubert':
            features = exhubert_feats
        elif feat_type == 'finnsentiment':
            features = finnsentiment_feats
        else:
            sys.exit(f'Feature type {feat_type} not implemented!')
        
        # We pre-compute the distance matrix
        distance_matrix_chebyshev = compute_chebyshev_distance_matrix(features)
        
        # Now we can run the different clustering algorithms
        for random_seed in random_seeds:
            for clustering_algorithm in ['minibatchkmeans', 'agglomerative_euclidean', 'agglomerative_manhattan', 'agglomerative_cosine',
                                         'agglomerative_chebyshev', 'bisectingkmeans', 'clara_euclidean', 'clara_manhattan', 'clara_cosine',
                                         'clara_chebyshev', 'clara_correlation']:
            
                for num_clusters in num_clusters_list:    
            
                    start_time = time.time()
                    
                    if clustering_algorithm == 'minibatchkmeans':
                        clustering = MiniBatchKMeans(n_clusters=num_clusters, random_state=random_seed, init='k-means++', batch_size=1024,
                                                     n_init=1, max_iter=300, tol=0.0001, reassignment_ratio=0.0).fit(features)
                    
                    elif clustering_algorithm == 'agglomerative_euclidean':
                        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward').fit(features)
                    
                    elif clustering_algorithm == 'agglomerative_manhattan':
                        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='manhattan', linkage='average').fit(features)
                    
                    elif clustering_algorithm == 'agglomerative_cosine':
                        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average').fit(features)
                    
                    elif clustering_algorithm == 'agglomerative_chebyshev':
                        clustering = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='average').fit(distance_matrix_chebyshev)
                    
                    elif clustering_algorithm == 'bisectingkmeans':
                        clustering = BisectingKMeans(n_clusters=num_clusters, random_state=random_seed, init='k-means++', algorithm='lloyd',
                                                     n_init=1, max_iter=300, tol=0.0001, bisecting_strategy='biggest_inertia').fit(features)
                    
                    elif clustering_algorithm == 'clara_euclidean':
                        clustering = CLARA(n_clusters=num_clusters, metric='euclidean', random_state=random_seed, max_iter=5, n_sampling_iter=1).fit(features)
                    
                    elif clustering_algorithm == 'clara_manhattan':
                        clustering = CLARA(n_clusters=num_clusters, metric='manhattan', random_state=random_seed, max_iter=5, n_sampling_iter=1).fit(features)
                    
                    elif clustering_algorithm == 'clara_cosine':
                        clustering = CLARA(n_clusters=num_clusters, metric='cosine', random_state=random_seed, max_iter=5, n_sampling_iter=1).fit(features)
                    
                    elif clustering_algorithm == 'clara_chebyshev':
                        clustering = CLARA(n_clusters=num_clusters, metric='chebyshev', random_state=random_seed, max_iter=5, n_sampling_iter=1).fit(features)
                    
                    elif clustering_algorithm == 'clara_correlation':
                        clustering = CLARA(n_clusters=num_clusters, metric='correlation', random_state=random_seed, max_iter=5, n_sampling_iter=1).fit(features)
                    
                    else:
                        sys.exit(f'The clustering algorithm {clustering_algorithm} is not yet implemented!')
                    
                    # Get the cluster labels for each sample
                    cluster_indices = clustering.labels_
                    
                    # Split the features array into clusters
                    clusters = [features[cluster_indices == i] for i in range(num_clusters)]
                    
                    # Split the labels into clusters
                    valence_labels_clusters = [valence_labels[cluster_indices == i] for i in range(num_clusters)]
                    arousal_labels_clusters = [arousal_labels[cluster_indices == i] for i in range(num_clusters)]
                    
                    # Get the sizes of each cluster
                    cluster_sizes = [len(cluster) for cluster in clusters]
                    
                    # Sort labels by cluster size in descending order
                    cluster_size_indices_sorted = np.argsort(cluster_sizes)[::-1]
                    valence_labels_clusters = [valence_labels_clusters[i] for i in cluster_size_indices_sorted]
                    arousal_labels_clusters = [arousal_labels_clusters[i] for i in cluster_size_indices_sorted]
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    with open(os.path.join(results_save_dir, name_of_result_textfile), 'a') as f:
                        f.write(f'Algorithm {clustering_algorithm}, num clusters {num_clusters}: Total duration {total_time} seconds\n')
                    
                    # We compute cluster purity for the given features and the clustering algorithm
                    valence_purity = compute_cluster_purity(valence_labels_clusters)
                    arousal_purity = compute_cluster_purity(arousal_labels_clusters)
                    valence_purity_no_single_sample_clusters = compute_cluster_purity(valence_labels_clusters, ignore_single_sample_clusters=True)
                    arousal_purity_no_single_sample_clusters = compute_cluster_purity(arousal_labels_clusters, ignore_single_sample_clusters=True)
                    
                    with open(os.path.join(results_save_dir, name_of_result_textfile), 'a') as f:
                        f.write(f'Algorithm {clustering_algorithm}, {feat_type} feats, number of clusters: {len(valence_labels_clusters)}\n')
                        f.write(f'Algorithm {clustering_algorithm}, {feat_type} feats, cluster purity (all clusters, valence) {valence_purity}\n')
                        f.write(f'Algorithm {clustering_algorithm}, {feat_type} feats, cluster purity (all clusters, arousal) {arousal_purity}\n')
                        f.write(f'Algorithm {clustering_algorithm}, {feat_type} feats, cluster purity (excluding singleton clusters, valence) {valence_purity_no_single_sample_clusters}\n')
                        f.write(f'Algorithm {clustering_algorithm}, {feat_type} feats, cluster purity (excluding singleton clusters, arousal) {arousal_purity_no_single_sample_clusters}\n\n\n\n')
            
    
    
    
    
    