import numpy as np
import random
from sklearn.cluster import KMeans

from framework.utils.log import log_message

def run_cluster(prob_array, cluster_num):
    n_class = prob_array.shape[1]

    log_message('cluster', '---Running cluster with n_c = {}...---'.format(cluster_num))
    kmeans = KMeans(n_clusters = cluster_num).fit(prob_array)
    log_message('cluster', 'K-Means stats:')
    np.set_printoptions(precision=4)
    log_message('cluster', 'Num in each cluster: {}'.format(np.bincount(kmeans.labels_)))
    for i in range(0, cluster_num):
        np.set_printoptions(precision=4)
        log_message('cluster', 'Mean prob for cluster {}: {}'.format(i, kmeans.cluster_centers_[i]))

    return kmeans.labels_.astype(np.int32)