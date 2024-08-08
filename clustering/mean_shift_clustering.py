import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# mean shift clustering

def mean_shift_clustering(X, bandwidth, max_iter=300):
    
    #calculate mode centroids
    centroids = X.copy()
    for _ in range(max_iter):
        new_centroids = []
        for i in range(len(centroids)):
            points_within_bandwidth = [X[j] for j in range(len(X)) if np.linalg.norm(X[j] - centroids[i] <=bandwidth)]
            new_centroids.append(np.mean(points_within_bandwidth,axis = 0))
        new_centroids = np.array(new_centroids)
        if np.allclose(centroids, new_centroids,atol=1e-3):
            break
        centroids = new_centroids

    #assign labels
    labels = np.zeros(len(X))
    for i in range(len(X)):
        d = np.linalg.norm(X[i] - centroids, axis=1)
        labels[i] = np.argmin(d)
    
    return labels, centroids

def main():
    
    #Generate test data set
    centers = [[1,1], [-1,-1],[1,-1]]
    X, _ = make_blobs(n_samples=500, centers=centers, cluster_std=0.5)
    bandwidth = 2
    #Run clustering
    labels, centroids = mean_shift_clustering(X, bandwidth)

    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, alpha=0.6, marker='x')
    plt.title('Mean Shift Clustering')
    plt.savefig("clustering/output/mean_shift_clustering.png")

if __name__=='__main__':
    main()

    
