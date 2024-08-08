import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class k_means_clustering:
    def __init__(self, tol, max_iter):
        self.tol = tol
        self.max_iter = max_iter

    def generate_centroids(self, X,k):
        idx = np.random.permutation(len(X))
        centroids = X[idx[:k]]
        return centroids
    
    def compute_distances(self, X, centroids):
        distances = np.zeros((X.shape[0],len(centroids)))
        for i in range(len(centroids)):
            distances[:,i] = np.linalg.norm(X - centroids[i],axis = 1)
        return distances
    
    def compute_labels(self, distances):
        labels = np.argmin(distances,axis = 1)
        return labels
    
    def recompute_centroids(self, X, labels, centroids):
        for i in range(len(centroids)):
            centroids[i] = X[labels == i].mean(axis = 0)
        return centroids
    
    def process(self, X, k):
        centroids = self.generate_centroids(X,k)
        for i in range(self.max_iter):
            old_centroids = centroids
            distances = self.compute_distances(X, centroids)
            labels = self.compute_labels(distances)
            centroids = self.recompute_centroids(X, labels, centroids)
            if np.all(np.abs(centroids - old_centroids) < self.tol):
                break
        return labels, centroids
    
    def display(self,centroids, labels, X):
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, alpha=0.6, marker='x')
        plt.title('K means Clustering')
        plt.savefig("clustering/output/k_means_clustering.png")

def main():
    
    #Generate Random data
    X = np.random.rand(200,2)
    k = 5
    
    kmeans = k_means_clustering(1e-3,1000)
    labels, centroids = kmeans.process(X,k)
    print(centroids)
    kmeans.display(centroids,labels,X)

if __name__=='__main__':
    main()