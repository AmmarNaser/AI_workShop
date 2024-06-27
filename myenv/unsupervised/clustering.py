import numpy as np
import matplotlib.pyplot as plt 

#####------------------(1st Fun)-------------------------#####

def init_centroids(X,k):
    rand_indices = np.random.choice(len(X),k,replace=False)
    centroids = X[rand_indices]
    return centroids

#####----------------------(2nd Fun)---------------------#####

def assign_clusters(X, centroids):
    clusters = []
    for i in X:
        dist =np.linalg.norm(i - centroids, axis = 1)
        closest_centroid = np.argmin(dist)
        clusters.append(closest_centroid)
    return np.array(clusters)

#####---------------------(3rd Fun)----------------------#####

def update_centroids(X, clusters, k):
    new_centroids = []

    for i in range(k):
        cluster_points = X[clusters == i ]
        if len(cluster_points) > 0 :
            new_centroids.append(cluster_points.mean(axis = 0))
        else:
            new_centroids.append(X[np.random.choice(len(X))])
    return np.array(new_centroids)

#####-----------------(4th Fun)--------------------------#####


def plot(X,clusters, centroids,iterations):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1],c=clusters,cmap="viridis" ,marker="o",alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=200, label='Centroids')
    plt.title(f'Iterations: {iterations}')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

#####-----------------(5th Fun)--------------------------#####

def kmean(X,k,max_i=100,plot_progress=False):
    i_centroid = init_centroids(X, k)
    for i in range(max_i):
        a_clusters = assign_clusters(X, i_centroid)
        u_centroids = update_centroids(X, a_clusters, k)
        if np.all(i_centroid == u_centroids):
            break
        i_centroid = u_centroids

        if plot_progress:
            plot(X, a_clusters, u_centroids, i+1)
    if plot_progress:
            plot(X, a_clusters, u_centroids, i+1)

    return a_clusters, u_centroids

#####-----------------(main Fun)--------------------------#####

if __name__ == "__main__" : 
    np.random.seed(42)
    X = np.random.rand(100, 2)
    k = 3
    clusters, centroids = kmean(X,k,plot_progress=True)
    print("Centroids:\n", centroids)
    print("Clusters:\n", clusters)