# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Constant
DATASET1 = "../dataSet/DSclustering/DS_3Clusters_999Points.txt"
DATASET2 = "../dataSet/DSclustering/DS2_3Clusters_999Points.txt"
DATASET3 = "../dataSet/DSclustering/DS_5Clusters_10000Points.txt"
DATASET4 = "../dataSet/DSclustering/DS_7Clusters_100000Points.txt"
NUM_CLUSTERS = 3
MAX_ITERATIONS = 10
INITIALIZE_CLUSTERS = ['k-means++', 'random']
CONVERGENCE_TOLERANCE = 0.001
NUM_THREADS = 8
COLORS = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 'brown',
          'cyan', 'magenta']


def dataSet2ListPoints(dirDataSet):
    '''
    Read a txt file with a set of points and return a list of objects Point
    '''
    points = list()
    with open(dirDataSet, 'rt') as reader:
        for point in reader:
            points.append(np.asarray(map(float, point.split("::"))))
    return points


def printResults(centroids, numClusterPoints):
    print '\n\nFINAL RESULT:'
    for i, c in enumerate(centroids):
        print '\tCluster %d' % (i + 1)
        print '\t\tNumber Points in Cluster %d' % numClusterPoints.count(i)
        print '\t\tCentroid: %s' % str(centroids[i])


def plotResults(centroids, numClusterPoints, points):
    plt.plot()
    for nc in range(len(centroids)):
        # plot points
        pointsInCluster = [boolP == nc for boolP in numClusterPoints]
        for i, p in enumerate(pointsInCluster):
            if bool(p):
                plt.plot(points[i][0], points[i][1], linestyle='None',
                         color=COLORS[nc], marker='.')
        # plot centroids
        centroid = centroids[nc]
        plt.plot(centroid[0], centroid[1], 'o', markerfacecolor=COLORS[nc],
                 markeredgecolor='k', markersize=10)
    plt.show()


def kMeans(dataSet, numClusters, maxIterations, initCluster, tolerance,
           numThreads):
    # Read data set
    points = dataSet2ListPoints(dataSet)

    # Object KMeans
    kmeans = KMeans(n_clusters=numClusters, max_iter=maxIterations,
                    init=initCluster, tol=tolerance, n_jobs=numThreads)

    # Calculate Kmeans
    kmeans.fit(points)

    # Obtain centroids and number Cluster of each point
    centroids = kmeans.cluster_centers_
    numClusterPoints = (kmeans.labels_).tolist()

    # Print final result
    printResults(centroids, numClusterPoints)

    # Plot Final results
    plotResults(centroids, numClusterPoints, points)


if __name__ == '__main__':
    kMeans(DATASET1, NUM_CLUSTERS, MAX_ITERATIONS, INITIALIZE_CLUSTERS[0],
           CONVERGENCE_TOLERANCE, NUM_THREADS)
