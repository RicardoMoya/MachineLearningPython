# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from Point import Point
from Cluster import Cluster

DATASET1 = "../dataSet/DSclustering/DS_3Clusters_999Points.txt"
DATASET2 = "../dataSet/DSclustering/DS2_3Clusters_999Points.txt"
DATASET3 = "../dataSet/DSclustering/DS_5Clusters_10000Points.txt"
DATASET4 = "../dataSet/DSclustering/DS_7Clusters_100000Points.txt"
NUM_CLUSTERS = 3
ITERATIONS = 1000
COLORS = ['red', 'blue', 'green', 'yellow', 'gray', 'pink', 'violet', 'brown',
          'cyan', 'magenta']


def dataSet2ListPoints(dirDataSet):
    '''
    Read a txt file with a set of points and return a list of objects Point
    '''
    points = list()
    with open(dirDataSet, 'rt') as reader:
        for point in reader:
            points.append(Point(np.asarray(map(float, point.split("::")))))
    return points


def getNearestCluster(clusters, point):
    '''
    Calculate the nearest cluster
    :param clusters: old clusters
    :param point: point to assign cluster
    :return: index of list cluster
    '''
    dist = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        dist[i] = distance.euclidean(point.coordinates, c.centroid)
    return np.argmin(dist)


def printClustersStatus(itCounter, clusters):
    print '\nITERATION %d' % itCounter
    for i, c in enumerate(clusters):
        print '\tCentroid Cluster %d: %s' % (i + 1, str(c.centroid))


def printResults(clusters):
    print '\n\nFINAL RESULT:'
    for i, c in enumerate(clusters):
        print '\tCluster %d' % (i + 1)
        print '\t\tNumber Points in Cluster %d' % len(c.points)
        print '\t\tCentroid: %s' % str(c.centroid)


def plotResults(clusters):
    plt.plot()
    for i, c in enumerate(clusters):
        # plot points
        x, y = zip(*[p.coordinates for p in c.points])
        plt.plot(x, y, linestyle='None', color=COLORS[i], marker='.')
        # plot centroids
        plt.plot(c.centroid[0], c.centroid[1], 'o', color=COLORS[i],
                 markeredgecolor='k', markersize=10)
    plt.show()


def kMeans(dataSet, numClusters, iterations):
    # Read data set
    points = dataSet2ListPoints(dataSet)

    # Select N points random to initiacize the N Clusters
    initial = random.sample(points, numClusters)

    # Create N initial Clusters
    clusters = [Cluster([p]) for p in initial]

    # Inicialize list of lists to save the new points of cluster
    newPointsCluster = [[] for i in range(numClusters)]

    converge = False
    itCounter = 0
    while (not converge) and (itCounter < iterations):
        # Assign points in nearest centroid
        for p in points:
            iCluster = getNearestCluster(clusters, p)
            newPointsCluster[iCluster].append(p)

        # Set new points in clusters and calculate de new centroids
        for i, c in enumerate(clusters):
            c.updateCluster(newPointsCluster[i])

        # Check that converge all Clusters
        converge = [c.converge for c in clusters].count(False) == 0

        # Increment counter and delete lists of clusters points
        itCounter += 1
        newPointsCluster = [[] for i in range(numClusters)]

        # Print clusters status
        printClustersStatus(itCounter, clusters)

    # Print final result
    printResults(clusters)

    # Plot Final results
    plotResults(clusters)


if __name__ == '__main__':
    kMeans(DATASET1, NUM_CLUSTERS, ITERATIONS)
