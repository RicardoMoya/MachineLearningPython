# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import random
import numpy as np
from scipy.spatial import distance
from Point import Point
from Cluster import Cluster

DATASET1 = "./dataSet/Datos1.txt"
DATASET2 = "./dataSet/Datos2.txt"
NUMCLUSTERS = 3
ITERATIONS = 1000


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
        convergeAll = [c.converge for c in clusters]
        converge = convergeAll.count(False) == 0

        # Increment counter and delete lists of clusters points
        itCounter += 1
        newPointsCluster = [[] for i in range(numClusters)]

        # Print clusters status
        print '\nITERATION %d' %itCounter
        for c in clusters:
            print '\t%s' %str(c.centroid)

    # Print final result
    print '\n\nFINAL RESULT:'
    for i,c in enumerate(clusters):
        print '\tCluster %d' %(i+1)
        print '\t\tNum Points in Cluster %d' %len(c.points)
        print '\t\tCentroid: %s' %str(c.centroid)


if __name__ == '__main__':
    kMeans(DATASET1, NUMCLUSTERS, ITERATIONS)
