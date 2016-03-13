# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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


def getProbabilityCluster(point, cluster):
    '''
    Calculate the probability that the point belongs to the Cluster
    :param point:
    :param cluster:
    :return: probability = prob * SUM(e ^ (-1/2 * ((x(i) - mean)^2 / std(i)^2 )) / std(i))
    '''
    mean = cluster.mean
    std = cluster.std
    prob = 1.0
    for i in range(point.dimension):
        prob *= (math.exp(-0.5 * (
            math.pow((point.coordinates[i] - mean[i]), 2) / math.pow(std[i],
                                                                     2))) /
                 std[i])

    return cluster.clusterProbability * prob


def getExpecationCluster(clusters, point):
    '''
    Returns the Cluster that has the highest probability of belonging to it
    :param clusters:
    :param point:
    :return: argmax (probability clusters)
    '''
    expectation = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        expectation[i] = getProbabilityCluster(point, c)

    return np.argmax(expectation)


def printClustersStatus(itCounter, clusters):
    print '\nITERATION %d' % itCounter
    for i, c in enumerate(clusters):
        print '\tCluster %d: Probability = %s; Mean = %s; Std = %s;' % (
            i + 1, str(c.clusterProbability), str(c.mean), str(c.std))


def printResults(clusters):
    print '\n\nFINAL RESULT:'
    for i, c in enumerate(clusters):
        print '\tCluster %d' % (i + 1)
        print '\t\tNumber Points in Cluster: %d' % len(c.points)
        print '\t\tProbability: %s' % str(c.clusterProbability)
        print '\t\tMean: %s' % str(c.mean)
        print '\t\tStandard Desviation: %s' % str(c.std)


def plotEllipse(center, points, alpha, color):
    '''
    Plot the Ellipse that defines the area of Cluster
    :param center:
    :param points: points of cluster
    :param alpha:
    :param color:
    :return: Ellipse
    '''

    # Matrix Covariance
    cov = np.cov(points, rowvar=False)

    # eigenvalues and eigenvector of matrix covariance
    eigenvalues, eigenvector = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvector = eigenvector[:, order]

    # Calculate Angle of ellipse
    angle = np.degrees(np.arctan2(*eigenvector[:, 0][::-1]))

    # Calculate with, height
    width, height = 4 * np.sqrt(eigenvalues[order])

    # Ellipse Object
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      alpha=alpha, color=color)

    ax = plt.gca()
    ax.add_artist(ellipse)

    return ellipse


def plotResults(clusters):
    plt.plot()
    for i, c in enumerate(clusters):
        # plot points
        x, y = zip(*[p.coordinates for p in c.points])
        plt.plot(x, y, linestyle='None', color=COLORS[i], marker='.')
        # plot centroids
        plt.plot(c.mean[0], c.mean[1], 'o', color=COLORS[i],
                 markeredgecolor='k', markersize=10)
        # plot area
        plotEllipse(c.mean, [p.coordinates for p in c.points], 0.2, COLORS[i])

    plt.show()


def expectationMaximization(dataSet, numClusters, iterations):
    # Read data set
    points = dataSet2ListPoints(dataSet)

    # Select N points random to initiacize the N Clusters
    initial = random.sample(points, numClusters)

    # Create N initial Clusters
    clusters = [Cluster([p], len(initial)) for p in initial]

    # Inicialize list of lists to save the new points of cluster
    newPointsCluster = [[] for i in range(numClusters)]

    converge = False
    itCounter = 0
    while (not converge) and (itCounter < iterations):
        # Expectation Step
        for p in points:
            iCluster = getExpecationCluster(clusters, p)
            newPointsCluster[iCluster].append(p)

        # Maximization Step
        for i, c in enumerate(clusters):
            c.updateCluster(newPointsCluster[i], len(points))

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
    expectationMaximization(DATASET1, NUM_CLUSTERS, ITERATIONS)
