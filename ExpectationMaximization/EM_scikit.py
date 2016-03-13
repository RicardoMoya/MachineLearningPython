# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GMM

# Constant
DATASET1 = "../dataSet/DSclustering/DS_3Clusters_999Points.txt"
DATASET2 = "../dataSet/DSclustering/DS2_3Clusters_999Points.txt"
DATASET3 = "../dataSet/DSclustering/DS_5Clusters_10000Points.txt"
DATASET4 = "../dataSet/DSclustering/DS_7Clusters_100000Points.txt"
NUM_CLUSTERS = 3
MAX_ITERATIONS = 10
CONVERGENCE_TOLERANCE = 0.001
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


def printResults(meansClusters, probabilityClusters, labelClusterPoints):
    print '\n\nFINAL RESULT:'
    for i, c in enumerate(meansClusters):
        print '\tCluster %d' % (i + 1)
        print '\t\tNumber Points in Cluster %d' % labelClusterPoints.count(i)
        print '\t\tCentroid: %s' % str(meansClusters[i])
        print '\t\tProbability: %02f%%' % (probabilityClusters[i] * 100)


def plotEllipse(center, covariance, alpha, color):
    '''
    Plot the Ellipse that defines the area of Cluster
    :param center:
    :param covariance: covariance matrix
    :param alpha:
    :param color:
    :return: Ellipse
    '''
    # eigenvalues and eigenvector of matrix covariance
    eigenvalues, eigenvector = np.linalg.eigh(covariance)
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


def plotResults(points, meansClusters, labelClusterPoints,
                covarsMatrixClusters):
    plt.plot()
    for nc in range(len(meansClusters)):
        # Plot points in cluster
        pointsCluster = list()
        for i, p in enumerate(labelClusterPoints):
            if p == nc:
                plt.plot(points[i][0], points[i][1], linestyle='None',
                         color=COLORS[nc], marker='.')
                pointsCluster.append(points[i])
        # Plot mean
        mean = meansClusters[nc]
        plt.plot(mean[0], mean[1], 'o', markerfacecolor=COLORS[nc],
                 markeredgecolor='k', markersize=10)

        # Plot Ellipse
        plotEllipse(mean, covarsMatrixClusters[nc], 0.2, COLORS[nc])

    plt.show()


def expectationMaximization(dataSet, numClusters, tolerance, maxIterations):
    # Read data set
    points = dataSet2ListPoints(dataSet)

    # Object GMM
    gmm = GMM(n_components=numClusters, covariance_type='full', tol=tolerance,
              n_init=maxIterations, params='wmc')

    # Estimate Model (params='wmc'). Calculate, w=weights, m=mean, c=covars
    gmm.fit(points)

    # Predict Cluster of each point
    labelClusterPoints = gmm.predict(points)

    meansClusters = gmm.means_
    probabilityClusters = gmm.weights_
    covarsMatrixClusters = gmm.covars_

    # Print final result
    printResults(meansClusters, probabilityClusters,
                 labelClusterPoints.tolist())

    # Plot Final results
    plotResults(points, meansClusters, labelClusterPoints, covarsMatrixClusters)


if __name__ == '__main__':
    expectationMaximization(DATASET1, NUM_CLUSTERS, CONVERGENCE_TOLERANCE,
                            MAX_ITERATIONS)
