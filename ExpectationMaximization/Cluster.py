# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np


class Cluster:
    '''
    Class to represent a Cluster: set of points and their parameters (mean,
    standard desviation and probability of belonging to Cluster)
    '''

    def __init__(self, points, totalPoints):
        if len(points) == 0:
            raise Exception("Cluster cannot have 0 Points")
        else:
            self.points = points
            self.dimension = points[0].dimension

        # Check that all elements of the cluster have the same dimension
        for p in points:
            if p.dimension != self.dimension:
                raise Exception(
                    "Point %s has dimension %d different with %d from the rest of points") % (
                          p, len(p), self.dimension)

        # Calculate mean, std and probability
        pointsCoordinates = [p.coordinates for p in self.points]
        self.mean = np.mean(pointsCoordinates, axis=0)
        self.std = np.array([1.0, 1.0])
        self.clusterProbability = len(self.points) / float(totalPoints)
        self.converge = False

    def updateCluster(self, points, totalPoints):
        '''
        Calculate new parameters and check if converge
        :param points: list of new points
        :return: updated cluster
        '''
        oldMean = self.mean
        self.points = points
        pointsCoordinates = [p.coordinates for p in self.points]
        self.mean = np.mean(pointsCoordinates, axis=0)
        self.std = np.std(pointsCoordinates, axis=0, ddof=1)
        self.clusterProbability = len(points) / float(totalPoints)
        self.converge = np.array_equal(oldMean, self.mean)

    def __repr__(self):
        cluster = 'Mean: ' + str(self.mean) + '\nDimension: ' + str(
            self.dimension)
        for p in self.points:
            cluster += '\n' + str(p)

        return cluster + '\n\n'
