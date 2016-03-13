# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import unittest
import numpy as np
import EM
from Point import Point
from Cluster import Cluster


class MyTestCase(unittest.TestCase):
    global DATASET
    DATASET = "../dataSet/DSclustering/DS_3Clusters_999Points.txt"
    global point
    point = Point(np.array([2, 2]))
    global listPoints
    listPoints = [Point(np.array([1, 1])), Point(np.array([1, 3])),
                  Point(np.array([3, 1])), Point(np.array([3, 3]))]
    global cluster
    cluster = Cluster(listPoints, len(listPoints))

    # Check point dimension

    def testDimensionPoint(self):
        self.assertEqual(point.dimension, 2)
        self.assertNotEquals(point.dimension, 1)

    # Check cluster dimension
    def testDimensionCluster(self):
        self.assertEquals(cluster.dimension, 2)
        self.assertNotEquals(cluster.dimension, 3)

    # Check mean and  calculation
    def testMeanStdCluster(self):
        mean = cluster.mean
        std = cluster.std
        self.assertEquals(mean[0], 2)
        self.assertEquals(mean[1], 2)
        self.assertEquals(std[0] - std[1], 0)

    # Check read data set file
    def testReadFilePoints(self):
        points = EM.dataSet2ListPoints(DATASET)
        self.assertTrue(len(points) > 0)
        self.assertTrue(points[0].dimension == 2)

    # Check probabilityCluster
    def testGetProbabilityCluster(self):
        (point, cluster)
        self.assertEquals(EM.getProbabilityCluster(point, Cluster([point], 1)),
                          1)

    # Check cluster's method
    def testCluster(self):
        cluster = Cluster([point], 1)
        self.assertEquals(cluster.dimension, 2)
        self.assertFalse(cluster.converge)
        np.testing.assert_array_equal(cluster.mean, np.array([2, 2]))
        np.testing.assert_array_equal(cluster.std, np.array([1, 1]))
        self.assertEquals(cluster.clusterProbability, 1)
        cluster.updateCluster(listPoints, 4)
        self.assertEquals(cluster.dimension, 2)
        self.assertTrue(cluster.converge)
        np.testing.assert_array_equal(cluster.mean, np.array([2, 2]))
        self.assertEquals(cluster.std[0] - cluster.std[1], 0)
        self.assertEquals(cluster.clusterProbability, 1)


if __name__ == '__main__':
    unittest.main()
