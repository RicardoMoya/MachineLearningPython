# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import unittest
import numpy as np
import KMeans
from Point import Point
from Cluster import Cluster


class MyTestCase(unittest.TestCase):
    global DATASET
    DATASET = "./dataSet/Datos1.txt"
    global point
    point = Point([2, 2])
    global cluster
    cluster = Cluster([Point([1, 1]), Point([1, 3]), Point([3, 1]), Point([3, 3])])

    # Check point dimension
    def testDimensionPoint(self):
        self.assertEqual(point.dimension, 2)
        self.assertNotEquals(point.dimension, 1)

    # Check cluster dimension
    def testDimensionCluster(self):
        self.assertEquals(cluster.dimension, 2)
        self.assertNotEquals(cluster.dimension, 3)

    # Check centroid calculation
    def testCentroideCluster(self):
        centroid = cluster.centroid
        self.assertEquals(centroid[0], 2)
        self.assertEquals(centroid[1], 2)

    # Check read data set file 
    def testReadFilePoints(self):
        points = KMeans.dataSet2ListPoints(DATASET)
        self.assertTrue(len(points) > 0)
        self.assertTrue(points[0].dimension == 2)

    def testEuclideanDistance(self):
        x = np.array([1, 0])
        y = np.array([0, 0])
        self.assertEquals(KMeans.calculateEuclideanDistance(x, y), 1.0)
        


if __name__ == '__main__':
    unittest.main()
