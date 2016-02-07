# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import unittest
from Elemento import Elemento
from Cluster import Cluster


class MyTestCase(unittest.TestCase):
    global elemento
    elemento = Elemento([2, 2])
    global cluster
    cluster = Cluster([Elemento([1, 1]), Elemento([1, 3]), Elemento([3, 1]), Elemento([3, 3])])

    # Dimensión del elemento
    def test_dimension_elemento(self):
        self.assertEqual(elemento.dimension, 2)
        self.assertNotEquals(elemento.dimension, 1)

    # Dimensión del cluster
    def test_dimension_cluster(self):
        self.assertEquals(cluster.dimension, 2)
        self.assertNotEquals(cluster.dimension, 3)

    def test_centroide_cluster(self):
        centroide = cluster.centroide
        self.assertEquals(centroide[0], 2)
        self.assertEquals(centroide[1], 2)


if __name__ == '__main__':
    unittest.main()
