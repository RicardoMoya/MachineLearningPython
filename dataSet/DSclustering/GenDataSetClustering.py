# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

from sklearn.datasets.samples_generator import make_blobs

THEORETICAL_CENTROIDS = [[-1, 3], [0, 0], [0, 6], [2, 3], [4, 5], [5, 2],
                         [7, 5]]
NUM_POINTS = 100000

points, labels = make_blobs(n_samples=NUM_POINTS,
                            n_features=len(THEORETICAL_CENTROIDS[0]),
                            centers=THEORETICAL_CENTROIDS,
                            cluster_std=0.7)

file_dataset = './DS_' + str(len(THEORETICAL_CENTROIDS)) + 'Clusters_' + str(
    NUM_POINTS) + 'Points.txt'
fichero = open(file_dataset, 'w')
for p in points:
    fichero.write('::'.join(['%f' % num for num in p]) + '\n')
fichero.close()
