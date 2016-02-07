# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'

import numpy as np


class Cluster:
    '''
    Clase que representa a un cluster formado por uno o más elementos
    '''
    def __init__(self, elementos):
        if len(elementos) == 0:
            raise Exception("Un Cluster no puede tener 0 elementos")
        else:
            self.elementos = elementos
            self.dimension = elementos[0].dimension

        # Comprobamos que todos los elementos del cluster tienen la misma dimensión
        for ele in elementos:
            if ele.dimension != self.dimension:
                raise Exception("El elemento %s tiene dimensión %d distinta a la del resto de elementos %d") % (
                    ele, len(ele), self.dimension)

        # Calculamos el centroide del Cluster
        self.centroide = self.calculoCentroide()

    def calculoCentroide(self):
        '''
        Método que calcula el centroide del Cluster, haciendo la média de cada una de las coordenadas de los elementos
        :return: Centroide del cluster
        '''
        sumCoordenadas = np.zeros(self.dimension)
        for elem in self.elementos:
            for i, x in enumerate(elem.coordenadas):
                sumCoordenadas[i] += x

        return (sumCoordenadas / len(self.elementos)).tolist()
