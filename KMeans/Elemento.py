# -*- coding: utf-8 -*-
__author__ = 'RicardoMoya'


class Elemento:
    '''
    Clase que representa a un elemento con sus coordenadas
    '''

    def __init__(self, coordenadas):
        self.coordenadas = coordenadas
        self.dimension = len(coordenadas)

    def __repr__(self):
        return str(self.coordenadas)
