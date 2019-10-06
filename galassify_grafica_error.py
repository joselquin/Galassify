#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:40:53 2019

@author: joselquin
"""

import matplotlib.pyplot as plt

# Dibujaremos el espectro de entrada, el de salida y el error entre ambos
# - original es el dataset de espectros de entrada
# - salida es la matriz np con los espectros de salida del autoencoder
def grafica_error(original, entrada, salida, i, espectros_salida):
    plt.figure(figsize=(20, 10))
    original.iloc[i].plot();
    plt.ylim(top=1, bottom=0);
    plt.figure(figsize=(20, 10))
    plt.plot(salida[i]);
    plt.ylim(top=1, bottom=0);
    plt.xlim(left=0, right=len(espectros_salida[i]));
    plt.figure(figsize=(20,5))
    plt.ylim(top=1, bottom=-1);
    (entrada.iloc[i]-salida[i]).plot();