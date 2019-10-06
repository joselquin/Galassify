#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:46:23 2019

@author: joselquin
"""

import matplotlib.pyplot as plt

# Función par agraficar el espectro original completo, así como los detalles de las bandas (6480, 6650) y
# (4800, 5050)
def grafOutlier(data_origen, cluster, orden, salva=False):
    filtro1 = [x for x in data_origen.columns[4:] if ((float(x) >= 6480) & (float(x) <= 6650))]
    filtro2 = [x for x in data_origen.columns[4:] if ((float(x) >= 4800) & (float(x) <= 5050))]
    #filtro3 = [x for x in data_origen.columns[4:] if ((float(x) >= 5150) & (float(x) <= 5200))]
    
    print("----------\n")
    print(data_origen.iloc[cluster[orden]][:4])
    
    plt.figure(figsize=(30, 20));
    plt.title("Espectro original", fontsize=30)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    data_origen[data_origen.columns[4:]].iloc[cluster[orden]].plot();
    if salva:
        nombre_archivo = "Espectro_" + str(orden) +  "_1.png"
        plt.savefig(nombre_archivo)
    plt.figure(figsize=(30, 20));
    plt.title("Detalle (6480, 6650)", fontsize=30)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)
    data_origen.filter(filtro1, axis=1).iloc[cluster[orden]].plot();
    if salva:
        nombre_archivo = "Espectro_" + str(orden) +  "_2.png"
        plt.savefig(nombre_archivo)
    plt.figure(figsize=(30, 20));
    plt.title("Detalle (4800, 5050)", fontsize=30)
    plt.xticks(fontsize=15, rotation=90)
    plt.yticks(fontsize=20)
    data_origen.filter(filtro2, axis=1).iloc[cluster[orden]].plot();
    if salva:
        nombre_archivo = "Espectro_" + str(orden) +  "_3.png"
        plt.savefig(nombre_archivo)
