#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Sun Oct  6 19:32:07 2019

@author: joselquin
"""

import pandas as pd
from tensorflow.keras.backend import backend
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import optimizers, backend
import galassify_grafica_error

# Función para convertir un dataFrame de Pandas en dataset de Tensorflow
#

def df_to_dataset(dataframe, shuffle=False, batch_size=32):
  dataframe = dataframe.copy()
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


# Función para entrenar los dos modelos requeriodos: el autoencoder, que es el modelo completo, y el encoder, que
# general el espectro latente. Tiene bastantes posibilidades de configuración para automatizar procesos
# - entrada: el dataset de entrada, ya escalado
# - dim latente, el número de dimensiones del espacio latente. Por defecto 250
# - epochs - ciclos del entrenamiento, 125 por defecto
# - lr - learning rate. El optimizador será Adadelta, por lo que irá reduciéndose de forma adaptativa
#        con un factor rho de 0.95
# - grabar, por se deseamos grabar los modelos resultantes en un archivo h5 o no
# - cargar, por si deseamos cargar un modelo grabado previamente en archivo h5 o no (si cargamos, el modelo no
#        se entrenará).
# - Si hemos dado a grabar o a cargar el valor True, hay que especificar el nombre de los archivos h5 para ambos
#        modelor (autoencoder y encoder)
# - grafica, por si deseamos imprimir la gráfica de errores durante el entrenamiento o no
def autoencoder_stacked(entrada, train_set, test_set, dim_latente = 250, epochs = 125, lr = 0.30,
                grabar = False, cargar = False, archivo_encoder = " ", 
                archivo_autoencoder = " ", grafica = False):

    if cargar:
        encoder = load_model(archivo_encoder)
        autoencoder_deep = load_model(archivo_autoencoder)
    else:    
        dim_input = entrada.shape[1]
        adadelta = optimizers.Adadelta(lr=lr, rho=0.95)
        input = Input(shape=((dim_input, )))
        encoded = Dense(2000, activation='selu')(input)
        encoded = Dense(1000, activation='selu')(encoded)
        encoded = Dense(dim_latente, activation='selu')(encoded)
        decoded = Dense(1000, activation='selu')(encoded)
        decoded = Dense(2000, activation='selu')(decoded)
        decoded = Dense(dim_input, activation='sigmoid')(decoded)

        autoencoder_deep = Model(input, decoded)
        encoder = Model(input, encoded)
    
        autoencoder_deep.compile(optimizer=adadelta, loss='binary_crossentropy', metrics=["accuracy"])
        autoencoder_deep.fit(train_set, train_set,
                             epochs=epochs,
                             batch_size=256,
                             shuffle=True,
                             validation_data=(test_set, test_set))  
        if grafica:
            galassify_grafica_error.grafica_loss(autoencoder_deep, epochs)
        if grabar:
            autoencoder_deep.save(archivo_autoencoder)
            encoder.save(archivo_encoder)
    
    return encoder, autoencoder_deep