import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#https://ourcodeworld.co/articulos/leer/1433/como-corregir-la-advertencia-de-tensorflow-could-not-load-dynamic-library-cudart64-110dll-dlerror-cudart64-110dll-not-found
#
# Formula F = C * 1.8 + 32
# Programación regular
def convertirCaF(centigrados):
    return centigrados * 1.8 +32
# 
#Datos para en entrenamiento
gradosC = np.array([3, 30, 12,1, 22, 14, 50, 20, 10, -3, -10], dtype=float)
gradosF = np.array([37.4, 86, 53.6, 33.8, 71.6, 57.2, 122, 68, 50, 26.6, 14], dtype=float)

#Creamos la red neuronal con el framework Keras (Tipo de red neuronal más básica)
#capa = tf.keras.layers.Dense(units=1,input_shape=[1])

capa_entrada = tf.keras.layers.Dense(units=3, input_shape=[1])
capa_oculta = tf.keras.layers.Dense(units=3)
capa_salida = tf.keras.layers.Dense(units=1)

#Creamos ahora el modelo, escogemos el más sencillo secuencial pasandole la capa como parametro
modelo = tf.keras.Sequential([capa_entrada,capa_oculta,capa_salida])

#Ahora se prepara el modelo para el entrenamiento.

modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss="mean_squared_error")


#Comenzamos el entrenamiento...
historial = modelo.fit(gradosC,gradosF,epochs=1000)

plt.xlabel("Tiempos")
plt.ylabel("Magnitud de perdida(loss)")
plt.plot(historial.history["loss"])
plt.show()

resultado = modelo.predict([38])
#print("Varriables internas:", capa.get_weights())
print("Programación IA", resultado)
print("Programación regular",convertirCaF(38))
