# Machine Learning - Aprendizaje supervisado
# Por Adrian Davila Guerra

# Concesionario

import numpy as np
from sklearn import tree

# Clase: Modelo
# Precio, Tamaño maletero (pequeño: 0, mediano: 1, grande: 2), num. caballos, ABS (0: NO, 1: SÍ), Consumo
atributos = [[12000,0,65,0,4.7],
             [12500,0,80,1,4.9],
             [13000,1,100,1,7.8],
             [14000,2,125,1,6.0],
             [15000,0,147,1,8.5]] 
etiquetas = [0,1,2,3,4] # Cada uno es un modelo distinto

clasificador = tree.DecisionTreeClassifier()
train = clasificador.fit(atributos,etiquetas) 

print ("Introduzca el precio: ", end="") 
precio = input()
print ("Introduzca el tamaño del maletero (pequeño: 0, mediano: 1, grande: 2): ", end="") 
tam_maletero = input()
print ("Introduzca el número de caballos: ", end="") 
caballos = input()
print ("Introduzca si tiene ABS (0: NO, 1: SÍ): ", end="") 
ABS = input()
print ("Introduzca el consumo deseado: ", end="") 
consumo = input()

res = train.predict([[precio,tam_maletero,caballos,ABS,consumo]])

print ("Modelo recomendado"+ str(res) +", consumo máximo:", float(atributos[int(res)][4])*100,"tras recorrer 100 KM.")
