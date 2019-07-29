# Machine Learning - Aprendizaje supervisado
# Por Adrian Davila Guerra

# ¿Es una manzana o es una naranja?

import numpy as np
from sklearn import tree

# Peso, Textura (0: Rugosa, 1: Lisa)
atributos = [[150,0],[170,0],[140,1],[130,1]] 
etiquetas = [0,0,1,1] # 0: Manzana, 1: Naranja

clasificador = tree.DecisionTreeClassifier()
train = clasificador.fit(atributos,etiquetas) 

print ("Introduzca el peso: ", end="") 
peso = input()
print ("Introduzca la textura (0: Rugosa, 1: Lisa): ", end="") 
textura = input()

res = train.predict([[peso,textura]])

def switch(var):
    switcher = {
        0: "Naranja", 
        1: "Manzana"
    }
    return switcher.get(var,"Tipo de fruta no identificada.")

print ("La fruta indicada es:", switch(int(res)))
